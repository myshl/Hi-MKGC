"""CPU-only mechanism checks; no external weights, datasets or downloads."""
from contextlib import ExitStack
from types import SimpleNamespace
import unittest
from unittest.mock import mock_open, patch

import numpy as np
import torch
from transformers import BertConfig

from models.P_model import KGCPromptTuner
from models.bert_for_layerwise import BertModelForLayerwise


def make_fixture():
    torch.manual_seed(41504)
    config = SimpleNamespace(
        dataset='DB15K', feature_dir='./img_data', n_ent=5, n_rel=2,
        embed_dim=256, pretrained_model='unused-test-model', prompt_length=2,
        prompt_hidden_dim=16, graph_model='conve', n_lar=0, alpha=0.,
        alpha_step=0., label_smoothing=.1, use_fp16=False, gamma=1.,
        num_filt=2, ker_sz=3, k_w=16, k_h=16, bias=False,
        hid_drop=0., hid_drop2=0., feat_drop=0.,
        is_temporal=False, use_log_ranks=False, visualize_embeddings=False,
        lr=5e-4,
    )
    bert = BertConfig(vocab_size=32, hidden_size=32, num_hidden_layers=12,
                      num_attention_heads=4, intermediate_size=64,
                      max_position_embeddings=64, hidden_dropout_prob=0.,
                      attention_probs_dropout_prob=0.)
    random = np.random.RandomState(41504)
    arrays = [random.normal(size=(5, 4096)).astype('float32'),
              random.normal(size=(5, 256)).astype('float32'),
              random.normal(size=(4, 256)).astype('float32')]
    texts = dict(ent_names=['e0', 'e1', 'e2', 'e3', 'e4'],
                 rel_names=['r0', 'r1'], ent_descs=[''] * 5)
    gt = dict(all_tail_gt={(0, 0): [1, 2]}, all_head_gt={(1, 2): [0, 3]})
    with ExitStack() as stack:
        stack.enter_context(patch('models.P_model.open', mock_open(), create=True))
        stack.enter_context(patch('models.P_model.pickle.load', side_effect=arrays))
        stack.enter_context(patch('models.P_model.AutoConfig.from_pretrained', return_value=bert))
        stack.enter_context(patch('models.P_model.BertModelForLayerwise.from_pretrained',
                                  side_effect=lambda *args: BertModelForLayerwise(bert)))
        model = KGCPromptTuner(config, texts, gt)
    queries = torch.tensor([[0, 0], [1, 2]])
    ids = torch.tensor([[1, 2, 3, 4], [1, 5, 6, 4]])
    mask = torch.ones_like(ids)
    return model, queries, ids, mask


class SoftPromptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_prompt_order_and_replacement_at_all_12_layers(self):
        model, queries, ids, mask = make_fixture()
        model.eval()
        visual, structural, layer_inputs = [], [], []
        hooks = [model.img_prompter.register_forward_hook(lambda m, i, o: visual.append(o.detach())),
                 model.gat_prompter.register_forward_hook(lambda m, i, o: structural.append(o.detach()))]
        hooks += [layer.register_forward_pre_hook(lambda m, args: layer_inputs.append(args[0].detach().clone()))
                  for layer in model.plm.encoder.layer]
        try:
            with torch.no_grad():
                logits, _, _, _ = model(queries, ids, mask)
        finally:
            for hook in hooks:
                hook.remove()
        prompts = torch.cat([visual[0], structural[0]], dim=1)
        self.assertEqual(tuple(prompts.shape), (2, 12, 4, 32))
        self.assertEqual(len(layer_inputs), 12)
        for depth, value in enumerate(layer_inputs):
            self.assertEqual(tuple(value.shape), (2, 8, 32))
            torch.testing.assert_close(value[:, :4], prompts[:, depth], rtol=0, atol=0)
        self.assertEqual(tuple(logits.shape), (2, 5))
        self.assertTrue(torch.isfinite(logits).all())

    def test_frozen_backbone_and_both_prompters_receive_gradients(self):
        model, queries, ids, mask = make_fixture()
        batch = dict(ent_rel=queries, source_ids=ids, source_mask=mask,
                     tgt_ent=[1, 0], labels=torch.tensor([1, 0]))
        loss = model.training_step(batch, 0)['loss']
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.plm.parameters()))
        for module in [model.img_prompter, model.gat_prompter]:
            grads = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in grads))
            self.assertGreater(sum(g.abs().sum().item() for g in grads), 0.)
        self.assertFalse(hasattr(model, 'query_gate'))
        self.assertFalse(any('router' in name or 'adapter' in name or 'verifier' in name
                             for name, _ in model.named_parameters()))

    def test_classification_loss_has_no_later_topk_auxiliary(self):
        model, queries, ids, mask = make_fixture()
        model.eval()
        labels = torch.tensor([1, 0])
        expected = model.loss_fn(model(queries, ids, mask)[0], labels)
        batch = dict(ent_rel=queries, source_ids=ids, source_mask=mask,
                     tgt_ent=[1, 0], labels=labels)
        actual = model.training_step(batch, 0)['loss']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_filtered_tail_rank_preserves_target(self):
        harness = SimpleNamespace(configs=SimpleNamespace(is_temporal=False, use_log_ranks=False),
                                  _MASKING_VALUE=-1e9)
        rank = KGCPromptTuner._filtered_ranks(harness, torch.tensor([[0., 3., 5., 2., 1.]]),
                    torch.tensor([0]), torch.tensor([0]), [1], [[0, 1, 0]], {(0, 0): [1, 2]}, 0)
        self.assertEqual(rank, [1])

    def test_filtered_head_rank_preserves_target(self):
        harness = SimpleNamespace(configs=SimpleNamespace(is_temporal=False, use_log_ranks=False),
                                  _MASKING_VALUE=-1e9)
        rank = KGCPromptTuner._filtered_ranks(harness, torch.tensor([[3., 0., 2., 5., 1.]]),
                    torch.tensor([1]), torch.tensor([2]), [0], [[0, 1, 0]], {(1, 2): [0, 3]}, 0)
        self.assertEqual(rank, [1])


if __name__ == '__main__':
    unittest.main()
