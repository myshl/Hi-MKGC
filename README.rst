Hi-MKGC: multilayer multimodal Soft Prompts
=========================================

This is the early Soft Prompt-only implementation restored from
``CSProm-KG-main-824``. It is separate from ReLook-KGC. The original local
directory is preserved; this release removes the subsequently added optional
query gate and hard-Top-k auxiliary objective from the publication copy.

Model design
------------

Image and structural entity/relation embeddings feed separate prompt MLPs.
Each produces six depth slices. The code concatenates the image slices before
the structural slices along the Transformer-depth axis:

.. list-table:: BERT-base layer schedule, using one-based layer numbers
   :header-rows: 1

   * - Layers
     - Newly inserted Soft Prompt
   * - 1–6
     - Image-conditioned prompt
   * - 7–12
     - Structure-conditioned prompt

At every layer, the old prompt prefix is replaced by the next depth-specific
prompt. The text hidden states continue across layers; the total sequence
length does not grow with depth. The final prompt states are projected to
query embeddings, then a ConvE decoder scores all candidate entities using
the structural embedding table. BERT parameters are frozen when prompting is
enabled; gradients still pass through BERT to the prompt generators.

The released schedule, prompt tensor reshape, feature pooling, decoder,
dropout behavior and filtered-ranking convention preserve the early
implementation. This release does not reverse the modality order, repair the
historical LAR objective, or import later ReLook model changes.

There are no shared/role experts, adapters, query gates, evidence routers,
score-level visual corrections, candidate rerankers or ensembles. The later
``.361`` ensemble is not a result of this single-model release, and no such
metric is claimed here.

Files
-----

* ``models/P_model.py``: feature loading, multimodal prompt composition,
  objectives and filtered head/tail evaluation.
* ``models/prompter.py``: six-slice modality-specific prompt generator.
* ``models/bert_for_layerwise.py``: per-layer prompt replacement.
* ``models/graph_models.py``: ConvE and the inherited scoring alternatives.
* ``kgc_data.py`` and ``helper.py``: text construction, data loading and metrics.
* ``main.py``: training, validation checkpoint selection and evaluation.
* ``tests/test_softprompt.py``: CPU-only layer-order, gradient, loss and ranking
  checks using synthetic inputs, without external model downloads.
* ``release_manifest.json``: source provenance and release verification.

Install without Conda
---------------------

From the repository root:

.. code-block:: bash

    python3.10 -m venv .venv
    .venv/bin/python -m pip install -r requirements.txt
    .venv/bin/python -m nltk.downloader stopwords
    .venv/bin/python -m unittest discover -s tests -v

Use Python 3.10, PyTorch 1.13.1, Lightning 1.9.5 and Transformers 4.26.1.
The historical runtime used the CUDA 11.7 PyTorch build. A compatible driver
is required for GPU execution. No Conda commands or environments are used.

External inputs
---------------

Place the processed triples/text, precomputed feature files and a trusted
BERT-base-uncased model/tokenizer separately:

::

    Hi-MKGC/
      data/processed/{DB15K,YAGO15K,FB15k-237}/
        entity2id.txt
        relation2id.txt
        entityid2name.txt
        relationid2name.txt
        entityid2description.txt
        train2id.txt
        valid2id.txt
        test2id.txt
      img_data/
        <dataset>_img_features.pkl
        <dataset>_entity_embed.pkl
        <dataset>_relation_embed.pkl
      model_path/bert-base-uncased/
        ... model configuration, tokenizer and pretrained weights ...

The image feature matrix is ``number_of_entities x 4096``; the historical
normalization and 4-by-4 average pooling reduce it to 256 dimensions.
Structural entity embeddings are ``number_of_entities x 256``; ConvE uses
``2 * number_of_relations x 256`` structural relation embeddings. Entity and
relation ID order must agree across every file. For FB15k-237, both feature
prefixes ``FB15k-237`` and ``FB15K-237`` are supported.

Triple files begin with a triple count, followed by space-separated
``head tail relation`` IDs. Text maps have a header followed by tab-separated
ID/text rows. Supply alternate locations with ``-dataset_path``,
``-feature_dir`` and ``-pretrained_model``. Feature paths are resolved relative
to this code directory unless absolute.

Data and pretrained weights are not included in this release. The old data
archives have been removed from the current tree and remain recoverable from
Git history; their compatibility is not certified. Local historical DB15K splits were found to overlap
between training and test. Filtered ranking alone does not remove that split
overlap, so those scores must not be presented as clean held-out evidence.
Only load trusted pickle/checkpoint files.

Train and evaluate
------------------

The shell scripts below are example recipes, not a claim of new complete
benchmark results. They default to 256-dimensional embeddings and 10 prompt
tokens per group. Extra arguments override the example's defaults.

.. code-block:: bash

    bash train.sh DB15K -gpu 0
    bash train.sh YAGO15K -gpu 0
    bash train.sh FB15k-237 -gpu 0

    bash evaluation.sh FB15k-237 /path/to/selected.ckpt -gpu 0

``train.sh`` retains the early CE+LAR example recipe; use ``-n_lar 0 -alpha 0``
for classification alone. The optional inherited LAR machinery is unchanged,
including its historical distance/sampling behavior; it is not the corrected
LAR recipe of the later ReLook release. Label smoothing is configurable.
There is no later Top-k auxiliary objective or multi-answer set-loss module.

The training script validates every epoch and returns validation metrics for
the selected checkpoint. It does not automatically inspect test performance
during tuning. ``evaluation.sh`` performs the full test evaluation separately;
use ``-eval_split valid`` to evaluate a supplied checkpoint on validation.
Checkpoint loading is strict, so a gated or structurally different checkpoint
cannot be silently treated as this version. Do not pass ReLook checkpoints.

Evaluation scores every entity for both head and tail prediction, filters
other known true answers from train+validation+test, and restores the target
before ranking. It retains the historical ``torch.sort`` tie ordering rather
than introducing a new average-tie convention. Evaluation filtering does not
mean that the training classification loss masks all alternative positives.

For a small infrastructure check after providing the real external assets:

.. code-block:: bash

    bash train.sh DB15K -gpu 0 -batch_size 4 -val_batch_size 4 -num_workers 0 \
      -epoch 1 -n_lar 0 -alpha 0 -limit_train_batches 1 \
      -limit_val_batches 1 -save_dir checkpoint/smoke

Such a limited run is a smoke test, not a full reported result. A full training
run needs appropriate GPU memory; the scripts do not silently alter batches
or the model after an out-of-memory error. Set ``SOFTPROMPT_PYTHON`` to an
existing compatible Python interpreter if needed.

Release changes and attribution
-------------------------------

The active baseline path is preserved. Packaging adds configurable feature
locations, the Freebase filename fallback, strict checkpoint loading, an
explicit validation/test switch, a clear missing-checkpoint error, tests and
documentation. It removes the later optional query gate and Top-k objective.
The original Prompter and layer-wise encoder files are preserved byte-for-byte.

This code derives from the user's early multimodal extension of CSProm-KG.
The inherited helper retains its fastai attribution; the Transformer code
uses Hugging Face Transformers and PyTorch. No new blanket license is assigned
to upstream code, pretrained models or data. This implementation replaces the
old root-level code and the temporary ``softprompt_only/`` layout. Old code,
data archives, bytecode caches and the placeholder checkpoint have been removed
from the current tree; previous commits remain available in Git history.
