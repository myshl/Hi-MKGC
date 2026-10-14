# Hi-MKGC

## Soft Prompt-only release

The restored early multimodal Soft Prompt implementation is in
[`softprompt_only/`](softprompt_only/). Start with its
[installation, training and evaluation instructions](softprompt_only/README.rst).

This version uses image-conditioned prompts in BERT layers **1–6** and
structure-conditioned prompts in layers **7–12**, with text states propagated
through all 12 layers. It does **not** include ReLook experts, routing,
score-level visual refinement, the later query gate, or model ensembling.

The original root-level files and existing data archives are retained unchanged
below for history; run the new release **from `softprompt_only/`**. The new
upload contains code and tests only, not new datasets or model weights.

## Original repository instructions (historical)

- Commands for reproducing:
- Install all the requirements from `./requirements.txt`
- Unzip data.zip and img_data.zip
  ##### Train and Evaluate
  ```shell
  python main.py -dataset DB15K \
                  -batch_size 112 \
                  -pretrained_model pretrained_model_path \
                  -lr 5e-4 \
                  -prompt_length 10 \
                  -label_smoothing 0.1 \
                  -embed_dim 256 \
                  -k_w 16 \
                  -k_h 16 \
                  -epoch 30

  python main.py -dataset DB15K \
                  -batch_size 112 \
                  -pretrained_model pretrained_model_path \
                  -lr 5e-4 \
                  -prompt_length 10 \
                  -label_smoothing 0.1 \
                  -embed_dim 256 \
                  -k_w 16 \
                  -k_h 16 \
                  -epoch 30 \
                  -model_path model.ckpt

  ```                
  
