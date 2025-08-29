---
library_name: transformers
license: apache-2.0
base_model: distilbert/distilbert-base-uncased
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: jutsu_classifier
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# jutsu_classifier

This model is a fine-tuned version of [distilbert/distilbert-base-uncased](https://huggingface.co/distilbert/distilbert-base-uncased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3218
- Accuracy: 0.8675

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.9986        | 1.0   | 276  | 1.0343          | 0.8911   |
| 1.0573        | 2.0   | 552  | 0.9748          | 0.8675   |
| 0.9842        | 3.0   | 828  | 1.2195          | 0.8675   |
| 1.0225        | 4.0   | 1104 | 1.3356          | 0.8113   |
| 1.0494        | 5.0   | 1380 | 1.3218          | 0.8675   |


### Framework versions

- Transformers 4.49.0
- Pytorch 2.5.1+cu121
- Datasets 4.0.0
- Tokenizers 0.21.4
