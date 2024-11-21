# Test Time Training with Masked Autoencoders: Further Experiments

⚠️Work in development⚠️

This repo aims at reproducing
[ttt-mae](https://yossigandelsman.github.io/ttt_mae/index.html) results and
experiment further.

## Install

1. Create a Python virtual environment

```bash
pyenv virtualenv 3.12.2 ttt-online
pyenv activate ttt-online
```

2. Install the dependencies

```bash
pip install poetry
poetry install
```

## Run

```bash
poetry run python main.py --help
```

```
options:
  -h, --help            show this help message and exit
  --print_freq PRINT_FREQ
  --finetune_mode FINETUNE_MODE
                        all, encoder, encoder_no_cls_no_msk.
  --model MODEL         Name of model to train
  --input_size INPUT_SIZE
                        images input size
  --classifier_depth N  number of blocks in the classifier
  --mask_ratio MASK_RATIO
                        Masking ratio (percentage of removed patches).
  --steps_per_example STEPS_PER_EXAMPLE
  --stored_latents STORED_LATENTS
                        have we generated the latents already?
  --weight_decay WEIGHT_DECAY
                        weight decay (default: 0.05)
  --blr LR              base learning rate: absolute_lr = base_lr * total_batch_size / 256
  --batch_size BATCH_SIZE
  --data_path DATA_PATH
                        dataset path
  --dataset_name DATASET_NAME
                        dataset name
  --output_dir OUTPUT_DIR
                        path where to save, empty for no saving
  --log_dir LOG_DIR     path where to tensorboard log
  --device DEVICE       device to use for training / testing
  --accum_iter ACCUM_ITER
                        Accumulate gradient iterations (for increasing the effective batch
                        size under memory constraints)
  --load_loss_scalar
  --optimizer_type OPTIMIZER_TYPE
                        adam, adam_w, sgd.
  --optimizer_momentum OPTIMIZER_MOMENTUM
                        adam, adam_w, sgd.
  --seed SEED
  --resume_model RESUME_MODEL
                        resume from checkpoint
  --resume_finetune RESUME_FINETUNE
                        resume from checkpoint
  --num_workers NUM_WORKERS
  --pin_mem             Pin CPU memory in DataLoader for more efficient (sometimes) transfer
                        to GPU.
  --no_pin_mem
  --norm_pix_loss       Use (per-patch) normalized pixels as targets for computing loss
  --verbose
  --head_type HEAD_TYPE
                        Head type - linear or vit_head
  --single_crop         single_crop training
  --no_single_crop
  --world_size WORLD_SIZE
                        number of distributed processes
  --local_rank LOCAL_RANK
  --dist_on_itp
  --dist_url DIST_URL   url used to set up distributed training
  --corruption_type CORRUPTION_TYPE
                        corruption type to train on
  --corruption_level CORRUPTION_LEVEL
                        corruption level to train on
  --num_classes NUM_CLASSES
                        number of classes in the dataset
```

## State of Development

- [x] Run test time training
- [ ] Run online test time training
- [ ] Characterize failure cases
- [ ] Impact of the number of steps
