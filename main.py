import argparse
import datetime
import glob
import os
import time
from pathlib import Path

import torch
from torchinfo import summary
from torchvision import transforms

from dataloaders import ImageDatasetWithMetadata
from engine_test_time import train_on_test
from test_time_training import load_combined_model


def get_args_parser():
    parser = argparse.ArgumentParser("main", add_help=True)
    parser.add_argument("--print_freq", default=50, type=int)
    parser.add_argument(
        "--finetune_mode",
        default="encoder",
        type=str,
        help="all, encoder, encoder_no_cls_no_msk.",
    )
    # Model parameters
    parser.add_argument(
        "--model",
        default="mae_vit_large_patch16",
        type=str,
        metavar="MODEL",
        help="Name of model to train",
    )
    parser.add_argument("--input_size", default=224, type=int, help="images input size")
    parser.add_argument(
        "--classifier_depth",
        type=int,
        metavar="N",
        default=12,
        help="number of blocks in the classifier",
    )
    # Test time training
    parser.add_argument(
        "--mask_ratio",
        default=0.75,
        type=float,
        help="Masking ratio (percentage of removed patches).",
    )
    parser.add_argument(
        "--steps_per_example",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--stored_latents", default="", help="have we generated the latents already?"
    )
    # Optimizer parameters
    parser.add_argument(
        "--weight_decay", type=float, default=0.05, help="weight decay (default: 0.05)"
    )

    parser.add_argument(
        "--blr",
        type=float,
        default=5e-3,
        metavar="LR",
        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256",
    )
    # Dataset parameters
    parser.add_argument(
        "--batch_size",
        default=128,
        type=int,
    )
    parser.add_argument(
        "--data_path", default="./imagenet_c", type=str, help="dataset path"
    )
    parser.add_argument(
        "--dataset_name", default="imagenet_c", type=str, help="dataset name"
    )
    parser.add_argument(
        "--output_dir",
        default="./output_dir",
        help="path where to save, empty for no saving",
    )
    parser.add_argument(
        "--log_dir", default="./output_dir", help="path where to tensorboard log"
    )
    parser.add_argument(
        "--device", default="cpu", help="device to use for training / testing"
    )
    parser.add_argument(
        "--accum_iter",
        default=1,
        type=int,
        help="Accumulate gradient iterations (for increasing the effective batch size under memory constraints)",
    )
    parser.add_argument("--load_loss_scalar", action="store_true")
    parser.set_defaults(load_loss_scalar=False)
    parser.add_argument("--optimizer_type", default="sgd", help="adam, adam_w, sgd.")
    parser.add_argument(
        "--optimizer_momentum", default=0.9, type=float, help="adam, adam_w, sgd."
    )
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--resume_model",
        default="checkpoints/mae_pretrain_vit_large_full.pth",
        help="resume from checkpoint",
    )
    parser.add_argument(
        "--resume_finetune",
        default="checkpoints/prob_lr1e-3_wd.2_blk12_ep20.pth",
        help="resume from checkpoint",
    )
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--pin_mem",
        action="store_true",
        help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.",
    )
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=False)
    parser.add_argument(
        "--norm_pix_loss",
        action="store_true",
        help="Use (per-patch) normalized pixels as targets for computing loss",
    )
    parser.set_defaults(norm_pix_loss=False)
    parser.add_argument("--verbose", action="store_true")
    parser.set_defaults(verbose=False)
    parser.add_argument(
        "--head_type", default="vit_head", help="Head type - linear or vit_head"
    )

    parser.add_argument(
        "--single_crop", action="store_true", help="single_crop training"
    )
    parser.add_argument("--no_single_crop", action="store_false", dest="single_crop")
    parser.set_defaults(single_crop=False)
    # distributed training parameters
    parser.add_argument(
        "--world_size", default=1, type=int, help="number of distributed processes"
    )
    parser.add_argument("--local_rank", default=-1, type=int)
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument(
        "--dist_url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument(
        "--corruption_type",
        default="gaussian_noise",
        type=str,
        help="corruption type to train on",
    )
    parser.add_argument(
        "--corruption_level", default=5, type=int, help="corruption level to train on"
    )
    parser.add_argument(
        "--num_classes", default=1000, type=int, help="number of classes in the dataset"
    )
    parser.add_argument(
        "--no_reset_model",
        action="store_true",
        help="not reset the encoder weights after each iteration",
    )
    parser.set_defaults(no_reset_model=False)

    return parser


def main(args):
    print("Loading model")
    model, optimizer, scalar = load_combined_model(args, args.num_classes)
    torch.manual_seed(args.seed)
    max_known_file = max(
        [
            int(i.split("results_")[-1].split(".npy")[0])
            for i in glob.glob(os.path.join(args.output_dir, "results_*.npy"))
        ]
        + [-1]
    )

    print(summary(model, input_size=(args.batch_size, 3, 224, 224), verbose=1))
    transform_val = transforms.Compose(
        [
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(args.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if not args.single_crop:
        transform_train = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    args.input_size, scale=(0.2, 1.0), interpolation=3
                ),  # 3 is bicubic
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.Resize(256, interpolation=3),
                transforms.CenterCrop(args.input_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    dataset_train = ImageDatasetWithMetadata(
        data_folder=args.data_path,
        transform=transform_train,
        corruption_type=args.corruption_type,
        corruption_level=args.corruption_level,
    )
    dataset_val = ImageDatasetWithMetadata(
        data_folder=args.data_path,
        transform=transform_val,
        corruption_type=args.corruption_type,
        corruption_level=args.corruption_level,
    )
    print("Model and dataloader loaded successfully")
    eff_batch_size = args.batch_size * args.accum_iter
    args.lr = args.blr * eff_batch_size / 256
    base_lr = args.lr * 256 / eff_batch_size
    print("base lr: %.2e" % base_lr)
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)
    device = torch.device(args.device)
    start_time = time.time()
    test_stats = train_on_test(
        model,
        optimizer,
        scalar,
        dataset_train,
        dataset_val,
        device,
        log_writer=None,
        args=args,
        num_classes=args.num_classes,
        iter_start=max_known_file + 1,
    )

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
