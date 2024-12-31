# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import copy
import glob

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------
import math
import os.path
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from scipy import stats
from tqdm import tqdm

import models_mae_shared
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


@torch.no_grad()
def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    # output is (B, classes)
    # target is (B)
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:k].reshape(-1).float().sum(0) * 100.0 / batch_size for k in topk]


def get_prameters_from_args(model, args):
    if args.finetune_mode == "encoder":
        for name, p in model.named_parameters():
            if name.startswith("decoder"):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    elif args.finetune_mode == "all":
        parameters = model.parameters()
    elif args.finetune_mode == "encoder_no_cls_no_msk":
        for name, p in model.named_parameters():
            if (
                name.startswith("decoder")
                or name == "cls_token"
                or name == "mask_token"
            ):
                p.requires_grad = False
        parameters = [p for p in model.parameters() if p.requires_grad]
    else:
        raise ValueError(f"Unknown finetune_mode: {args.finetune_mode}")
    return parameters


def _reinitialize_model(
    base_model, base_optimizer, base_scalar, clone_model, args, device
):
    if args.stored_latents:
        # We don't need to change the model, as it is never changed
        base_model.train(True)
        base_model.to(device)
        return base_model, base_optimizer, base_scalar
    if not args.no_reset_encoder:
        clone_model.load_state_dict(copy.deepcopy(base_model.state_dict()))
    clone_model.train(True)
    clone_model.to(device)
    if args.optimizer_type == "sgd":
        optimizer = torch.optim.SGD(
            get_prameters_from_args(clone_model, args),
            lr=args.lr,
            momentum=args.optimizer_momentum,
        )
    elif args.optimizer_type == "adam":
        optimizer = torch.optim.Adam(
            get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95)
        )
    else:
        assert args.optimizer_type == "adam_w"
        optimizer = torch.optim.AdamW(
            get_prameters_from_args(clone_model, args), lr=args.lr, betas=(0.9, 0.95)
        )
    optimizer.zero_grad()
    loss_scaler = NativeScaler()
    if args.load_loss_scalar:
        loss_scaler.load_state_dict(base_scalar.state_dict())
    return clone_model, optimizer, loss_scaler


def save_failure_case(
    image,
    output_dir,
    corruption_type,
    initial_rec_loss,
    final_rec_loss,
    initial_cls_loss,
    final_cls_loss,
    acc_before,
    acc_after,
):
    failure_dir = os.path.join(output_dir, corruption_type)
    os.makedirs(failure_dir, exist_ok=True)
    file_name = f"i_rec_{initial_rec_loss:.4f}_f_rec_{final_rec_loss:.4f}_i_cls_{initial_cls_loss:.4f}_f_cls_{final_cls_loss:.4f}_i_acc_{acc_before}_f_acc_{acc_after}.JPEG"
    file_path = os.path.join(failure_dir, file_name)
    if len(glob.glob(os.path.join(failure_dir, "*.JPEG"))) > 5:
        return
    else:
        image = image.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
        image = image * np.array([0.229, 0.224, 0.225])
        image = image + np.array([0.485, 0.456, 0.406])
        image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
        pil_image.save(file_path)


def save_mosaic_of_reconstructions(
    reconstructed_images, rec_losses, cls_losses, preds, data_iter_step, args
):
    """
    Creates a 5x8 mosaic of reconstructed images from 40 steps.
    Each subplot has two lines of text:
      * Red:  rec loss
      * Blue: cls loss + prediction (0 or 1)

    :param reconstructed_images: list of np arrays, each shape (H, W, 3)
    :param rec_losses: list of float
    :param cls_losses: list of float
    :param preds: list of int, either 0 or 1
    :param data_iter_step: current data iteration step (for naming/labeling)
    :param args: your argparse or config object containing output_dir, corruption_type, etc.
    """
    cols = 4
    original_image = reconstructed_images[0]
    reconstructed_images = reconstructed_images[1:]
    n_steps = len(reconstructed_images)

    index_to_keep = [0, n_steps // 2, -1]
    reconstructed_images = [reconstructed_images[i] for i in index_to_keep]
    rec_losses = [rec_losses[i] for i in index_to_keep]
    cls_losses = [cls_losses[i] for i in index_to_keep]
    preds = [preds[i] for i in index_to_keep]
    steps = [0, n_steps // 2, n_steps - 1]

    figsize = (20, 10)
    fig, axs = plt.subplots(1, cols, figsize=figsize)

    axs = axs.flat
    axs[0].imshow(original_image)
    axs[0].axis("off")
    axs[0].text(
        0.5,
        1.02,
        "Original Image",
        color="black",
        fontsize=8,
        ha="center",
        va="bottom",
        transform=axs[0].transAxes,
    )

    for i in range(0, len(reconstructed_images)):
        ax = axs[i + 1]
        ax.imshow(reconstructed_images[i])
        ax.axis("off")

        rec_loss_text = f"Rec: {rec_losses[i]:.4f}, TTT step {steps[i] + 1}"
        ax.text(
            0.5,
            1.08,
            rec_loss_text,
            color="red",
            fontsize=8,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

        cls_loss_text = f"Cls: {cls_losses[i]:.4f}, Pred: {preds[i]}"
        ax.text(
            0.5,
            1.02,
            cls_loss_text,
            color="blue",
            fontsize=8,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )

    title = args.corruption_type.split("_")
    title[0] = title[0].capitalize()

    fig.suptitle(
        f"Corruption: {' '.join(title)}",
        fontsize=16,
    )

    plt.tight_layout()

    mosaic_path = os.path.join(
        args.output_dir, f"mosaic_{data_iter_step}_{args.corruption_type}.png"
    )
    plt.savefig(mosaic_path, dpi=150)
    plt.close(fig)


def train_on_test(
    base_model: torch.nn.Module,
    base_optimizer,
    base_scalar,
    dataset_train,
    dataset_val,
    device: torch.device,
    log_writer=None,
    args=None,
    num_classes: int = 1000,
    iter_start: int = 0,
):
    if args.model == "mae_vit_small_patch16":
        classifier_depth = 8
        classifier_embed_dim = 512
        classifier_num_heads = 16
    else:
        assert (
            "mae_vit_huge_patch14" in args.model
            or args.model == "mae_vit_large_patch16"
        )
        classifier_embed_dim = 768
        classifier_depth = 12
        classifier_num_heads = 12
    clone_model = models_mae_shared.__dict__[args.model](
        num_classes=num_classes,
        head_type=args.head_type,
        norm_pix_loss=args.norm_pix_loss,
        classifier_depth=classifier_depth,
        classifier_embed_dim=classifier_embed_dim,
        classifier_num_heads=classifier_num_heads,
        rotation_prediction=False,
    )
    # Intialize the model for the current run
    all_results = [list() for i in range(args.steps_per_example)]
    all_losses = [list() for i in range(args.steps_per_example)]
    metric_logger = misc.MetricLogger(delimiter="  ")
    train_loader = iter(
        torch.utils.data.DataLoader(
            dataset_train, batch_size=1, shuffle=False, num_workers=args.num_workers
        )
    )
    val_loader = iter(
        torch.utils.data.DataLoader(
            dataset_val, batch_size=1, shuffle=False, num_workers=args.num_workers
        )
    )
    accum_iter = args.accum_iter
    metric_logger.add_meter("lr", misc.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    init_no_reset_encoder = args.no_reset_encoder
    args.no_reset_encoder = False
    model, optimizer, loss_scaler = _reinitialize_model(
        base_model, base_optimizer, base_scalar, clone_model, args, device
    )
    args.no_reset_encoder = init_no_reset_encoder
    pbar = tqdm(total=len(dataset_val))
    if log_writer is not None:
        print("log_dir: {}".format(log_writer.log_dir))
    dataset_len = len(dataset_val)
    for data_iter_step in range(iter_start, dataset_len):
        val_data = next(val_loader)
        test_samples, test_label = val_data
        if args.save_failures:
            cls_losses, rec_losses, preds, reconstructed_images = [], [], [], []
            test_image_path = os.path.join(
                args.output_dir,
                f"test_image_{data_iter_step}_{args.corruption_type}.png",
            )
            to_save = test_samples.squeeze().detach().cpu().numpy().transpose(1, 2, 0)
            to_save = to_save * np.array([0.229, 0.224, 0.225])
            to_save = to_save + np.array([0.485, 0.456, 0.406])
            to_save = (to_save * 255).astype(np.uint8)
            reconstructed_images.append(to_save)
            Image.fromarray(to_save).save(test_image_path)
        test_samples = test_samples.to(device, non_blocking=True)[0]
        test_label = test_label.to(device, non_blocking=True)

        pseudo_labels = None

        rec_loss_before = None
        cls_loss_before = None

        # Test time training:
        for step_per_example in range(args.steps_per_example * accum_iter):
            train_data = next(train_loader)
            mask_ratio = args.mask_ratio
            samples, _ = train_data
            # samples = samples
            targets_rot, samples_rot = None, None
            samples = samples.to(device, non_blocking=True)[
                0
            ]  # index [0] becuase the data is batched to have size 1.
            loss_dict, reconstruction, _, _ = model(
                samples, None, mask_ratio=mask_ratio
            )

            loss = torch.stack([loss_dict[l] for l in loss_dict]).sum()
            loss_value = loss.item()
            loss /= accum_iter
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            loss_scaler(
                loss,
                optimizer,
                parameters=model.parameters(),
                update_grad=(step_per_example + 1) % accum_iter == 0,
            )

            if step_per_example == 0:
                rec_loss_before = loss_value
            if step_per_example == args.steps_per_example * accum_iter - 1:
                rec_loss_after = loss_value
            if (step_per_example + 1) % accum_iter == 0:
                if args.verbose:
                    print(
                        f"datapoint {data_iter_step} iter {step_per_example}: rec_loss {loss_value}"
                    )

                all_losses[step_per_example // accum_iter].append(
                    loss_value / accum_iter
                )
                optimizer.zero_grad()

            metric_logger.update(**{k: v.item() for k, v in loss_dict.items()})
            lr = optimizer.param_groups[0]["lr"]
            metric_logger.update(lr=lr)
            # Test:
            if (step_per_example + 1) % accum_iter == 0:
                with torch.no_grad():
                    model.eval()
                    all_pred = []
                    for _ in range(accum_iter):
                        loss_d, _, _, pred = model(
                            test_samples,  # .unsqueeze(0),
                            test_label,
                            mask_ratio=0,
                            reconstruct=False,
                        )
                        if step_per_example == 0:
                            cls_loss_before = loss_d["classification"].item()
                        if step_per_example == args.steps_per_example * accum_iter - 1:
                            cls_loss_after = loss_d["classification"].item()
                        if args.verbose:
                            cls_loss = loss_d["classification"].item()
                            print(
                                f"datapoint {data_iter_step} iter {step_per_example}: class_loss {cls_loss}"
                            )
                        all_pred.extend(
                            list(pred.argmax(axis=1).detach().cpu().numpy())
                        )
                    acc1 = (
                        stats.mode(all_pred).mode
                        == test_label[0].cpu().detach().numpy()
                    ) * 100.0
                    if args.save_failures:
                        cls_loss = loss_d["classification"].item()
                        cls_losses.append(cls_loss)
                        rec_losses.append(loss_value)
                        preds.append(
                            int(
                                (
                                    pred.argmax(axis=1).detach().cpu().numpy()
                                    == test_label[0].detach().cpu().numpy()
                                )[0]
                            )
                        )
                        pred_image = model.unpatchify(reconstruction)
                        reconstruct_to_save = (
                            pred_image.squeeze()
                            .detach()
                            .cpu()
                            .numpy()
                            .transpose(1, 2, 0)
                        )
                        reconstruct_to_save = reconstruct_to_save * np.array(
                            [0.229, 0.224, 0.225]
                        )
                        reconstruct_to_save = reconstruct_to_save + np.array(
                            [0.485, 0.456, 0.406]
                        )
                        reconstruct_to_save = (reconstruct_to_save * 255).astype(
                            np.uint8
                        )
                        reconstructed_images.append(reconstruct_to_save)
                    if (step_per_example + 1) // accum_iter == args.steps_per_example:
                        metric_logger.update(top1_acc=acc1)
                        metric_logger.update(loss=loss_value)
                    all_results[step_per_example // accum_iter].append(acc1)
                    if step_per_example == 0:
                        acc_before = acc1
                    if step_per_example == args.steps_per_example * accum_iter - 1:
                        acc_after = acc1
                    model.train()
        if args.save_failures:
            if args.verbose:
                print("cls_losses:", cls_losses)
                print("rec_losses:", rec_losses)
                print("preds:", preds)
            reconstruction_dir = os.path.join(
                args.output_dir, f"reconstruction_{data_iter_step}"
            )
            os.makedirs(reconstruction_dir, exist_ok=True)

            if os.path.exists(test_image_path):
                test_image_pil = Image.open(test_image_path)
            else:
                test_image_pil = None

            # Setup figure with two columns using GridSpec
            fig = plt.figure(figsize=(12, 5))
            gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])

            # Left subplot: display the saved test image
            ax_left = fig.add_subplot(gs[0])
            if test_image_pil:
                ax_left.imshow(test_image_pil)
                ax_left.axis("off")
            else:
                ax_left.text(
                    0.5,
                    0.5,
                    "No test image found",
                    ha="center",
                    va="center",
                    fontsize=12,
                )
                ax_left.axis("off")

            # Right subplot: plot the rec_losses and cls_losses
            ax_right = fig.add_subplot(gs[1])
            steps_array = np.arange(len(rec_losses))
            ax_right.plot(
                steps_array, rec_losses, marker="o", label="Reconstruction Loss"
            )
            ax_right.plot(
                steps_array, cls_losses, marker="s", label="Classification Loss"
            )

            # Label each point with the pred value
            for i, (r_loss, c_loss, p) in enumerate(zip(rec_losses, cls_losses, preds)):
                ax_right.text(
                    i, r_loss, str(p), color="red", fontsize=8, ha="center", va="bottom"
                )
                ax_right.text(
                    i,
                    c_loss,
                    str(p),
                    color="blue",
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )

            ax_right.set_xlabel("Step")
            ax_right.set_ylabel("Loss Value")
            ax_right.set_title("rec vs cls losses")
            ax_right.legend()
            fig.suptitle("0: Misclassified, 1: Correctly classified", fontsize=12)

            # Save the plot in args.output_dir
            plot_path = os.path.join(
                args.output_dir,
                f"loss_plot_{data_iter_step}_{args.corruption_type}.png",
            )
            fig.savefig(plot_path, dpi=150)
            plt.close(fig)

            save_mosaic_of_reconstructions(
                reconstructed_images,
                rec_losses,
                cls_losses,
                preds,
                data_iter_step,
                args,
            )
        pbar.update(1)

        """
        if args.save_failures:
            with torch.no_grad():
                model.eval()
                if args.verbose:
                    print(
                        f"datapoint {data_iter_step} done: rec_loss_before {rec_loss_before} rec_loss_after {rec_loss_after} cls_loss_before {cls_loss_before} cls_loss_after {cls_loss_after}"
                    )
                if acc_before - acc_after > 0:  # Failure case
                    save_failure_case(
                        test_samples,
                        args.output_dir,
                        args.corruption_type,
                        rec_loss_before,
                        rec_loss_after,
                        cls_loss_before,
                        cls_loss_after,
                        acc_before,
                        acc_after,
                    )
        """
        if data_iter_step % 50 == 1:
            print(
                "step: {}, acc {} rec-loss {}".format(
                    data_iter_step, np.mean(all_results[-1]), loss_value
                )
            )
        if data_iter_step % 500 == 499 or (data_iter_step == dataset_len - 1):
            with open(
                os.path.join(args.output_dir, f"results_{data_iter_step}.npy"), "wb"
            ) as f:
                np.save(f, np.array(all_results))
            with open(
                os.path.join(args.output_dir, f"losses_{data_iter_step}.npy"), "wb"
            ) as f:
                np.save(f, np.array(all_losses))
            all_results = [list() for i in range(args.steps_per_example)]
            all_losses = [list() for i in range(args.steps_per_example)]
        model, optimizer, loss_scaler = _reinitialize_model(
            base_model, base_optimizer, base_scalar, clone_model, args, device
        )
    pbar.close()
    save_accuracy_results(args)
    # gather the stats from all processes
    try:
        print("Averaged stats:", metric_logger)
        return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    except:
        pass
    return


def save_accuracy_results(args):
    all_all_results = [list() for i in range(args.steps_per_example)]
    for file_number, f_name in enumerate(
        glob.glob(os.path.join(args.output_dir, "results_*.npy"))
    ):
        all_data = np.load(f_name)
        for step in range(args.steps_per_example):
            all_all_results[step] += all_data[step].tolist()
    with open(os.path.join(args.output_dir, "model-final.pth"), "w") as f:
        f.write(f"Done!\n")
    with open(os.path.join(args.output_dir, "accuracy.txt"), "a") as f:
        f.write(f"{str(args)}\n")
        for i in range(args.steps_per_example):
            # assert len(all_all_results[i]) == 50000, len(all_all_results[i])
            f.write(f"{i}\t{np.mean(all_all_results[i])}\n")


if __name__ == "__main__":
    import argparse

    args = argparse.Namespace()
    args.steps_per_example = 20
    args.output_dir = "recover_results_brightness"
    save_accuracy_results(args)
