import glob
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm


def save_failures(val_dataset, corruption_type, output_dir="failures"):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    folder = f"recover_results_{corruption_type}"
    if not os.path.isdir(folder):
        return

    all_loss_files = glob.glob(os.path.join(folder, "losses_*.npy"))
    all_result_files = glob.glob(os.path.join(folder, "results_*.npy"))

    if not all_loss_files or not all_result_files:
        return

    all_loss_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    all_result_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

    last_loss_file = all_loss_files[-1]
    last_result_file = all_result_files[-1]

    losses = np.load(last_loss_file)
    results = np.load(last_result_file)

    subdir = os.path.join(output_dir, corruption_type)
    os.makedirs(subdir, exist_ok=True)

    for i in range(losses.shape[1]):
        acc_first = results[0, i]
        acc_last = results[-1, i]
        if results[0, i] == 0:  # Skip images with initial accuracy of 0
            continue
        if acc_last < acc_first:  # Skip images where accuracy decreased
            loss_first = losses[0, i]
            loss_last = losses[-1, i]
            images, _ = val_dataset[i]  # Retrieve images for the index

            if images.ndim == 4:  # Batch of images (batch_size > 1)
                for batch_idx, img in enumerate(images):
                    save_image(
                        img,
                        subdir,
                        i,
                        batch_idx,
                        loss_first,
                        loss_last,
                        acc_first,
                        acc_last,
                        mean,
                        std,
                    )
            else:  # Single image (batch_size = 1)
                save_image(
                    images,
                    subdir,
                    i,
                    None,
                    loss_first,
                    loss_last,
                    acc_first,
                    acc_last,
                    mean,
                    std,
                )


def save_image(
    img, subdir, idx, batch_idx, loss_first, loss_last, acc_first, acc_last, mean, std
):
    img_unnorm = img.clone()
    for ch in range(3):
        img_unnorm[ch] = img_unnorm[ch] * std[ch] + mean[ch]
    img_unnorm = torch.clamp(img_unnorm, 0, 1)
    arr = (img_unnorm.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    batch_suffix = f"_batch{batch_idx}" if batch_idx is not None else ""
    name = f"loss1={loss_first:.4f}_loss20={loss_last:.4f}_acc1={acc_first:.4f}_acc20={acc_last:.4f}_id={idx}{batch_suffix}.png"
    Image.fromarray(arr).save(os.path.join(subdir, name))


if __name__ == "__main__":
    from data import tt_image_folder

    corruption_types = [
        "gaussian_noise",
        "shot_noise",
        "impulse_noise",
        "defocus_blur",
        "glass_blur",
        "motion_blur",
        "zoom_blur",
        "snow",
        "frost",
        "fog",
        "brightness",
        "contrast",
        "elastic_transform",
        "pixelate",
        "jpeg_compression",
    ]

    data_path = "mini_tiny_imagenet_c"

    for corruption_type in tqdm(corruption_types):
        val_dataset = tt_image_folder.ExtendedImageFolder(
            data_path + "/" + corruption_type + "/" + "5",
            transform=transforms.Compose(
                [
                    transforms.Resize(256, interpolation=3),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
            batch_size=1,
            minimizer=None,
            single_crop=True,
            start_index=0,
        )

        save_failures(val_dataset, corruption_type)
