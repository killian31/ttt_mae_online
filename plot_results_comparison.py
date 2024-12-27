import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_joint_histograms(output_filename="joint_histograms.png"):
    folder_list = sorted(glob.glob("recover_results_*"))
    corruption_types = [
        "brightness",
        "contrast",
        "defocus_blur",
        "elastic_transform",
        "fog",
        "frost",
        "gaussian_noise",
        "glass_blur",
        "impulse_noise",
        "jpeg_compression",
        "motion_blur",
        "pixelate",
        "shot_noise",
        "snow",
        "zoom_blur",
    ]

    joint_values = {
        "brightness": [68.3, 69.1],
        "contrast": [6.4, 9.8],
        "defocus_blur": [24.2, 34.4],
        "elastic_transform": [31.6, 50.7],
        "fog": [38.6, 44.7],
        "frost": [38.4, 50.7],
        "gaussian_noise": [17.4, 30.5],
        "glass_blur": [18.4, 36.9],
        "impulse_noise": [18.2, 32.4],
        "jpeg_compression": [51.2, 63.0],
        "motion_blur": [32.2, 41.9],
        "pixelate": [49.7, 63.0],
        "shot_noise": [18.2, 33.0],
        "snow": [35.9, 42.8],
        "zoom_blur": [32.2, 45.9],
    }

    data_dict = {}
    for folder in folder_list:
        c_type = folder.replace("recover_results_", "")
        with open(os.path.join(folder, "accuracy.txt"), "r") as f:
            lines = f.read().strip().split("\n")[-20:]
        final_accuracy = float(lines[-1].split("\t")[1])
        data_dict[c_type] = final_accuracy

    sorted_corruptions = sorted(data_dict.keys())
    sorted_corruptions_nl = []
    for corruption in sorted_corruptions:
        parts = corruption.split("_")
        label = parts[0].capitalize()
        if len(parts) > 1:
            label += " " + " ".join(parts[1:])
        sorted_corruptions_nl.append(label)
    bar_width = 0.2
    group_spacing = 0.3
    indices = np.arange(len(sorted_corruptions)) * (3 * bar_width + group_spacing)

    plt.figure(figsize=(12, 6), dpi=300)

    colors = ["mediumseagreen", "indianred", "mediumpurple"]
    labels = [
        "Baseline (ViT Probing)",
        "TTT-MAE (ViT Probing)",
        "Reproduction of TTT-MAE",
    ]

    for i, corruption in enumerate(sorted_corruptions):
        vit_probe, ttt_mae = joint_values[corruption]
        reproduction = data_dict[corruption]
        values = [vit_probe, ttt_mae, reproduction]
        for j, value in enumerate(values):
            plt.bar(
                indices[i] + j * bar_width,
                value,
                width=bar_width,
                color=colors[j],
                label=labels[j] if i == 0 else "",
            )

    plt.xticks(indices + 2 * bar_width, sorted_corruptions_nl, rotation=45, ha="right")
    plt.xlabel("Corruption Type")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_filename, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    plot_joint_histograms()
