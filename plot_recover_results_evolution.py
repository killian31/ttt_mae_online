import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def gather_and_plot_accuracies(output_filename="all_corruptions_accuracy_vs_steps.png"):
    folder_list = sorted(glob.glob("recover_results_*"))
    data_dict = {}
    for folder in folder_list:
        c_type = folder.replace("recover_results_", "")
        with open(os.path.join(folder, "accuracy.txt"), "r") as f:
            lines = f.read().strip().split("\n")[-20:]
        steps, accuracies = [], []
        for line in lines:
            step, acc = line.split("\t")
            steps.append(int(step) + 1)
            accuracies.append(float(acc))
        data_dict[c_type] = (steps, accuracies)
    plt.figure(figsize=(8, 5), dpi=300)
    color_vals = np.linspace(0, 1, len(data_dict))
    colors = plt.cm.rainbow(color_vals)
    sorted_data = sorted(data_dict.items())
    for i, (k, (s, a)) in enumerate(sorted_data):
        parts = k.split("_")
        label = parts[0].capitalize()
        if len(parts) > 1:
            label += " " + " ".join(parts[1:])
        plt.plot(s, a, marker="o", color=colors[i], label=label)
    plt.xticks(range(1, 21))
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", frameon=False)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_filename)
    plt.close()


def plot_accuracy_vs_steps_online(output_filename="on_line_accuracy_vs_steps.png"):
    folder_list = sorted(glob.glob("online_*"))
    data_dict = {"baseline": {}, "online": {}}

    for folder in folder_list:
        if "baseline" in folder:
            key = "baseline"
        else:
            key = "online"

        c_type = folder.replace("online_baseline_sgd_", "").replace("online_sgd_", "")
        with open(os.path.join(folder, "accuracy.txt"), "r") as f:
            lines = f.read().strip().split("\n")[-40:]
        steps, accuracies = [], []
        for line in lines:
            step, acc = line.split("\t")
            steps.append(int(step) + 1)
            accuracies.append(float(acc))
        data_dict[key][c_type] = (steps, accuracies)

    plt.figure(figsize=(10, 6), dpi=150)
    color_vals = np.linspace(0, 1, 15)
    colors = plt.cm.rainbow(color_vals)
    colors = [colors[0], colors[2], colors[5]]

    for i, (c_type, (steps, accuracies)) in enumerate(
        sorted(data_dict["online"].items())
    ):
        label = c_type.replace("_", " ").capitalize()
        plt.plot(
            steps,
            accuracies,
            linestyle="-",
            color=colors[i],
            label=f"Online - {label}",
        )

    for i, (c_type, (steps, accuracies)) in enumerate(
        sorted(data_dict["baseline"].items())
    ):
        label = c_type.replace("_", " ").capitalize()
        plt.plot(
            steps,
            accuracies,
            linestyle="--",
            color=colors[i],
            label=f"Baseline - {label}",
        )

    plt.xticks(range(1, 41, 2))
    plt.xlabel("Step")
    plt.ylabel("Accuracy (%)")
    plt.legend(loc="best")  # , bbox_to_anchor=(1, 1), fontsize="small", frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_filename)
    plt.close()


if __name__ == "__main__":
    # gather_and_plot_accuracies()
    plot_accuracy_vs_steps_online()
