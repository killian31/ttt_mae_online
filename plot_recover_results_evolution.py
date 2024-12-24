import glob
import os

import matplotlib.pyplot as plt
import numpy as np


def gather_and_plot_accuracies(output_filename="all_corruptions_accuracy_vs_steps.png"):
    folder_list = glob.glob("recover_results_*")
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
    for i, (k, (s, a)) in enumerate(data_dict.items()):
        parts = k.split("_")
        label = parts[0].capitalize()
        if len(parts) > 1:
            label += " " + " ".join(parts[1:])
        plt.plot(s, a, marker="o", color=colors[i], label=label)
    plt.xticks(range(1, 21))
    plt.xlabel("Step")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize="small", frameon=False)
    plt.tight_layout(rect=[0, 0, 0.95, 1])
    plt.savefig(output_filename)
    plt.close()


if __name__ == "__main__":
    gather_and_plot_accuracies()
