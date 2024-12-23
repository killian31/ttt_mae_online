import os
import random

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

source_folder = "imagenet_c"


def select_random_samples(source_folder, num_corruptions=5, num_levels=5):
    corruption_types = [
        d
        for d in os.listdir(source_folder)
        if os.path.isdir(os.path.join(source_folder, d))
    ]
    selected_corruptions = random.sample(corruption_types, num_corruptions)

    grid_images = {}

    for corruption in selected_corruptions:
        corruption_path = os.path.join(source_folder, corruption)
        levels = [str(i) for i in range(1, num_levels + 1)]
        random_class = random.choice(
            [
                d
                for d in os.listdir(os.path.join(corruption_path, "1"))
                if os.path.isdir(os.path.join(corruption_path, "1", d))
            ]
        )

        class_image = None

        for level in levels:
            level_path = os.path.join(corruption_path, level, random_class)
            images = [f for f in os.listdir(level_path) if f.endswith(".JPEG")]
            if class_image is None:
                class_image = random.choice(images)
            grid_images.setdefault(corruption, []).append(
                os.path.join(level_path, class_image)
            )

    return grid_images


def plot_image_rows(grid_images, output_prefix="corruption_row"):
    num_corruptions = len(grid_images)
    num_levels = len(next(iter(grid_images.values())))

    for i, (corruption, images) in enumerate(grid_images.items()):
        fig, axes = plt.subplots(1, num_levels, figsize=(15, 3), dpi=300)

        for j, image_path in enumerate(images):
            img = mpimg.imread(image_path)
            ax = axes[j]
            ax.imshow(img)
            ax.axis("off")
            if i == 0:
                ax.set_title(f"Level {j + 1}")

        plt.tight_layout()
        plt.savefig(f"{output_prefix}_{i + 1}.png", format="png")
        plt.close()


random.seed(29)
grid_images = select_random_samples(source_folder)
plot_image_rows(grid_images)
