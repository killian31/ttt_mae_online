import os
import random
import shutil

from tqdm import tqdm

random.seed(29)

source_folder = "imagenet_c"
destination_folder = "tiny_imagenet_c"

os.makedirs(destination_folder, exist_ok=True)

corruption_types = [
    d
    for d in os.listdir(source_folder)
    if os.path.isdir(os.path.join(source_folder, d))
]

total_classes = 0
for corruption_type in corruption_types:
    level_5_path = os.path.join(source_folder, corruption_type, "5")
    if os.path.exists(level_5_path):
        total_classes += len(
            [
                d
                for d in os.listdir(level_5_path)
                if os.path.isdir(os.path.join(level_5_path, d))
            ]
        )

with tqdm(total=total_classes, desc="Processing classes") as pbar:
    for corruption_type in corruption_types:
        corruption_path = os.path.join(source_folder, corruption_type)
        level_5_path = os.path.join(corruption_path, "5")
        if not os.path.exists(level_5_path):
            continue

        dest_corruption_path = os.path.join(destination_folder, corruption_type, "5")
        os.makedirs(dest_corruption_path, exist_ok=True)

        for class_folder in os.listdir(level_5_path):
            class_path = os.path.join(level_5_path, class_folder)
            if not os.path.isdir(class_path):
                continue

            dest_class_path = os.path.join(dest_corruption_path, class_folder)
            os.makedirs(dest_class_path, exist_ok=True)

            images = [f for f in os.listdir(class_path) if f.endswith(".JPEG")]
            selected_images = random.sample(images, min(10, len(images)))

            for image in selected_images:
                src_image_path = os.path.join(class_path, image)
                dest_image_path = os.path.join(dest_class_path, image)
                shutil.copy(src_image_path, dest_image_path)

            pbar.update(1)
