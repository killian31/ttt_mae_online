import glob
import os
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from classes import IMAGENET2012_CLASSES


class ImageDatasetWithMetadata(Dataset):
    def __init__(self, image_paths, transform, corruption_type=None):
        self.image_paths = image_paths
        self.transform = transform
        self.corruption_type = corruption_type

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        parts = image_path.split(os.sep)
        if self.corruption_type is None:
            corruption_type_name = parts[-4]
        else:
            corruption_type_name = self.corruption_type
        corruption_level_name = parts[-3]
        class_folder = parts[-2]
        class_label = IMAGENET2012_CLASSES.get(class_folder, "Unknown class")
        image = datasets.folder.default_loader(image_path)
        if self.transform:
            image = self.transform(image)
        return image, class_label, corruption_type_name, corruption_level_name


def create_imagenet_dataloader(
    data_folder: str,
    batch_size: int,
    corruption_type: Union[str, None] = None,
    corruption_level: Union[int, str] = "all",
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
):
    """
    Create a PyTorch DataLoader for the given data folder, supporting both single corruption
    type and all corruption types, with optional filtering by corruption level.
    Adds class name from IMAGENET2012_CLASSES.

    Args:
        data_folder (str): Path to the root data folder.
        batch_size (int): Batch size for the DataLoader.
        corruption_type (Union[str, None]): Corruption type (e.g., 'brightness') or None for all types.
        corruption_level (Union[int, str]): Level of corruption (1 to 5) or 'all' for all levels.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        transform: Transformations to apply to the images.

    Returns:
        DataLoader: A PyTorch DataLoader for the specified dataset.
    """
    print(
        f"Creating DataLoader with corruption type {corruption_type} and corruption level {corruption_level}."
    )
    if corruption_level != "all" and corruption_level not in range(1, 6):
        raise ValueError("corruption_level must be an integer from 1 to 5 or 'all'.")

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

    if corruption_type is None:
        if corruption_level == "all":
            pattern = os.path.join(data_folder, "*", "*", "*")
        else:
            pattern = os.path.join(data_folder, "*", str(corruption_level), "*")
    else:
        if corruption_level == "all":
            pattern = os.path.join(data_folder, corruption_type, "*", "*")
        else:
            pattern = os.path.join(
                data_folder, corruption_type, str(corruption_level), "*"
            )

    image_paths = glob.glob(os.path.join(pattern, "*.JPEG"))
    if not image_paths:
        raise FileNotFoundError(f"No images found with the pattern {pattern}")

    dataset = ImageDatasetWithMetadata(image_paths, transform, corruption_type)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":

    def annotate_image(ax, image, class_name, corruption_type, corruption_level):
        """
        Annotates a single image with class name, corruption type, and level.

        Args:
            ax: The matplotlib axis to draw on.
            image: The image tensor.
            class_name: The natural language class name.
            corruption_type: The corruption type.
            corruption_level: The corruption level.
        """
        ax.imshow(image.permute(1, 2, 0).numpy())
        title = f"{class_name}\n{corruption_type} - Level {corruption_level}"
        ax.set_title(title, fontsize=10)
        ax.axis("off")

    def display_batch(
        images, class_names, corruption_types, corruption_levels, title="Batch Display"
    ):
        """
        Displays a single batch of images in a grid.

        Args:
            images: List of images.
            class_names: List of class names.
            corruption_types: List of corruption types.
            corruption_levels: List of corruption levels.
            title: The title of the display.
        """
        num_images = len(images)
        cols = 5
        rows = (num_images + cols - 1) // cols

        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()

        for i, ax in enumerate(axes):
            if i < num_images:
                annotate_image(
                    ax,
                    images[i],
                    class_names[i],
                    corruption_types[i],
                    corruption_levels[i],
                )
            else:
                ax.axis("off")

        fig.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.show()

    def display_multiple_batches(data_loader_dict):
        """
        Displays multiple batches based on different filtering criteria.

        Args:
            data_loader_dict: A dictionary of data loaders with keys indicating the scenario.
        """
        pbar = tqdm(range(len(data_loader_dict)))
        for scenario, loader in data_loader_dict.items():
            pbar.set_description(scenario)
            for images, class_names, corruption_types, corruption_levels in loader:
                display_batch(
                    images,
                    class_names,
                    corruption_types,
                    corruption_levels,
                    title=scenario,
                )
                pbar.update(1)
                break

    def display_corruption_scenarios(data_folder, batch_size):
        """
        Displays batches for specific corruption scenarios.

        Args:
            data_folder: Path to the dataset.
            batch_size: Number of images in each batch.
        """
        scenarios = {
            "Brightness - Level 5": create_imagenet_dataloader(
                data_folder=data_folder,
                batch_size=batch_size,
                corruption_type="brightness",
                corruption_level=5,
            ),
            "Brightness - All Levels": create_imagenet_dataloader(
                data_folder=data_folder,
                batch_size=batch_size,
                corruption_type="brightness",
                corruption_level="all",
            ),
            "All Types - Level 5": create_imagenet_dataloader(
                data_folder=data_folder,
                batch_size=batch_size,
                corruption_type=None,
                corruption_level=5,
            ),
            "All Types - All Levels": create_imagenet_dataloader(
                data_folder=data_folder,
                batch_size=batch_size,
                corruption_type=None,
                corruption_level="all",
            ),
        }

        display_multiple_batches(scenarios)

    import matplotlib.pyplot as plt
    from tqdm import tqdm

    data_folder = "data"
    batch_size = 5

    display_corruption_scenarios(data_folder, batch_size)
