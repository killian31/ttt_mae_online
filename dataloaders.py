import glob
import os
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from classes import IMAGENET2012_CLASSES


def create_class_mapping():
    """
    Create a mapping from class folder names to integer labels.

    Returns:
        dict: A mapping from class folder names to integer labels.
    """
    class_folders = IMAGENET2012_CLASSES.keys()
    class_mapping = {class_folder: i for i, class_folder in enumerate(class_folders)}
    return class_mapping


class ImageDatasetWithMetadata(Dataset):
    def __init__(
        self,
        data_folder: str,
        transform: Optional[transforms.Compose] = None,
        corruption_type: Union[str, None] = None,
        corruption_level: Union[int, str] = "all",
        class_mapping: Optional[dict] = None,
    ):
        """
        Dataset class for loading images with corruption metadata.

        Args:
            data_folder (str): Path to the root data folder.
            transform (Optional[transforms.Compose]): Transformations to apply to the images.
            corruption_type (Union[str, None]): Corruption type (e.g., 'brightness') or None for all types.
            corruption_level (Union[int, str]): Corruption level (1 to 5) or 'all' for all levels.
            class_mapping (Optional[dict]): Optional precomputed mapping from class folder names to integers.
        """
        if corruption_level != "all" and corruption_level not in range(1, 6):
            raise ValueError(
                "corruption_level must be an integer from 1 to 5 or 'all'."
            )

        self.data_folder = data_folder
        self.transform = transform or transforms.Compose([transforms.ToTensor()])
        self.corruption_type = corruption_type
        self.corruption_level = corruption_level

        self.class_mapping = class_mapping or create_class_mapping()

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

        self.image_paths = glob.glob(os.path.join(pattern, "*.JPEG"))
        if not self.image_paths:
            raise FileNotFoundError(f"No images found with the pattern {pattern}")

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
        class_label = self.class_mapping[class_folder]

        image = datasets.folder.default_loader(image_path)
        if self.transform:
            image = self.transform(image)

        nl_class_name = IMAGENET2012_CLASSES.get(class_folder, "Unknown class")

        return (
            image,
            class_label,
            nl_class_name,
            corruption_type_name,
            corruption_level_name,
        )
