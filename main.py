import os
from typing import Optional, Union

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def create_imagenetc_dataloader(
    data_folder: str,
    batch_size: int,
    corruption_level: Union[int, str] = "all",
    shuffle: bool = True,
    num_workers: int = 4,
    transform: Optional[transforms.Compose] = None,
):
    """
    Create a PyTorch DataLoader for the given data folder and corruption level.

    Args:
        data_folder (str): Path to the root data folder.
        batch_size (int): Batch size for the DataLoader.
        corruption_level (Union[int, str]): Level of corruption (1 to 5) or 'all' for the entire dataset.
        shuffle (bool): Whether to shuffle the dataset.
        num_workers (int): Number of workers for data loading.
        transform: Transformations to apply to the images.

    Returns:
        DataLoader: A PyTorch DataLoader for the specified dataset.
    """
    if corruption_level != "all" and corruption_level not in range(1, 6):
        raise ValueError("corruption_level must be an integer from 1 to 5 or 'all'.")

    if transform is None:
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    # Construct the path based on corruption level
    if corruption_level == "all":
        dataset_path = data_folder
    else:
        dataset_path = os.path.join(data_folder, str(corruption_level))

    # Create the dataset
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)

    # Create the dataloader
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return dataloader


if __name__ == "__main__":
    # Example usage
    data_folder = "./data/"
    batch_size = 64
    corruption_level = 1
    shuffle = True
    num_workers = 4

    dataloader = create_imagenetc_dataloader(
        data_folder, batch_size, corruption_level, shuffle, num_workers
    )

    for images, labels in dataloader:
        # Process the images and labels as needed
        pass
