import os
import torchvision.transforms as transforms
import torchvision.datasets
from torch.utils.data import DataLoader

def get_dataloader(data_dir, image_size=48, batch_size=64):
    """
    Create a DataLoader for the dataset in the specified directory.

    Args:
        data_dir (str): Path to the directory containing the dataset.
        image_size (int): Size to which images will be resized.
        batch_size (int): Number of samples per batch.

    Returns:
        train_loader, val_loader, test_loader: DataLoaders for training, validation, and testing datasets.
    """

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.RandomHorizontalFlip(p=0.4),          
        transforms.RandomRotation(8),                    
        transforms.RandomAffine(degrees=0, translate=(0.08, 0.08)),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    train_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'train'),
        transform=train_transforms
        )
    val_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(data_dir, 'val'),
        transform=val_transforms
        )
    test_path = os.path.join(data_dir, 'test')
    if os.path.exists(test_path):
        test_dataset = torchvision.datasets.ImageFolder(
            root=test_path,
            transform=val_transforms
            )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False
            )
    else:
        test_loader = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
        )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
        )
    
    return train_loader, val_loader, test_loader