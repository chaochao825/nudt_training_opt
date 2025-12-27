import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, dataset_name='CIFAR-10', batch_size=32, num_workers=0, image_size=224):
    """
    Creates data loaders for training and testing.
    Supports CIFAR-10, MNIST, and ImageNet.
    """
    
    # Define normalization based on dataset
    if dataset_name.upper() == 'MNIST':
        normalize = transforms.Normalize((0.1307,), (0.3081,))
        input_channels = 1
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        input_channels = 3

    train_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip() if dataset_name.upper() != 'MNIST' else transforms.RandomRotation(10),
        transforms.Grayscale(num_output_channels=3) if dataset_name.upper() == 'MNIST' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        normalize,
    ])

    test_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=3) if dataset_name.upper() == 'MNIST' else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        normalize,
    ])

    # Construct path for specific dataset
    dataset_path = os.path.join(data_dir, dataset_name)
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    # Default to ImageFolder if directory exists, otherwise could add more logic
    if os.path.exists(train_path):
        train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
        test_dataset = datasets.ImageFolder(test_path, transform=test_transform)
    else:
        # Fallback for testing or when data is not in ImageFolder format
        # In a real scenario, we might want to download or handle differently
        raise FileNotFoundError(f"Dataset path not found: {train_path}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, train_dataset.classes
