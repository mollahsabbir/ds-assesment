import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms

# Constants
DATA_DIR = "data/dataset_256X256/dataset_256X256/"
EPOCHS = 3

def calculate_mean_std(data_dir=DATA_DIR):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor()
    ])

    dataset = ImageFolder(root=data_dir, transform=transform)

    mean = torch.zeros(3)
    std = torch.zeros(3)

    for data, _ in dataset:
        mean += torch.mean(data, dim=(1,2))
        std += torch.std(data, dim=(1,2))

    mean /= len(dataset)
    std /= len(dataset)

    return (mean, std)