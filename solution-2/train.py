import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import importlib
import matplotlib.pyplot as plt

from utils import DATA_DIR, EPOCHS
from utils import calculate_mean_std
from model import Conv2Model

def train_epoch(model, device, train_loader, optimizer, criterion):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    with tqdm(total=len(train_loader)) as progress_bar:
        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.update(1)

        train_loss /= len(train_loader.dataset)
        train_accuracy = 100 * correct / total
        return train_loss, train_accuracy
    
def validate_epoch(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:

            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader.dataset)
        val_accuracy = 100 * correct / total
        return val_loss, val_accuracy
    
def train(model):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("Calculating train mean and std.")
    mean, std = calculate_mean_std()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=30),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.RandomGrayscale(p=0.1),
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std)
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean,
                            std=std)
    ])
    
    train_set = ImageFolder(DATA_DIR+'train/', transform=train_transform)
    train_size = int(0.8 * len(train_set))
    val_size = len(train_set) - train_size
    train_set, val_set = random_split(train_set, [train_size, val_size])

    test_set = ImageFolder(DATA_DIR+'test/', transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=32, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=False)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in range(EPOCHS): 

        train_loss, train_accuracy = train_epoch(model, device, train_loader, optimizer, criterion)

        val_loss, val_accuracy = validate_epoch(model, device, val_loader, criterion)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)
        
        print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
            f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%')

    test_loss, test_accuracy = validate_epoch(model, device, test_loader, criterion)
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%')

    return {
        "train_losses" : train_losses,
        "val_losses" : val_losses,
        "train_accs" : train_accs,
        "val_accs" : val_accs,
        "model" : model
    }