import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, models, datasets
import matplotlib.pyplot as plt
import numpy as np
import os
from time import time

# Configuration
class Config:
    data_root = "/content/AI_WasteSorter/data"  # Google Drive symlink location
    batch_size = 32
    epochs = 15
    lr = 0.001
    num_classes = 3  # compost, recycle, trash
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_name = "trash_classifier_resnet18.pth"

# Data Augmentation
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load Datasets
def load_datasets():
    train_dir = os.path.join(Config.data_root, "train")
    val_dir = os.path.join(Config.data_root, "val")
    
    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    
    print(f"Found {len(train_data)} training images in {len(train_data.classes)} classes")
    print(f"Found {len(val_data)} validation images")
    
    return train_data, val_data

# Initialize Model
def create_model():
    model = models.resnet18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, Config.num_classes)
    return model.to(Config.device)

# Training Function
def train_and_validate():
    train_data, val_data = load_datasets()
    
    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size)
    
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    
    # Track metrics
    train_loss_history = []
    val_loss_history = []
    train_acc_history = []
    val_acc_history = []
    
    best_val_acc = 0.0
    
    print(f"\nTraining on {Config.device}...")
    
    for epoch in range(Config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_name)
        
        # Print epoch stats
        print(f"Epoch {epoch+1}/{Config.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Plot metrics
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_history, label='Train')
        plt.plot(val_loss_history, label='Validation')
        plt.title('Loss Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(train_acc_history, label='Train')
        plt.plot(val_acc_history, label='Validation')
        plt.title('Accuracy Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nTraining complete. Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Model saved as '{Config.model_save_name}'")

if __name__ == "__main__":
    train_and_validate()
