import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets
import os
from time import time
from tqdm import tqdm
import matplotlib.pyplot as plt
from model import WasteClassifier

class Config:
    data_root = "/content/AI_WasteSorter/data"  # Update this path
    batch_size = 32
    epochs = 20
    lr = 0.001
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_name = "waste_classifier_mobilenetv3.pth"
    early_stop_patience = 5

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

def load_datasets():
    transforms = get_transforms()
    train_data = datasets.ImageFolder(os.path.join(Config.data_root, 'train'), transform=transforms['train'])
    val_data = datasets.ImageFolder(os.path.join(Config.data_root, 'val'), transform=transforms['val'])
    
    print(f"\nDataset loaded:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Classes: {train_data.classes}\n")
    
    return train_data, val_data

def train_model():
    train_data, val_data = load_datasets()
    
    train_loader = DataLoader(train_data, batch_size=Config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_data, batch_size=Config.batch_size, num_workers=2)
    
    model = WasteClassifier(Config.num_classes).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nTraining on {Config.device}...")
    
    for epoch in range(Config.epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs} - Train"):
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
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{Config.epochs} - Val"):
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * correct / total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_name)
            patience_counter = 0
            print(f"New best model saved with val_acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            if patience_counter >= Config.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Print epoch stats
        print(f"\nEpoch {epoch+1}/{Config.epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Plot training history
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train')
        plt.plot(history['val_acc'], label='Validation')
        plt.title('Accuracy Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved as '{Config.model_save_name}'")

if __name__ == "__main__":
    train_model()
