import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class Config:
    data_root = "/content/AI_WasteSorter/data"
    batch_size = 64
    epochs = 20
    lr = 0.001
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_name = "waste_classifier.pth"
    num_workers = 4
    early_stop_patience = 5

def get_transforms():
    return {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
    """Load datasets with integrity checks"""
    train_dir = os.path.join(Config.data_root, "train")
    val_dir = os.path.join(Config.data_root, "val")
    
    # Verify folder structure
    required_folders = {'Compost', 'Recycling', 'Trash'}
    assert required_folders.issubset(os.listdir(train_dir)), "Missing class folders in train"
    assert required_folders.issubset(os.listdir(val_dir)), "Missing class folders in val"

    train_data = datasets.ImageFolder(train_dir, transform=get_transforms()['train'])
    val_data = datasets.ImageFolder(val_dir, transform=get_transforms()['val'])
    
    # Critical: Verify class alignment
    assert train_data.class_to_idx == val_data.class_to_idx, \
        f"Class mismatch!\nTrain: {train_data.class_to_idx}\nVal: {val_data.class_to_idx}"
    
    print(f"\nDataset loaded (Classes: {train_data.classes})")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    return train_data, val_data

def create_model():
    """Initialize model with modern weights"""
    model = models.mobilenet_v3_small(weights='DEFAULT')
    # Freeze all layers except classifier
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, Config.num_classes)
    return model.to(Config.device)

def train_model():
    train_data, val_data = load_datasets()
    
    train_loader = DataLoader(
        train_data,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_data,
        batch_size=Config.batch_size,
        num_workers=Config.num_workers
    )
    
    model = create_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=Config.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)
    
    best_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nTraining on {Config.device}...")
    print(f"Using {len(train_loader)} batches per epoch")

    for epoch in range(Config.epochs):
        # Training Phase
        model.train()
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs} - Train")
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loop:
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
            train_loop.set_postfix(loss=loss.item())
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation Phase (CRITICAL FIX)
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100 * val_correct / val_total
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), Config.model_save_name)
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Print epoch stats
        print(f"\nEpoch {epoch+1}/{Config.epochs}:")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        print(f"Best Val Acc: {best_acc:.2f}%")
        
        # Early stopping
        if patience_counter >= Config.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
        
        scheduler.step(val_acc)
    
    print(f"\nTraining complete. Best validation accuracy: {best_acc:.2f}%")
    print(f"Model saved to {Config.model_save_name}")

if __name__ == "__main__":
    # Debugging aids
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # Better error messages
    
    # Verify GPU
    print(f"Using device: {Config.device}")
    if Config.device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    train_model()
