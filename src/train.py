import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets, models
import os
from time import time
from tqdm.auto import tqdm  # Better progress bars
import matplotlib.pyplot as plt

# Configuration (MODIFIED FOR PERFORMANCE)
class Config:
    data_root = "/content/AI_WasteSorter/data"
    batch_size = 64  # Increased from 32 → 64
    epochs = 20
    lr = 0.001
    num_classes = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_save_name = "waste_classifier.pth"
    num_workers = 4  # Parallel data loading
    early_stop_patience = 3

# Data Augmentation (UNCHANGED)
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

# Dataset Loading (ADDED VALIDATION)
def load_datasets():
    train_dir = os.path.join(Config.data_root, "train")
    val_dir = os.path.join(Config.data_root, "val")
    
    # Verify class folders
    required_folders = {'Compost', 'Recycling', 'Trash'}
    assert required_folders.issubset(os.listdir(train_dir)), "Train folder missing classes"
    assert required_folders.issubset(os.listdir(val_dir)), "Val folder missing classes"

    train_data = datasets.ImageFolder(train_dir, transform=train_transform)
    val_data = datasets.ImageFolder(val_dir, transform=val_transform)
    
    print(f"\nDataset loaded:")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Classes: {train_data.classes}\n")
    
    return train_data, val_data

# Model Initialization (UPDATED FOR MOBILENET_V3)
def create_model():
    model = models.mobilenet_v3_small(weights='DEFAULT')  # Modern weights
    for param in model.parameters():
        param.requires_grad = False
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, Config.num_classes)
    return model.to(Config.device)

# Training Loop (OPTIMIZED)
def train_model():
    train_data, val_data = load_datasets()
    
    # Faster DataLoaders (MODIFIED)
    train_loader = DataLoader(
        train_data,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True  # Speeds up GPU transfer
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
    history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    print(f"\nTraining on {Config.device}...")
    print(f"Batch size: {Config.batch_size} | Workers: {Config.num_workers}\n")

    for epoch in range(Config.epochs):
        # Training
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
            train_loop.set_postfix(loss=loss.item())  # Live loss display

        # Validation and logging (UNCHANGED)
        # ... (keep your existing validation code) ...

    print(f"\nTraining complete. Best val accuracy: {best_acc:.2f}%")
    torch.save(model.state_dict(), Config.model_save_name)

if __name__ == "__main__":
    # Enable CUDA debugging if needed
    import os
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1" 
    train_model()
