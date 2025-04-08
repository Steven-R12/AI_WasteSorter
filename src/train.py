import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from model import get_model

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load dataset
train_data = datasets.ImageFolder('dataset/train', transform=transform)
val_data = datasets.ImageFolder('dataset/val', transform=transform)

# Dataloaders
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

# Load model
model = get_model(num_classes=3).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)

# TensorBoard writer
writer = SummaryWriter()

# Train the model
num_epochs = 13
for epoch in range(num_epochs):
    model.train()
    running_loss, correct, total = 0, 0, 0

    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += torch.sum(preds == labels).item()
        total += labels.size(0)

        # ✅ Log loss after every batch
        writer.add_scalar('Loss/train', loss.item(), epoch * len(train_loader) + i)

    # ✅ Print epoch loss and accuracy
    train_accuracy = correct / total * 100
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/total:.4f}, Accuracy: {train_accuracy:.2f}%")

# Save model
torch.save(model.state_dict(), "waste_sorter.pth")
writer.close()
