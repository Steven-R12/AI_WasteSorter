import torch
from torchvision import transforms
from PIL import Image
from model import get_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model
model = get_model(num_classes=3)
model.load_state_dict(torch.load("waste_sorter.pth", map_location=device))
model.to(device)
model.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Predict function
def predict(image_path):
    img = Image.open(image_path)
    img = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img)
        _, pred = torch.max(output, 1)
    
    class_names = ['Trash', 'Recycling', 'Compost']
    return class_names[pred.item()]

# Example
image_path = "dataset/test_image.jpg"
print(f"Predicted Class: {predict(image_path)}")

