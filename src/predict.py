import torch
from torchvision import transforms
from PIL import Image
from model import get_model
import numpy as np
import time

class WasteSorter:
    def __init__(self, model_path="waste_classifier_mobilenetv3.pth"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = get_model(quantized=self.device.type == 'cpu')
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
        
        self.class_names = ['Compost', 'Recycling', 'Trash']
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
    def predict(self, image):
        """Predict waste class from PIL Image or file path"""
        if isinstance(image, str):
            img = Image.open(image).convert('RGB')
        else:
            img = image.convert('RGB')
            
        img_t = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            start_time = time.time()
            outputs = self.model(img_t)
            _, pred = torch.max(outputs, 1)
            inference_time = time.time() - start_time
        
        # Get confidence scores
        probs = torch.nn.functional.softmax(outputs, dim=1)[0] * 100
        confidences = {self.class_names[i]: f"{probs[i].item():.1f}%" 
                      for i in range(len(self.class_names))}
        
        return {
            'class': self.class_names[pred.item()],
            'confidence': confidences,
            'inference_time': f"{inference_time*1000:.1f}ms"
        }

# Example usage
if __name__ == "__main__":
    sorter = WasteSorter()
    
    # Test with an image
    result = sorter.predict("test_image.jpg")
    print("\nPrediction Result:")
    print(f"Class: {result['class']}")
    print(f"Confidences: {result['confidence']}")
    print(f"Inference Time: {result['inference_time']}")
