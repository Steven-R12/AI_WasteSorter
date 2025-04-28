import torch
import torch.nn as nn
import torchvision.models as models
from torch.quantization import quantize_dynamic

class WasteClassifier(nn.Module):
    def __init__(self, num_classes=3):
        super(WasteClassifier, self).__init__()
        # Using MobileNetV3 for better efficiency
        self.model = models.mobilenet_v3_small(pretrained=True)
        
        # Freeze all layers initially
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Replace classifier
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, num_classes)
        
        # Unfreeze last few layers for fine-tuning
        for param in self.model.features[-4:].parameters():
            param.requires_grad = True
        for param in self.model.classifier.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        return self.model(x)
    
    def quantize(self):
        # Dynamic quantization for Raspberry Pi deployment
        return quantize_dynamic(self, {nn.Linear}, dtype=torch.qint8)

def get_model(num_classes=3, quantized=False):
    model = WasteClassifier(num_classes=num_classes)
    if quantized:
        model = model.quantize()
    return model
