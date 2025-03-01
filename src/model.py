import torch
import torch.nn as nn
import torchvision.models as models

def get_model(num_classes=3):
    model = models.mobilenet_v2(pretrained=True)
    
    # Freeze feature extractor layers
    for param in model.features.parameters():
        param.requires_grad = False
    
    # Modify the classifier
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    
    return model

