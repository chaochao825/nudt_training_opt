import os
import torch
import torch.nn as nn
import torchvision.models as models

def get_model(model_name, num_classes=10, pretrained_path=None):
    if model_name.lower() == 'vgg16' or model_name.lower() == 'vgg':
        model = models.vgg16(pretrained=False)
        model.classifier[6] = nn.Linear(4096, num_classes)
    elif model_name.lower() == 'resnet50' or model_name.lower() == 'resnet':
        model = models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name.lower() == 'inception_v3' or model_name.lower() == 'inception':
        model = models.inception_v3(pretrained=False, aux_logits=True)
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, num_classes)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    if pretrained_path and os.path.exists(pretrained_path):
        state_dict = torch.load(pretrained_path, map_location='cpu')
        # Handle cases where state_dict might be wrapped
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        
        # Load state dict with strict=False to allow for different fc layers
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained weights from {pretrained_path}")

    return model

