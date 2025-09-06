import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class DefaultCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(DefaultCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        # 使用自适应平均池化确保固定输出大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)  # 确保固定大小
        x = self.classifier(x)
        return x

def get_model(model_name, num_classes=10):
    if model_name == 'default':
        return DefaultCNN(num_classes=num_classes)
    elif model_name == 'cnn':
        return DefaultCNN(num_classes=num_classes)
    elif model_name == 'mobilenetv2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
        return model
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"Unsupported model: {model_name}")