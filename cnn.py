import torch.nn as nn
import torchvision.models as tvm

class SmallCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2), nn.Dropout(0.1),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Dropout(0.2), nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.net(x); return self.fc(x)

def build_model(name: str = "resnet18", num_classes: int = 10):
    name = name.lower()
    if name == "smallcnn":
        return SmallCNN(num_classes)
    elif name == "resnet18":
        m = tvm.resnet18(weights=None)  # start from scratch for CIFAR-10
        m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        m.maxpool = nn.Identity()
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")
