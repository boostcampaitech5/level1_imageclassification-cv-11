import torch.nn as nn
import torch.nn.functional as F
import timm

############################## Backbone Models ##############################

class EfficientBase(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.net = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)

        return out

class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        
        self.net = timm.create_model('resnet18', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)
        
        return out

class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = timm.create_model('resnet34', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)

        return out

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = timm.create_model('efficientnet_b1', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)

        return out

class EfficientNetB2(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = timm.create_model('efficientnet_b2', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)

        return out

class ViTTiny_Patch16_384(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = timm.create_model('vit_tiny_patch16_384', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)

        return out

class ViTSmall_Patch16_384(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()

        self.net = timm.create_model('vit_small_patch16_384', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.net(x)

        return out

############################## Custom Output Models ##############################

class SingleOutputModel(nn.Module):
    def __init__(self, in_features=1000, model=EfficientBase()):
        super().__init__()

        self.backbone = model

        self.branch_class = nn.Linear(in_features=in_features, out_features=18)
        self.branch_age_val = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        out = self.backbone(x)
        
        out_class = self.branch_class(out)
        out_age_num = self.branch_age_val(out)

        return out_class, out_age_num

class MultiOutputModel(nn.Module):
    def __init__(self, in_features=1000, model=EfficientBase()):
        super().__init__()

        self.backbone = model
        
        self.branch_mask = nn.Linear(in_features=in_features, out_features=3)
        self.branch_gender = nn.Linear(in_features=in_features, out_features=2)
        self.branch_age_class = nn.Linear(in_features=in_features, out_features=3)
        self.branch_age_val = nn.Linear(in_features=in_features, out_features=1)

    def forward(self, x):
        out = self.backbone(x)

        out_mask = self.branch_mask(out)
        out_gender = self.branch_gender(out)
        out_age_class = self.branch_age_class(out)
        out_age_num = self.branch_age_val(out)
        
        return out_mask, out_gender, out_age_class, out_age_num
