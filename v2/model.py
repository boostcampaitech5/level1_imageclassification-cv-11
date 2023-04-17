import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

classification_models = models.list_models(module=models)

class BaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.25)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout2(x)

        x = self.avgpool(x)
        x = x.view(-1, 128)
        return self.fc(x)


# Custom Model Template
class EfficientBase(nn.Module):
    def __init__(self, num_classes=18):
        super().__init__()
        import timm
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        out = self.backbone(x)
       
        return out
    
class ResNet18(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.net = models.get_model('ResNet18', weights='DEFAULT')

    def forward(self, x):
        out = self.net(x)
       
        return out

class ResNet34(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = models.get_model('ResNet34', weights='DEFAULT')

    def forward(self, x):
        out = self.net(x)

        return out

class EfficientNetB1(nn.Module):
    def __init__(self):
        super().__init__()

        self.net = models.get_model('EfficientNet_B1', weights='DEFAULT')

    def forward(self, x):
        out = self.net(x)

        return out

# class MultiLabelModel(nn.module):
#     def __init__(self):
#         super().__init__()
#         from torchvision.models import efficientnet_b4
#         self.backbone = efficientnet_b4(pretrained=True)
#         self.backbone.classifier[1]= nn.Linear(1792,1792)
#         self.branch_age_class = nn.Linear(in_features=1792, out_features=3)
#         self.branch_age_val = nn.Linear(in_features=1792, out_features=1)
#         self.branch_mask = nn.Linear(in_features=1792, out_features=3)
#         self.branch_gender = nn.Linear(in_features=1792, out_features=2)
#                 """
#         1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
#         2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
#         3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
#         """

        
#     def forward(self, x):
#          """
#         1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
#         2. 결과로 나온 output 을 return 해주세요
#         """
#         out = self.backbone(x)
#         out_mask = self.branch_mask(out)
#         out_gender = self.branch_gender(out)
#         out_age_class = self.branch_age_class(out)
#         out_age_num = self.branch_age_val(out)
        
#         return out_mask, out_gender, out_age_class, out_age_num
