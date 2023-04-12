### 개요
우선 모델의 전처리 - 증강 - 학습 - 제출까지를 테스트하는 것이 목적이였다. <br/>
때문에 미션과 베이스라인의 코드를 거의 그대로 사용하여 특별한 부분은 없다. <br/>

1. 전처리 <br/>
기본적인 코드를 가져왔기 때문에, transform 부분도 동일하다. <br/>
때문에 전처리는 아래와 같다.
```
Resize(img_size[0], img_size[1]),
Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
ToTensorV2(p=1.0),
```
2. 증강 <br/>
전처리와 마찬가지로, 미션의 코드와 동일하다. <br/>
단, 현재 코드는 train dataset과 validation dataset의 transform에 오류가 있다. <br/>
이는 dataset으로 나누는 과정에서 얕은 복사를 하기 때문이라고 한다.
```
HorizontalFlip(p=0.5),ShiftScaleRotate(p=0.5),
HueSaturationValue(hue_shift_limit=0.2, sat_shift_limit=0.2, val_shift_limit=0.2, p=0.5),
RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.5),
GaussNoise(p=0.5),
```
3. 모델 <br/>
모델은 AlexNet을 사용했으며, Classfier 부분만 수정하였다.
```
model = alexnet(pretrained = True)
model.classifier = nn.Sequential(
    nn.Dropout(),
    nn.Linear(256 * 6 * 6, 4096),
    nn.ReLU(True),
    nn.Dropout(),
    nn.Linear(4096, 4096),
    nn.ReLU(True),
    nn.Linear(4096, 18),
)

model.features.requires_grad_(False)
```
4. loss && optimizer<br/>
```
nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```
5. epoch
```
num_epochs = 30
```
6. F1 score/accuracy
```
F1 score = 0.5446	accuracy = 61.9524	
```
