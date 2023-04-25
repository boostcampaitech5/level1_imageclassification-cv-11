# Gender and age classification for wearing mask image
This project is the Naver Boost Camp CV11 team's submission code for the mask wearing status classification competition.
When a picture of a person wearing a mask is given, it is a matter of classifying the person's age, sex, and whether he or she is wearing a mask.
### Team Members

<div align="left">
  <table>
    <tr>
      <td align="center">
        <a href="https://github.com/Mugamta/">
            <img src="https://avatars.githubusercontent.com/u/62493933?v=4" alt="서지훈 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/junha-lee">
          <img src="https://avatars.githubusercontent.com/u/44857783?v=4" alt="이준하 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/jiyoulee">
          <img src="https://avatars.githubusercontent.com/u/55631731?v=4" alt="이지유 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/Chaewon829">
          <img src="https://avatars.githubusercontent.com/u/126534080?v=4" alt="이채원 프로필" width=120 height=120 />
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/guk98">
          <img src="https://avatars.githubusercontent.com/u/78603611?v=4" alt="최지욱 프로필" width=120 height=120 />
        </a>
      </td>
    </tr>
    <tr>
      <td align="center">
        <a href="https://github.com/Mugamta/">
          서지훈
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/junha-lee">
          이준하
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/jiyoulee">
          이지유
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/Chaewon829">
          이채원
        </a>
      </td>
      <td align="center">
        <a href="https://github.com/guk98">
          최지욱
        </a>
      </td>
    </tr>
  </table>
</div>

<br/>
<div id="5"></div>
 
# Environment
- OS : Linux Ubuntu 18.04.5
- GPU : Tesla V100 (32GB)

# Usage
## 1. git clone
- `git clone https://github.com/boostcampaitech5/level1_imageclassification-cv-11.git`

## 2. Install Requirements
- `pip install -r requirements.txt`

## 3. Training
To solve this problem, we conducted 18 classifications including gender, age, and whether to wear a mask, experimented with multi-output-classicaiton by setting each as a separate output, and conducted many experiments in the training process, such as removing the background of a given image through segmentation or adding regression loss for age. 
The method of execution for this experiment is as follows.

### 3-1. single-output-classification
- `python train.py`

### 3-2. multi-output-classification
- `python train_multi.py`

#### Key Options
- `--save_dir` : save path
- `--data_dir` : input path
- `--use_age` : weight of mseloss(age)
- `--seg` : enable segmentation
- `--mislabel` : train with corrected label
- `--model` : model type (default: EfficientNet B0)
- `--batch_size` : batch size for training
- `--criterion` : criterion type
- `--epoch` : number of epochs to train

## 4. Inference
### 4-1. Inference without segmentation image
- `python inference.py`

#### Key Options
- `--data_dir` : input path
- `--model_dir` : model weight path
- `--output_dir` : output path
- `--is_multi` : enable multi output classification
- `--model` : model type (default: EfficientNet B0)
- `--batch_size` : input batch size for validing


# result
**Metric** : f1 score

| Model       | f1 score    |
| ----------- | ----------- |
| EfficientNet B0 | 0.93093 |
| ResNet 18  | 0.88177 |
| ResNet 34  | 0.93165 |
| EfficientNet B1 | 0.94594 |
| EfficientNet B2 | 0.95842 |
| ViT Tiny (Patch 16, 384) | 0.76766 |
| ViT Small (Patch 16, 384)  | 0.77515 |

**Best model : Efficientnet B2**
