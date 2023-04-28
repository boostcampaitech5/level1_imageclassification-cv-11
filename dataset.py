import os
import random
from collections import defaultdict
from enum import Enum
from typing import Tuple, List
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, Subset, random_split
from torchvision.transforms import Resize, ToTensor, Normalize, Compose, CenterCrop



class MaskLabels(int, Enum):
    MASK = 0
    INCORRECT = 1
    NORMAL = 2


class GenderLabels(int, Enum):
    MALE = 0
    FEMALE = 1

    @classmethod
    def from_str(cls, value: str) -> int:
        value = value.lower()
        if value == "male":
            return cls.MALE
        elif value == "female":
            return cls.FEMALE
        else:
            raise ValueError(f"Gender value should be either 'male' or 'female', {value}")


class AgeLabels(int, Enum):
    YOUNG = 0
    MIDDLE = 1
    OLD = 2

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        if value < 30:
            return cls.YOUNG
        elif value < 60:
            return cls.MIDDLE
        else:
            return cls.OLD
         
class AgeNumLabels(int, Enum):

    @classmethod
    def from_number(cls, value: str) -> int:
        try:
            value = int(value)
        except Exception:
            raise ValueError(f"Age value should be numeric, {value}")

        return value


class MaskBaseDataset(Dataset):
    num_classes = 3 * 2 * 3


    def __init__(self, data_dir, val_ratio=0.2, segmentation = False):
        self.data_dir = data_dir

        self.val_ratio = val_ratio
        
        if segmentation == False : 
            self._file_names = {
                "mask1": MaskLabels.MASK,
                "mask2": MaskLabels.MASK,
                "mask3": MaskLabels.MASK,
                "mask4": MaskLabels.MASK,
                "mask5": MaskLabels.MASK,
                "incorrect_mask": MaskLabels.INCORRECT,
                "normal": MaskLabels.NORMAL
            }
            self.mean = (0.548, 0.504, 0.479)
            self.std = (0.237, 0.247, 0.246)
        else : #segmentation =True
            self._file_names = {
                "seg_mask1": MaskLabels.MASK,
                "seg_mask2": MaskLabels.MASK,
                "seg_mask3": MaskLabels.MASK,
                "seg_mask4": MaskLabels.MASK,
                "seg_mask5": MaskLabels.MASK,
                "seg_incorrect_mask": MaskLabels.INCORRECT,
                "seg_normal": MaskLabels.NORMAL
            }
            self.mean =(0.22367465, 0.19352587, 0.18368957)
            self.std = (0.27654092, 0.24772727, 0.23997965)

        self.image_paths = []
        self.mask_labels = []
        self.gender_labels = []
        self.age_labels = []
        self.age_num_labels= []
        self.multi_class_label = []

        self.transform = None
        self.setup()
        self.calc_statistics()

    def setup(self):
        profiles = os.listdir(self.data_dir)
        for profile in profiles:
            if profile.startswith("."):  # "." 로 시작하는 파일은 무시합니다
                continue

            img_folder = os.path.join(self.data_dir, profile)
            for file_name in os.listdir(img_folder):
                _file_name, ext = os.path.splitext(file_name)
                if _file_name not in self._file_names:  # "." 로 시작하는 파일 및 invalid 한 파일들은 무시합니다
                    continue

                img_path = os.path.join(self.data_dir, profile, file_name)  # (resized_data, 000004_male_Asian_54, mask1.jpg)
                mask_label = self._file_names[_file_name]

                id, gender, race, age = profile.split("_")
                gender_label = GenderLabels.from_str(gender)
                age_label = AgeLabels.from_number(age)
                age_num_label = AgeNumLabels.from_number(age)

                self.image_paths.append(img_path)
                self.mask_labels.append(mask_label)
                self.gender_labels.append(gender_label)
                self.age_labels.append(age_label)
                self.age_num_labels.append(age_num_label)
                self.multi_class_label.append(mask_label * 6 + gender_label * 3 + age_label)

    def calc_statistics(self):
        has_statistics = self.mean is not None and self.std is not None
        if not has_statistics:
            print("[Warning] Calculating statistics... It can take a long time depending on your CPU machine")
            sums = []
            squared = []
            for image_path in self.image_paths[:3000]:
                image = np.array(Image.open(image_path)).astype(np.int32)
                sums.append(image.mean(axis=(0, 1)))
                squared.append((image ** 2).mean(axis=(0, 1)))

            self.mean = np.mean(sums, axis=0) / 255
            self.std = (np.mean(squared, axis=0) - self.mean ** 2) ** 0.5 / 255

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, index):
        assert self.transform is not None, ".set_tranform 메소드를 이용하여 transform 을 주입해주세요"

        image = self.read_image(index)
        mask_label = self.get_mask_label(index)
        gender_label = self.get_gender_label(index)
        age_label = self.get_age_label(index)
        age_num_label = self.get_age_num_label(index)
        multi_class_label = self.multi_class_label[index]

        image_transform = self.transform(image=image)
        return image_transform, multi_class_label, age_num_label

    def __len__(self):
        return len(self.image_paths)

    def get_mask_label(self, index) -> MaskLabels:
        return self.mask_labels[index]

    def get_gender_label(self, index) -> GenderLabels:
        return self.gender_labels[index]

    def get_age_label(self, index) -> AgeLabels:
        return self.age_labels[index]

    def get_age_num_label(self,index) -> AgeNumLabels:
        return self.age_num_labels[index]
    
    def read_image(self, index):
        image_path = self.image_paths[index]
        return np.array(Image.open(image_path))

    @staticmethod
    def decode_multi_class(multi_class_label) -> Tuple[MaskLabels, GenderLabels, AgeLabels]:
        mask_label = (multi_class_label // 6) % 3
        gender_label = (multi_class_label // 3) % 2
        age_label = multi_class_label % 3
        return mask_label, gender_label, age_label

    @staticmethod
    def denormalize_image(image, mean, std):
        img_cp = image.copy()
        img_cp *= std
        img_cp += mean
        img_cp *= 255.0
        img_cp = np.clip(img_cp, 0, 255).astype(np.uint8)
        return img_cp

    def split_dataset(self, dataset, k = 5) -> Tuple[Subset, Subset]:
        train_datasets = []
        val_datasets=[]
        skf = StratifiedKFold(n_splits=k,shuffle=False)
        for train_indices, val_indices in skf.split(dataset, np.array(dataset.multi_class_label)%6):
            train_datasets.append(torch.utils.data.Subset(dataset, train_indices))
            val_datasets.append(torch.utils.data.Subset(dataset, val_indices))
        """
        데이터셋을 train 과 val 로 나눕니다,
        pytorch 내부의 torch.utils.data.random_split 함수를 사용하여
        torch.utils.data.Subset 클래스 둘로 나눕니다.
        구현이 어렵지 않으니 구글링 혹은 IDE (e.g. pycharm) 의 navigation 기능을 통해 코드를 한 번 읽어보는 것을 추천드립니다^^
        """
        #n_val = int(len(self) * self.val_ratio)
        #n_train = len(self) - n_val
        #train_set, val_set = random_split(self, [n_train, n_val])
        return train_datasets, val_datasets

class TestDataset(Dataset):
    def __init__(self, img_paths, resize, segmentation = False):
        self.img_paths = img_paths
        if segmentation == False : 
            self.mean = (0.548, 0.504, 0.479)
            self.std = (0.237, 0.247, 0.246)
        else : 
            self.mean = (0.22367465, 0.19352587, 0.18368957)
            self.std = (0.27654092, 0.24772727, 0.23997965)
        self.transform = Compose([
            Resize(resize, Image.BILINEAR),
            ToTensor(),
            Normalize(mean=self.mean, std=self.std),
        ])


    def __getitem__(self, index):
        image = Image.open(self.img_paths[index])

        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.img_paths)
