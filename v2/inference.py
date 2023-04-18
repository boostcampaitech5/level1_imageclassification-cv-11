import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset, MaskBaseDataset
import model as Models

from albumentations import *
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

<<<<<<< HEAD
def load_model(saved_model, device):
    model_cls = getattr(import_module("model"), args.model)(num_classes=1000)

    # -- multi output model
    if args.is_multi:
        model = Models.MultiOutputModel(model=model_cls)
    
    # -- single output model
    else:
        model = Models.SingleOutputModel(model=model_cls)
=======
>>>>>>> f29f573... add inference.py

def load_model(saved_model, num_classes, device):
    model_module = getattr(import_module("model"), args.model)(num_classes=1000)
    model = Models.SingleOutputModel(model=model_module).to(device)
    # tarpath = os.path.join(saved_model, 'best.tar.gz')
    # tar = tarfile.open(tarpath, 'r:gz')
    # tar.extractall(path=saved_model)
    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # num_classes = MaskBaseDataset.num_classes  # 18
    model = load_model(model_dir, device).to(device)
    model.eval()

    img_root = os.path.join(data_dir, 'eval_segimages')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
<<<<<<< HEAD
    dataset = TestDataset(img_paths, args.resize)
    
    loader = torch.utils.data.DataLoader(
=======

    mean, std = (0.56019265, 0.52410305, 0.50145299), (0.23308824, 0.24294489, 0.2456003)

    transform = Compose([
        Resize(512, 384),
        RandomCrop(384, 384),
        Normalize(mean=mean, std=std, max_pixel_value=255.0, p=1.0),
        ToTensorV2(p=1.0),
    ], p=1.0)

    dataset = TestDataset(img_paths, transform)

    loader = DataLoader(
>>>>>>> f29f573... add inference.py
        dataset,
        shuffle=False
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in tqdm(enumerate(loader)):
            images = images.to(device)
<<<<<<< HEAD

            # -- multi output model
            if args.is_multi:
                pred_mask, pred_gender, pred_age, _ = model(images)

                pred_mask = pred_mask.argmax(1).detach()
                pred_gender = pred_gender.argmax(1).detach()
                pred_age = pred_age.argmax(1).detach()

                pred = pred_mask * 6 + pred_gender * 3 + pred_age

            # -- single output model
            else:
                pred, _ = model(images)
            
                pred = pred.argmax(dim=-1)

=======
            outs, age_num_outs = model(images)
            pred = torch.argmax(outs, dim=-1)
>>>>>>> f29f573... add inference.py
            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
<<<<<<< HEAD
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--resize', type=tuple, default=(512, 384), help='resize size for image when you trained (default: (512, 384))')
    parser.add_argument('--model', type=str, default='EfficientBase', help='model type (default: EfficientBase)')
    parser.add_argument('--is_multi', type=bool, default='false', help='enable multi output classification (default: false)')
=======
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=[128, 96], help='resize size for image when you trained (default: (96, 128))')
    parser.add_argument('--model', type=str, default='EfficientBase', help='model type (default: BaseModel)')
>>>>>>> f29f573... add inference.py

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', './model/exp'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)
