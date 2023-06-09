import argparse
import multiprocessing
import os
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
import model as Models


def load_model(saved_model, device):
    model_cls = getattr(import_module("model"), args.model)(num_classes=1000)

    # -- multi output model
    if args.is_multi:
        model = Models.MultiOutputModel(model=model_cls)
    
    # -- single output model
    else:
        model = Models.SingleOutputModel(model=model_cls)


    model_path = os.path.join(saved_model, 'best.pth')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    """
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = load_model(model_dir, device).to(device)
    model.eval()
    if args.seg == False :
        img_root = os.path.join(data_dir, 'images')
    else : #segmentation True
        img_root = os.path.join(data_dir, 'seg_images')
    info_path = os.path.join(data_dir, 'info.csv')
    info = pd.read_csv(info_path)

    img_paths = [os.path.join(img_root, img_id) for img_id in info.ImageID]
    dataset = TestDataset(img_paths, args.resize, args.seg)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("Calculating inference results..")
    preds = []
    with torch.no_grad():
        for idx, images in enumerate(loader):
            images = images.to(device)

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

            preds.extend(pred.cpu().numpy())

    info['ans'] = preds
    save_path = os.path.join(output_dir, f'output.csv')
    info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--resize', type=tuple, default=(512, 384), help='resize size for image when you trained (default: (512, 384))')
    parser.add_argument('--model', type=str, default='EfficientBase', help='model type (default: EfficientBase)')
    parser.add_argument('--is_multi', type=bool, default=False, help='enable multi output classification (default: false)')
    parser.add_argument('--seg', type=bool, default = False, help='segmentation (default = False)')
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
