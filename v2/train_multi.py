import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import MaskBaseDataset
from loss import create_criterion
import augmentation
import model as Models
import wandb
from sklearn.metrics import f1_score, accuracy_score


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
def get_k(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['fold']

def grid_image(np_images, gts, preds, n=16, shuffle=False):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))  # cautions: hardcoded, 이미지 크기에 따라 figsize 를 조정해야 할 수 있습니다. T.T
    plt.subplots_adjust(top=0.8)  # cautions: hardcoded, 이미지 크기에 따라 top 를 조정해야 할 수 있습니다. T.T
    n_grid = int(np.ceil(n ** 0.5))
    tasks = ["mask", "gender", "age"]
    for idx, choice in enumerate(choices):
        gt = gts[choice].item()
        pred = preds[choice].item()
        image = np_images[choice]
        gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
        pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
        title = "\n".join([
            f"{task} - gt: {gt_label}, pred: {pred_label}"
            for gt_label, pred_label, task
            in zip(gt_decoded_labels, pred_decoded_labels, tasks)
        ])

        plt.subplot(n_grid, n_grid, idx + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(image, cmap=plt.cm.binary)

    return figure

def increment_path(path, exist_ok=False):
    """ Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

    Args:
        path (str or pathlib.Path): f"{model_dir}/{args.name}".
        exist_ok (bool): whether increment path (increment if False).
    """
    path = Path(path)
    if (path.exists() and exist_ok) or (not path.exists()):
        return str(path)
    else:
        dirs = glob.glob(f"{path}*")
        matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]
        n = max(i) + 1 if i else 2
        return f"{path}{n}"

def train(data_dir, model_dir, save_dir, args):
    seed_everything(args.seed)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
    )
    
    mean, std = (0.56019265, 0.52410305, 0.50145299), (0.23308824, 0.24294489, 0.2456003)
    
    train_dataset = MaskBaseDataset(
        data_dir=data_dir,
        mean=mean,
        std=std,
    )

    val_dataset = MaskBaseDataset(
        data_dir=data_dir,
        mean=mean,
        std=std,
    )
    
    num_classes = dataset.num_classes  # 18

    # -- augmentation
    transform = augmentation.get_transforms()  # default: BaseAugmentation

    train_dataset.set_transform(transform['train'])
    val_dataset.set_transform(transform['val'])

    # -- data_loader
    train_sets, _ = train_dataset.split_dataset(train_dataset)
    _ , val_sets = val_dataset.split_dataset(val_dataset)
    
    def fold(k):
        train_loader = DataLoader(
            train_sets[k],
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=True,
            pin_memory=use_cuda,
            drop_last=True,
        )

        val_loader = DataLoader(
            val_sets[k],
            batch_size=args.valid_batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=True,
        )
        return train_loader, val_loader

    # -- model
    model_module = getattr(import_module("model"), args.model)(num_classes=1000)
    opt_module = getattr(import_module("torch.optim"), args.optimizer)
    
    i = 0
    train_loader, val_loader = fold(i)
    
    model = Models.MultiOutputModel(in_features=1000, model=model_module).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion_mask = create_criterion(args.criterion)   # default: cross_entropy
    criterion_gender = create_criterion(args.criterion)
    criterion_age = create_criterion(args.criterion)
    criterion_mse = create_criterion('mse')             # age regression loss
    
    # set alpha value to penalize age MSE loss
    alpha = args.use_age
    
    # opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    # -- logging
    logger = SummaryWriter(log_dir=save_dir)
    with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)

    best_val_f1 = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        
        category_preds = []
        category_trues = []
        
        train_loss_all =[]
        train_loss1 = []
        train_loss2 = []
        train_loss3 = []
        train_loss4 = []
        
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in tqdm(enumerate(train_loader)):
            inputs, labels, age_num_labels = train_batch
            
            mask_labels = ((labels // 6) % 3)
            gender_labels = ((labels // 3) % 2)
            age_labels = (labels % 3)
            
            inputs = inputs['image'].to(device)
            mask_labels = mask_labels.to(device)
            gender_labels = gender_labels.to(device)
            age_labels = age_labels.to(device)
            age_num_labels = age_num_labels.to(device)

            optimizer.zero_grad()

            mask_outs, gender_outs, age_outs, age_num_outs = model(inputs)
            
            # preds = torch.argmax(outs, dim=-1)
            loss1 = criterion_mask(mask_outs, mask_labels)
            loss2 = criterion_gender(gender_outs, gender_labels)
            loss3 = criterion_age(age_outs, age_labels)
            loss4 = criterion_mse(age_num_outs, age_num_labels.unsqueeze(1).float()) * alpha

            loss = loss1 + loss2 + loss3 + loss4
            loss.backward()
            optimizer.step()

            loss_value += loss.item()
            # mask_matches += (mask_outs.argmax(1).detach().cpu().numpy().tolist() == mask_labels).sum().item()
            # gender_matches += (gender_outs.argmax(1).detach().cpu().numpy().tolist() == gender_labels).sum().item()
            # age_matches += (age_outs.argmax(1).detach().cpu().numpy().tolist() == age_labels).sum().item()
            pred1 = mask_outs.argmax(1).detach().cpu().numpy()
            pred2 = gender_outs.argmax(1).detach().cpu().numpy()
            pred3 = age_outs.argmax(1).detach().cpu().numpy()

            matches += (torch.Tensor(pred1*6+pred2*3+pred3)==labels).sum().item()
            
            category_preds += (pred1*6+pred2*3+pred3).tolist()
            category_trues += labels.detach().cpu().numpy().tolist()
            
            train_loss_all.append(loss.item())
            train_loss1.append(loss1.item())
            train_loss2.append(loss2.item())
            train_loss3.append(loss3.item())
            train_loss4.append(loss4.item())
            
            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
                logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                loss_value = 0
                matches = 0

        scheduler.step()

        # val loop
        with torch.no_grad():
            print("Calculating validation results...")
            model.eval()
            val_loss_items = []
            val_acc_items = []
            figure = None
            
            val_category_preds = []
            val_category_trues = []
            
            val_loss = []
            val_loss1 = []
            val_loss2 = []
            val_loss3 = []
            val_loss4 = []
            
            
            for val_batch in val_loader:
                inputs, labels, age_num_labels = val_batch
                
                mask_labels = ((labels // 6) % 3)
                gender_labels = ((labels // 3) % 2)
                age_labels = (labels % 3)
                
                inputs = inputs['image'].to(device)
                mask_labels = mask_labels.to(device)
                gender_labels = gender_labels.to(device)
                age_labels = age_labels.to(device)
                age_num_labels = age_num_labels.to(device)

                                
                mask_outs, gender_outs, age_outs, age_num_outs = model(inputs)
                
                # preds = torch.argmax(outs, dim=-1)
                loss1 = criterion_mask(mask_outs, mask_labels)
                loss2 = criterion_gender(gender_outs, gender_labels)
                loss3 = criterion_age(age_outs, age_labels)
                loss4 = criterion_mse(age_num_outs, age_num_labels.unsqueeze(1).float()) * alpha
                
                loss = loss1 + loss2 + loss3 + loss4

                loss_item = loss.item()
                
                val_loss1.append(loss1.item())
                val_loss2.append(loss2.item())
                val_loss3.append(loss3.item())
                val_loss4.append(loss4.item())


                
                pred1 = mask_outs.argmax(1).detach().cpu().numpy()
                pred2 = gender_outs.argmax(1).detach().cpu().numpy()
                pred3 = age_outs.argmax(1).detach().cpu().numpy()
                
                acc_item = (torch.Tensor(pred1*6+pred2*3+pred3)==labels).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item/val_loader.batch_size)
                
                val_category_preds += (pred1*6+pred2*3+pred3).tolist()
                val_category_trues += labels.detach().cpu().numpy().tolist()

                if figure is None:
                    preds = torch.Tensor(pred1*6+pred2*3+pred3)
                    inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                    inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                    figure = grid_image(
                        inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset"
                    )

            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.mean(val_acc_items)
            best_val_loss = min(best_val_loss, val_loss)
            
            _train_f1 = f1_score(category_trues, category_preds, average='macro')
            _train_accuracy = accuracy_score(category_trues, category_preds)

            _train_loss = np.mean(train_loss_all)
            _train_loss1 = np.mean(train_loss1)
            _train_loss2 = np.mean(train_loss2)
            _train_loss3 = np.mean(train_loss3)
            _train_loss4 = np.mean(train_loss4)            
            _val_loss1 = np.mean(val_loss1)
            _val_loss2 = np.mean(val_loss2)
            _val_loss3 = np.mean(val_loss3)            
            _val_loss4 = np.mean(val_loss4)
            
            _val_f1 = f1_score(val_category_trues,val_category_preds, average='macro')

            wandb.log({'Train loss':_train_loss,'Train F1':_train_f1,'Train Acc':_train_accuracy,'Train loss 1' : _train_loss1,'Train loss 2' : _train_loss2,'Train loss 3' : _train_loss3,'Train loss 4':_train_loss4,'Train loss 123':_train_loss-_train_loss4,
                   'Val loss':val_loss,'Val F1':_val_f1, 'Val Acc':val_acc,'Val loss 1' : _val_loss1,'Val loss 2' : _val_loss2,'Val loss 3' : _val_loss3,'Val loss 4':_val_loss4,'Val loss 123':val_loss-_val_loss4,'Val best f1':max(_val_f1, best_val_f1)})      ## logging wandb
            
            
            
            if _val_f1 > best_val_f1:
                print(f"New best model for val f1 : {_val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                best_val_f1 = _val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] f1 : {_val_f1:4.2%}, loss: {val_loss:4.2} || "
                f"best f1 : {best_val_f1:4.2%}, best loss: {best_val_loss:4.2}"
            )
            logger.add_scalar("Val/loss", val_loss, epoch)
            logger.add_scalar("Val/f1", _val_f1, epoch)
            logger.add_figure("results", figure, epoch)
            print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=15, help='number of epochs to train (default: 15)')
    parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='EfficientBase', help='model type (default: EfficientBase)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--fold', default=1, help = 'kfold')
    parser.add_argument('--use_age', type=float, default=0, help='weight of mseloss(age) (default: 0)')
    parser.add_argument('--seg', type=bool, default=False, help='enable segmentation (default: False)')
    parser.add_argument('--mislabel', type=bool, default=False, help='train with corrected label (default: False)')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    save_dir = increment_path(os.path.join(model_dir, args.name))
    
    config = {'save_dir' : save_dir,
            'use_age': args.use_age,
            'multi':True, 
            'seg':args.seg,
            'mislabel': args.mislabel,
            'model':args.model,
            'augmentation':args.augmentation,
            'batch_size':args.batch_size,
            'criterion':args.criterion,
            'epoch':args.epochs,
            'fold':args.fold,
            'lr':args.lr,
            'lr_decay_step':args.lr_decay_step,
            'optimizer':args.optimizer,
            'resize':str(args.resize[0])+'X'+str(args.resize[1]),
            'seed':args.seed}
    
    dir = save_dir.split('/')[-1]
    
    # # wandb.init(project="test",name=f'use_age:{args.use_age}-multi:0-seg:{args.seg}-mislabel:{args.mislabel}-model:{args.model}-augmentation:{args.augmentation}-batch_size:{args.batch_size}-criterion:{args.criterion}-epoch:{args.epochs}-fold:{args.fold}-lr:{args.lr}-lr_decay_step:{args.lr_decay_step}-optimizer:{args.optimizer}-resize:{args.resize[0]}X{args.resize[1]}-seed:{args.seed}')
    wandb.init(project='image-classification-challenge',name=f'use_age:{args.use_age}-multi:{True}-seg:{args.seg}-mislabel:{args.mislabel}-model:{args.model}-exp:{dir}',config= config)

    train(data_dir, model_dir,save_dir, args)
