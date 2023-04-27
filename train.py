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
from dataset import MaskBaseDataset
from loss import create_criterion
import augmentation
import model as Models
# import wandb
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
    train_dataset = MaskBaseDataset(
        data_dir=data_dir,
        segmentation = args.seg
    )

    val_dataset = MaskBaseDataset(
        data_dir=data_dir,
        segmentation= args.seg
    )
    
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
    
    model = Models.SingleOutputModel(model=model_module).to(device)
    model = torch.nn.DataParallel(model)

    # -- loss & metric
    criterion = create_criterion(args.criterion)  # default: cross_entropy
    criterion_mse = create_criterion('mse')       # age regression loss

    # set alpha value to penalize age MSE loss
    alpha = args.use_age
    
    # opt_module = getattr(import_module("torch.optim"), args.optimizer)  # default: SGD
    optimizer = opt_module(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=5e-4
    )
    scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

    best_val_f1 = 0
    best_val_loss = np.inf
    for epoch in range(args.epochs):
        
        category_preds = []
        category_trues = []
        
        train_loss_all = []
        train_loss0 =[]
        train_loss4 =[]
        
        # train loop
        model.train()
        loss_value = 0
        matches = 0
        for idx, train_batch in tqdm(enumerate(train_loader)):
            inputs, labels, age_num_labels = train_batch
            
            inputs = inputs['image'].to(device)
            labels = labels.to(device)
            age_num_labels = age_num_labels.to(device)

            optimizer.zero_grad()

            outs, age_num_outs = model(inputs)
            preds = torch.argmax(outs, dim=-1)

            loss0 = criterion(outs, labels)
            loss4 = criterion_mse(age_num_outs, age_num_labels.unsqueeze(1).float()) * alpha
            
            loss = loss0 + loss4

            loss.backward()
            optimizer.step()
            
            category_preds += outs.argmax(1).detach().cpu().numpy().tolist()
            category_trues += labels.detach().cpu().numpy().tolist()
            
            loss_value += loss.item()
            matches += (preds == labels).sum().item()
            
            train_loss_all.append(loss.item())
            train_loss0.append(loss0.item())
            train_loss4.append(loss4.item())

            if (idx + 1) % args.log_interval == 0:
                train_loss = loss_value / args.log_interval
                train_acc = matches / args.batch_size / args.log_interval
                current_lr = get_lr(optimizer)
                print(
                    f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                    f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                )
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
            val_loss0 =[]
            val_loss4 =[]
            
            for val_batch in val_loader:
                inputs, labels, age_num_labels = val_batch
                
                inputs = inputs['image'].to(device)
                labels = labels.to(device)
                age_num_labels = age_num_labels.to(device)
                
                outs, age_num_outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)

                loss0 = criterion(outs, labels)
                loss4 = criterion_mse(age_num_outs, age_num_labels.unsqueeze(1).float()) * alpha

                loss = loss0 + loss4

                loss_item = loss.item()
                
                val_loss0.append(loss0.item())
                val_loss4.append(loss4.item())
                
                acc_item = (labels == preds).sum().item()
                val_loss_items.append(loss_item)
                val_acc_items.append(acc_item/val_loader.batch_size)

                val_category_preds += outs.argmax(1).detach().cpu().numpy().tolist()
                val_category_trues += labels.detach().cpu().numpy().tolist()
            
            val_loss = np.sum(val_loss_items) / len(val_loader)
            val_acc = np.mean(val_acc_items)
            best_val_loss = min(best_val_loss, val_loss)
                                   
            _train_f1 = f1_score(category_trues, category_preds, average='macro')
            _train_accuracy = accuracy_score(category_trues, category_preds)

            _train_loss = np.mean(train_loss_all)
            _train_loss0 = np.mean(train_loss0)
            _train_loss4 = np.mean(train_loss4)
            _val_loss0 = np.mean(val_loss0)
            _val_loss4 = np.mean(val_loss4)
            
            _val_f1 = f1_score(val_category_trues,val_category_preds, average='macro')

            # wandb.log({'Train loss':_train_loss,'Train F1':_train_f1,'Train Acc':_train_accuracy,'Train loss 0' : _train_loss0,'Train loss 4':_train_loss4,
            #        'Val loss':val_loss,'Val F1':_val_f1, 'Val Acc':val_acc,'Val loss 0' : _val_loss0,'Val loss 4':_val_loss4,'Val best f1':max(_val_f1, best_val_f1)})      ## logging wandb
            
            if _val_f1 > best_val_f1:
                print(f"New best model for val f1 : {_val_f1:4.2%}! saving the best model..")
                torch.save(model.module.state_dict(), f"{save_dir}/best.pth")
                
                best_val_f1 = _val_f1
            torch.save(model.module.state_dict(), f"{save_dir}/last.pth")
            print(
                f"[Val] f1 : {_val_f1:4.2%}, loss: {val_loss:4.2} || "
                f"best f1 : {best_val_f1:4.2%}, best loss: {best_val_loss:4.2}"
            )
            print()


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 42)')
    parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train (default: 50)')
    # parser.add_argument('--dataset', type=str, default='MaskBaseDataset', help='dataset augmentation type (default: MaskBaseDataset)')
    # parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    # parser.add_argument("--resize", nargs="+", type=list, default=[128, 96], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=64, help='input batch size for validing (default: 64)')
    parser.add_argument('--model', type=str, default='EfficientBase', help='model type (default: EfficientBase)')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    # parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument('--name', default='exp', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--fold', default=1, help = 'kfold')
    parser.add_argument('--use_age', type=float, default=0, help='weight of mseloss(age) (default: 0)')
    parser.add_argument('--seg', type=bool, default=False, help='enable segmentation (default: False)')
    # parser.add_argument('--mislabel', type=bool, default=False, help='train with corrected label (default: False)')


    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir
    save_dir = increment_path(os.path.join(model_dir, args.name))

    os.makedirs(save_dir, exist_ok=True)
    
    # config = {'save_dir' : save_dir,
    #           'use_age': args.use_age,
    #           'multi':False, 
    #           'seg':args.seg,
    #           'mislabel': args.mislabel,
    #           'model':args.model,
    #           'augmentation':args.augmentation,
    #           'batch_size':args.batch_size,
    #           'criterion':args.criterion,
    #           'epoch':args.epochs,
    #           'fold':args.fold,
    #           'lr':args.lr,
    #           'lr_decay_step':args.lr_decay_step,
    #           'optimizer':args.optimizer,
    #           'resize':str(args.resize[0])+'X'+str(args.resize[1]),
    #           'seed':args.seed}
    
    
    # dir = save_dir.split('/')[-1]
    # wandb.init(project='image-classification-challenge-v2',name=f'use_age:{args.use_age}-multi:{False}-seg:{args.seg}-mislabel:{args.mislabel}-model:{args.model}-exp:{dir}',config= config)
    
    train(data_dir, model_dir, save_dir, args)

