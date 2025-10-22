"""
Model definition and training loop for Spacing Predictor
"""

import time
import torch
import torch.nn as nn
import timm
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    CosineAnnealingWarmRestarts,
    CosineAnnealingLR,
    ReduceLROnPlateau
)
from torch.utils.data import DataLoader

from .config import CFG
from .dataset import TrainDataset, get_transforms
from .trainer import train_fn, valid_fn
from .utils import get_score


def get_scheduler(optimizer):
    if CFG.scheduler == 'ReduceLROnPlateau':
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=CFG.factor,
            patience=CFG.patience, verbose=True, eps=CFG.eps
        )
    elif CFG.scheduler == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer, T_max=CFG.T_max, eta_min=CFG.min_lr, last_epoch=-1
        )
    elif CFG.scheduler == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1
        )
    return scheduler


def train_loop(folds, fold, output_dir, logger):
    logger.info(f"========== fold: {fold} training ==========")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ====================================================
    # loader
    # ====================================================
    trn_idx = folds[folds['fold'] != fold].index
    val_idx = folds[folds['fold'] == fold].index

    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)

    train_dataset = TrainDataset(
        train_folds,
        transform=get_transforms(data='train')
    )
    valid_dataset = TrainDataset(
        valid_folds,
        transform=get_transforms(data='valid')
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # ====================================================
    # model & optimizer
    # ====================================================
    model = timm.create_model(
        model_name=CFG.model_name,
        num_classes=CFG.target_size,
        pretrained=True
    )
    model.to(device)

    optimizer = Adam(
        model.parameters(),
        lr=CFG.lr,
        weight_decay=CFG.weight_decay,
        amsgrad=False
    )
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # apex (if needed)
    # ====================================================
    if CFG.apex:
        from apex import amp
        model, optimizer = amp.initialize(model, optimizer, opt_level='O1', verbosity=0)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.L1Loss()

    best_score = float('inf')  # MAE is lower is better
    best_loss = float('inf')

    for epoch in range(CFG.epochs):
        start_time = time.time()

        # train
        avg_loss = train_fn(train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds = valid_fn(valid_loader, model, criterion, device)
        valid_labels = valid_folds[CFG.target_cols].values

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(valid_labels, preds)

        elapsed = time.time() - start_time

        logger.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        logger.info(f'Epoch {epoch+1} - Score: {score}')

        if score < best_score:
            best_score = score
            logger.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(
                {'model': model.state_dict(), 'preds': preds},
                output_dir + f'{CFG.model_name}_fold{fold}_best.pth'
            )

        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            logger.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save(
                {'model': model.state_dict(), 'preds': preds},
                output_dir + f'{CFG.model_name}_fold{fold}_best_loss.pth'
            )

    check_point = torch.load(
        output_dir + f'{CFG.model_name}_fold{fold}_best_loss.pth',
        weights_only=False
    )
    valid_folds[[str(c) for c in range(CFG.target_size)]] = check_point['preds']

    return valid_folds