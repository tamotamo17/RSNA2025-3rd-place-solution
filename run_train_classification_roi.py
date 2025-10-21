#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Classification training script for patch-based learning
Based on run_train.py but adapted for classification modules

Features:
- Uses RSNAPatchDataset parameters from config
- Integrated data augmentation with profile-based selection
- All dataset parameters (patch_size_mm, out_size_zyx, etc.) configurable via YAML
- Automatic transform generation based on augmentation profile

Usage:
    # Default training with config
    poetry run python run_train_classification_patch.py
    
    # With specific config file (YAML)
    poetry run python run_train_classification_patch.py --config path/to/config.yaml
    
    # Debug mode with light augmentation
    poetry run python run_train_classification_patch.py --debug
"""

import os
import sys
import gc
import ast
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append('src')

from classification.config.config_roi import Config
from classification.datasets import RSNAPatchDataset, RSNAPatchDatasetV2, RSNAPatchDatasetV3, RSNAROIDatasetV1
from classification.models import create_model
from classification.losses import create_loss
from classification.utils import setup_logger, set_random_seed, RSNAMultiLabelAUC, RSNAPatchBinaryAUC, SimplePatchAUC, MultiPatchAUC, MultiROIAUC
from classification.trainer import train_roi_func, valid_roi_func, ModelCheckpoint

# Device will be set based on gpu argument in main()
device = None
torch.backends.cudnn.benchmark = True


def _make_optimizer(model, config: Config):
    """Create optimizer"""
    opt_cfg = config.training.optimizer
    
    if opt_cfg.name == 'AdamW':
        return optim.AdamW(
            model.parameters(),
            lr=opt_cfg.lr,
            weight_decay=opt_cfg.weight_decay,
            betas=opt_cfg.betas
        )
    elif opt_cfg.name == 'SGD':
        return optim.SGD(
            model.parameters(),
            lr=opt_cfg.lr,
            momentum=opt_cfg.momentum,
            weight_decay=opt_cfg.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_cfg.name}")


def _make_scheduler(optimizer, config: Config):
    """Create learning rate scheduler"""
    if config.training.scheduler is None:
        return None
    
    sch_cfg = config.training.scheduler
    
    if sch_cfg.name == 'CosineAnnealingLR':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=sch_cfg.T_max,
            eta_min=sch_cfg.eta_min
        )
    elif sch_cfg.name == 'CosineAnnealingWarmRestarts':
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config.training.epochs,
            eta_min=sch_cfg.eta_min
        )
    elif sch_cfg.name == 'ReduceLROnPlateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=sch_cfg.mode,
            factor=sch_cfg.factor,
            patience=sch_cfg.patience,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {sch_cfg.name}")


def _make_loader(dataset, config: Config, shuffle: bool, drop_last: bool = False):
    """Create data loader"""
    batch_size = config.training.batch_size if shuffle else config.training.val_batch_size
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=drop_last,
        prefetch_factor=4 if config.training.num_workers > 0 else 2,
        pin_memory_device="cuda",
        persistent_workers=True if config.training.num_workers > 0 else False,
    )


def _unwrap_state_dict_for_saving(model: nn.Module) -> dict[str, Any]:
    """Extract state_dict from DataParallel wrapper"""
    if isinstance(model, nn.DataParallel):
        return model.module.state_dict()
    return model.state_dict()


def _log_patch_metrics(logger, metrics: dict, prefix: str = ""):
    """Log patch-based metrics"""
    if not metrics:
        return
    
    logger.info(f"{prefix}Patch Metrics:")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")


def run_training(df, df_loc, fold: int, config: Config):
    """Main training function for patch-based classification"""
    
    # Setup logging
    log_dir = Path(config.training.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"fold{fold}.log"
    logger = setup_logger(name=f"fold{fold}", log_file=str(log_file))
    
    # Model paths
    model_dir = Path(config.training.output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_best = model_dir / f"best_fold{fold}.pth"
    model_last = model_dir / f"last_fold{fold}.pth"
    
    # ===== Data Split =====
    train_df = df[df["fold"] != fold].reset_index(drop=True)
    valid_df = df[df["fold"] == fold].reset_index(drop=True)
    # train_df = train_df[:100]
    # valid_df = valid_df[:100]
    logger.info(f"Fold {fold} | Train: {len(train_df)} | Valid: {len(valid_df)}")
    # ===== Data Augmentation =====
    # Get transforms from config
    train_transforms = config.augmentation.transforms_train
    valid_transforms = config.augmentation.transforms_valid
    
    # ===== Datasets & Loaders =====
    
    train_dataset = RSNAROIDatasetV1(
        df=train_df,
        df_loc=df_loc,
        mode='train',
        volume_size_mm=tuple(config.data.volume_size_mm),
        out_size_zyx=tuple(config.data.out_size_zyx),          # (Z_out, H_out, W_out)
        map_size_zyx=config.data.map_size_zyx,
        r=config.data.r,
        r_unit=config.data.r_unit,                      # "mm" or "px"
        p_flip=config.data.p_flip,
        align_plane=config.data.align_plane,
        cta_min=config.data.cta_min,
        cta_max=config.data.cta_max,
        transform=train_transforms,
    )
    print('Train transforms: ', train_transforms)
    valid_dataset = RSNAROIDatasetV1(
        df=valid_df,
        df_loc=df_loc,
        mode='valid',
        volume_size_mm=tuple(config.data.volume_size_mm),
        out_size_zyx=tuple(config.data.out_size_zyx),
        map_size_zyx=config.data.map_size_zyx,
        r=config.data.r,
        r_unit=config.data.r_unit,
        p_flip=config.data.p_flip,
        align_plane=config.data.align_plane,
        cta_min=config.data.cta_min,
        cta_max=config.data.cta_max,
        transform=valid_transforms,
    )
    print('Valid transforms: ', valid_transforms)
    train_loader = _make_loader(train_dataset, config, shuffle=True, drop_last=True)
    valid_loader = _make_loader(valid_dataset, config, shuffle=False, drop_last=False)
    
    logger.info(f"Train batches: {len(train_loader)} | Valid batches: {len(valid_loader)}")
    
    # Debug: Check data distribution
    logger.info("Dataset distribution:")
    if 'Aneurysm Present' in train_df.columns:
        train_pos = train_df['Aneurysm Present'].sum()
        train_total = len(train_df)
        logger.info(f"  Train: {train_pos}/{train_total} positive ({train_pos/train_total*100:.1f}%)")
        
        valid_pos = valid_df['Aneurysm Present'].sum()
        valid_total = len(valid_df)
        logger.info(f"  Valid: {valid_pos}/{valid_total} positive ({valid_pos/valid_total*100:.1f}%)")
        
        # Check unique values in the column
        train_unique = sorted(train_df['Aneurysm Present'].unique())
        valid_unique = sorted(valid_df['Aneurysm Present'].unique())
        logger.info(f"  Train unique values: {train_unique}")
        logger.info(f"  Valid unique values: {valid_unique}")
    
    # ===== Model =====
    model = create_model(**config.model.__dict__)
    model = model.to(device)
    print(model)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Data Parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    
    # ===== Optimizer, Scheduler, Loss =====
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    
    # Debug: Check loss configuration
    logger.info(f"Loss config name: {config.loss.name}")
    logger.info(f"Loss config dict: {config.loss.__dict__}")
    
    criterion = create_loss(**config.loss.__dict__)
    
    logger.info(f"Optimizer: {config.training.optimizer.name}")
    logger.info(f"Scheduler: {config.training.scheduler.name if config.training.scheduler else 'None'}")
    logger.info(f"Loss function created: {type(criterion).__name__}")
    
    # ===== Metrics =====
    if config.data.num_classes == 1:
        # Binary classification - use simple patch-level AUC
        metrics = SimplePatchAUC(compute_on_step=False, dist_sync_on_step=False)
    else:
        # Multi-label classification
        metrics = MultiROIAUC(aggregate_op="topk_mean",     # "mean" や "max" も可
                              topk_list=[2, 4, 8, 16, 32, 64],  # 複数Kを一括評価
                              topk_basis="anchor",
                              anchor_index=13,
                              # topk_ratio_list=[0.01, 0.02],   # 比率で指定する場合はこちら（topk_listと併用可）
                              use_sigmoid=True,
                              )
        
    logger.info(f"Metric function created: {type(metrics).__name__}")
    
    # ===== Training Loop =====
    best_metric = -np.inf
    early_stopping_counter = 0
    
    logger.info(f"Start training | Epochs: {config.training.epochs}")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, config.training.epochs + 1):
        # Update learning rate
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{config.training.epochs} | LR: {current_lr:.7f}")
        
        # ===== Training =====
        train_loss = train_roi_func(
            model,
            train_loader,
            optimizer,
            epoch,
            criterion,
            device="cuda",
            scaler=scaler,
            accum_steps=config.training.gradient_accumulation_steps,
            p_mixup=config.training.p_mixup,
            alpha_mixup=config.training.alpha_mixup,
            )
        
        # ===== Validation =====
        if epoch % config.training.validate_every_n_epochs == 0:
            valid_metrics = valid_roi_func(
                model=model,
                loader_valid=valid_loader,
                metric_meters=metrics,
                criterion=criterion,
                device=device,
                )
            
            # Update scheduler if ReduceLROnPlateau
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(valid_metrics['val_loss'])
            
            # Log metrics
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.5f} | "
            )
            
            if 'valid_loss' in valid_metrics:
                logger.info(
                    f"  Valid loss: {valid_metrics['valid_loss']:.5f}, "
                    )
            if 'valid_loss_patch' in valid_metrics:
                logger.info(
                    f"  Valid loss patch: {valid_metrics['valid_loss_patch']:.5f}, "
                    )
            if 'valid_loss_plane' in valid_metrics:
                logger.info(
                    f"  Valid loss plane: {valid_metrics['valid_loss_plane']:.5f}, "
                    )
            if 'valid_loss_modality' in valid_metrics:
                logger.info(
                    f"  Valid loss modality: {valid_metrics['valid_loss_modality']:.5f}, "
                    )
            
            if "by_topk" in valid_metrics:
                print(valid_metrics["by_topk"].keys())
                rep_k  = valid_metrics["representative_topk"]
                best_k = valid_metrics["best_topk"]

                logger.info(f"[REP] K={rep_k} Final={valid_metrics['rep_final_score']:.5f}  "
                            f"Aneu={valid_metrics['rep_aneurysm_auc']:.5f}  SitesMean={valid_metrics['rep_site_mean_auc']:.5f}")

                logger.info(f"[BEST] K={best_k} Final={valid_metrics['best_final_score']:.5f}  "
                            f"Aneu={valid_metrics['best_aneurysm_auc']:.5f}  SitesMean={valid_metrics['best_site_mean_auc']:.5f}")

                # ベストKのsiteごとのAUC
                for name, auc in valid_metrics["best_site_aucs_excluding_aneurysm"].items():
                    logger.info(f"[BEST K={best_k}] {name}: {auc}")

                # ついでに全Kの一覧
                for k, d in sorted(valid_metrics["by_topk"].items()):
                    logger.info(f"[K={k}] Final={d['final_score']:.5f}")
            else:
                # 単一K（mean/max含む）
                logger.info(f"[SINGLE] Final={valid_metrics['final_score']:.5f}")
            if 'best_final_score' in valid_metrics:
                current_metric = valid_metrics['best_final_score']
                is_best = current_metric > best_metric
            else:
                # Use negative loss for minimization (convert to maximization problem)
                current_metric = -valid_metrics['valid_loss']
                is_best = current_metric > best_metric
            
            if is_best:
                best_metric = current_metric
                early_stopping_counter = 0
                
                if 'best_final_score' in valid_metrics:
                    logger.info(f"New best model saved! AUC: {current_metric:.5f}")
                else:
                    logger.info(f"New best model saved! Loss: {valid_metrics['valid_loss']:.5f} (metric: {current_metric:.5f})")
            else:
                early_stopping_counter += 1
            
            # Early stopping
            if early_stopping_counter >= config.training.early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break
        
        else:
            # Training only epochs
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.5f}")
        
        # Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': _unwrap_state_dict_for_saving(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_metric': best_metric,
                'config': config.__dict__
            }
            
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_data, model_last)
            logger.info(f"Checkpoint saved to {model_last}")
    
    logger.info(f"Training completed! Best AUC: {best_metric:.5f}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_metric

def run_training_full(df, df_loc, config: Config):
    """Main training function for patch-based classification"""
    
    # Setup logging
    log_dir = Path(config.training.output_dir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"full.log"
    logger = setup_logger(name=f"full", log_file=str(log_file))
    
    # Model paths
    model_dir = Path(config.training.output_dir) / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_last = model_dir / f"last_full.pth"
    
    # ===== Data Split =====
    train_df = df
    logger.info(f"Train: {len(train_df)}")
    # ===== Data Augmentation =====
    # Get transforms from config
    train_transforms = config.augmentation.transforms_train
    
    # ===== Datasets & Loaders =====
    
    train_dataset = RSNAROIDatasetV1(
        df=train_df,
        df_loc=df_loc,
        mode='train',
        volume_size_mm=tuple(config.data.volume_size_mm),
        out_size_zyx=tuple(config.data.out_size_zyx),          # (Z_out, H_out, W_out)
        map_size_zyx=config.data.map_size_zyx,
        r=config.data.r,
        r_unit=config.data.r_unit,                      # "mm" or "px"
        p_flip=config.data.p_flip,
        transform=train_transforms,
    )
    print('Train transforms: ', train_transforms)
    train_loader = _make_loader(train_dataset, config, shuffle=True, drop_last=True)
    
    logger.info(f"Train batches: {len(train_loader)}")
    
    # Debug: Check data distribution
    logger.info("Dataset distribution:")
    if 'Aneurysm Present' in train_df.columns:
        train_pos = train_df['Aneurysm Present'].sum()
        train_total = len(train_df)
        logger.info(f"  Train: {train_pos}/{train_total} positive ({train_pos/train_total*100:.1f}%)")
        
        # Check unique values in the column
        train_unique = sorted(train_df['Aneurysm Present'].unique())
        logger.info(f"  Train unique values: {train_unique}")
        
    # ===== Model =====
    model = create_model(**config.model.__dict__)
    model = model.to(device)
    print(model)
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Data Parallel
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs")
    
    # ===== Optimizer, Scheduler, Loss =====
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    
    # Debug: Check loss configuration
    logger.info(f"Loss config name: {config.loss.name}")
    logger.info(f"Loss config dict: {config.loss.__dict__}")
    
    criterion = create_loss(**config.loss.__dict__)
    
    logger.info(f"Optimizer: {config.training.optimizer.name}")
    logger.info(f"Scheduler: {config.training.scheduler.name if config.training.scheduler else 'None'}")
    logger.info(f"Loss function created: {type(criterion).__name__}")
    
    # ===== Metrics =====
    if config.data.num_classes == 1:
        # Binary classification - use simple patch-level AUC
        metrics = BinaryROIAUC(aggregate_op="topk_mean",     # "mean" や "max" も可
                              topk_list=[2, 3, 4, 6, 8, 16],  # 複数Kを一括評価
                              use_sigmoid=True,
                              )
    else:
        # Multi-label classification
        metrics = MultiROIAUC(aggregate_op="topk_mean",     # "mean" や "max" も可
                              topk_list=[2, 3, 4, 6, 8, 16],  # 複数Kを一括評価
                              use_sigmoid=True,
                              )
        
    logger.info(f"Metric function created: {type(metrics).__name__}")
    
    # ===== Training Loop =====
    best_metric = -np.inf
    early_stopping_counter = 0
    
    logger.info(f"Start training | Epochs: {config.training.epochs}")
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(1, config.training.epochs + 1):
        # Update learning rate
        if scheduler and not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Epoch {epoch}/{config.training.epochs} | LR: {current_lr:.7f}")
        
        # ===== Training =====
        train_loss = train_roi_func(
            model,
            train_loader,
            optimizer,
            epoch,
            criterion,
            device="cuda",
            scaler=scaler,
            accum_steps=config.training.gradient_accumulation_steps,
            p_mixup=config.training.p_mixup,
            alpha_mixup=config.training.alpha_mixup,
            )
        
        # ===== Validation =====
        if epoch % config.training.validate_every_n_epochs == 0:
            
            
            # Log metrics
            logger.info(
                f"Epoch {epoch} | Train Loss: {train_loss:.5f} | "
                )           
        else:
            # Training only epochs
            logger.info(f"Epoch {epoch} | Train Loss: {train_loss:.5f}")
        
        # Save checkpoint
        if epoch % config.training.save_every_n_epochs == 0:
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': _unwrap_state_dict_for_saving(model),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config.__dict__
            }
            
            if scheduler:
                checkpoint_data['scheduler_state_dict'] = scheduler.state_dict()
            
            torch.save(checkpoint_data, model_last)
            logger.info(f"Checkpoint saved to {model_last}")
    
    # Cleanup
    del model
    torch.cuda.empty_cache()
    gc.collect()
    
    return

def load_config_from_path(path: Optional[str]) -> Config:
    """--config で渡されたパスから Config を読み込む。
    - .py: importlib で動的ロードし、Config（クラス or インスタンス or 生成関数）を取得
    - .yaml/.yml: 既存の Config.load() を利用
    - None: 既定の Config()
    """
    if path is None:
        return Config()

    p = Path(path)
    suffix = p.suffix.lower()

    if suffix in [".yaml", ".yml"]:
        # YAML は既存のロード関数がそのまま使える
        return Config.load(str(p))

    if suffix == ".py":
        # Python モジュールとしてロード
        import importlib.util
        spec = importlib.util.spec_from_file_location("user_config", str(p))
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load python config: {p}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # 実行してモジュール化

        # 取りうるパターンを順に拾う
        # 1) Config が「クラス」で定義されている
        if hasattr(mod, "Config") and isinstance(getattr(mod, "Config"), type):
            return mod.Config()  # デフォルト引数で生成（必要に応じて __post_init__ が走る）

        # 2) Config が「インスタンス」として置かれている
        if hasattr(mod, "Config"):
            cfg_obj = getattr(mod, "Config")
            if isinstance(cfg_obj, Config):
                return cfg_obj

        # 3) get_config() が Config を返す関数として定義されている
        if hasattr(mod, "get_config") and callable(getattr(mod, "get_config")):
            cfg_obj = mod.get_config()
            if isinstance(cfg_obj, Config):
                return cfg_obj

        # 4) config という変数名で置かれている
        if hasattr(mod, "config"):
            cfg_obj = getattr(mod, "config")
            if isinstance(cfg_obj, Config):
                return cfg_obj

        raise ValueError(
            "Python config file must expose one of:\n"
            "- class Config (dataclass) that can be instantiated with no args,\n"
            "- Config (instance),\n"
            "- get_config() -> Config,\n"
            "- config (Config instance)."
        )

    raise ValueError(f"Unsupported config extension: {suffix}. Use .py, .yaml, or .yml")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Classification Patch Model')
    parser.add_argument('--config', type=str, default=None, help='Config name')
    parser.add_argument('--fold', type=int, default=0, help='Fold number')
    parser.add_argument('--gpu', type=str, default='0', help='GPU number(s) to use (e.g., 0 or 0,1,2)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--full', action='store_true', help='Full data training')
    args = parser.parse_args()
    
    # Set GPU device
    global device
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"Using GPU(s): {args.gpu} ({gpu_count} device(s) available)")
        for i in range(gpu_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("No GPU available, using CPU")
    
    # Load configuration
    config = load_config_from_path(args.config)
        
    # Set random seed
    set_random_seed(config.seed)
    
    mod2idx = {'CTA': 0,
               'MRA': 1,
               'MRI T2': 2,
               'MRI T1post': 3
               }
    plane2idx = {'AXIAL': 0,
                 'SAGITTAL': 1,
                 'CORONAL': 2
                 }
    # Load data
    print(config)
    df = pd.read_csv(config.data.train_csv)
    df_loc = pd.read_csv(config.data.train_loc_csv)  # Adjust path as needed
    # Debug mode override
    if args.debug:
        config.training.epochs = 1
        df = df.iloc[:100]
    
    df['ModalityEncoded'] = df['Modality'].map(mod2idx)
    df['OrienationLabelEncoded'] = df['OrientationLabel'].map(plane2idx)  
    # Process dataframe
    if 'sorted_files' in df.columns:
        df["sorted_files"] = df["sorted_files"].apply(
            lambda x: ast.literal_eval(x) if pd.notna(x) else x
        )
    
    # Remove problematic series
    series_to_remove = [
        "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306",
        "1.2.826.0.1.3680043.8.498.82768897201281605198635077495114055892",
        "1.2.826.0.1.3680043.8.498.10063454172499468887877935052136698373",
        "1.2.826.0.1.3680043.8.498.22157965342587174310173115980837533982",
    ]
    df = df[~df["SeriesInstanceUID"].isin(series_to_remove)].reset_index(drop=True)
    df['file_name'] = df['SeriesInstanceUID'].apply(lambda x: Path(config.data.data_root)/x/(x+'.npy'))
    # Filter valid data
    df = df[~df["z_spacing"].isna()].reset_index(drop=True)
    if config.data.modality is not None:
        df = df[df['Modality']==config.data.modality].reset_index(drop=True)
    if config.data.plane is not None:
        df = df[df['OrientationLabel']==config.data.plane].reset_index(drop=True)
    
    if config.debug:
        df = df.head(100)
        df_loc = df_loc.head(100)
    
    print(f"Training with {len(df)} samples")
    print(f"Configuration: {config.experiment_name}")
    print(f"Output directory: {config.training.output_dir}")
    
    # Run training
    if not args.full:
        best_score = run_training(df, df_loc, args.fold, config)
        if best_score > 0:
            print(f"Training completed with best AUC: {best_score:.5f}")
        else:
            print(f"Training completed with best metric: {best_score:.5f} (negative loss)")
    else:
        run_training_full(df, df_loc, config)


if __name__ == "__main__":
    main()