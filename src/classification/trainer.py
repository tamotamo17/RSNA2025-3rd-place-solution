import os
import random
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Any
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from .utils import RSNAMultiLabelAUC, RSNAPatchBinaryAUC, RSNAPatchMultiAUC, SimplePatchAUC


def _to_device_batch_for_loss(data, device):
    """
    loss 計算に必要なテンソルだけ GPU へ。
    """
    b = {}
    if "mask" in data:
        b["mask"] = data["mask"].to(device, non_blocking=True).float()
    if "slice_weights" in data and data["slice_weights"] is not None:
        b["slice_weights"] = data["slice_weights"].to(device, non_blocking=True).float()
    if "targets" in data:
        b["targets"] = data["targets"].to(device, non_blocking=True).float()
    if "targets_slice" in data:
        b["targets_slice"] = data["targets_slice"].to(device, non_blocking=True).float()
    if "target_aneurysm" in data:
        b["target_aneurysm"] = data["target_aneurysm"].to(device, non_blocking=True).float()
    if "target_site" in data:
        b["target_site"] = data["target_site"].to(device, non_blocking=True).float()
    if "mask_map" in data:
        b["mask_map"] = data["mask_map"].to(device, non_blocking=True).float()
    if "plane_encoded" in data:
        b["plane_encoded"] = data["plane_encoded"].to(device, non_blocking=True).long()
    if "modality_encoded" in data:
        b["modality_encoded"] = data["modality_encoded"].to(device, non_blocking=True).long()
    return b


def train_func(
    model,
    loader_train,
    optimizer,
    epoch,
    criterion,
    device="cuda",
    scaler=None,
    accum_steps=1,
    p_mixup=0.0,
    alpha_mixup=0.4,
    use_coords=False,
    mode='image'
):
    model.train()
    train_loss_hist = []
    bar = tqdm(loader_train)
    use_amp = scaler is not None

    optimizer.zero_grad(set_to_none=True)

    for it, data in enumerate(bar):
        if mode=='image':
            images = data["image"].to(device, non_blocking=True).contiguous().float()  # [B,S,H,W]
        elif mode=='feature':
            images = data["features"].to(device, non_blocking=True).contiguous().float()
        if use_coords:
            coords = data["grid_norm_center_zyx"].to(device, non_blocking=True).contiguous().float()
        B = images.size(0)

        # まず“元ラベル”をGPUへ
        batch_A = _to_device_batch_for_loss(data, device)

        # ---- mixup ----
        do_mix = (p_mixup > 0.0) and (random.random() < p_mixup)
        if do_mix:
            lam = np.random.beta(alpha_mixup, alpha_mixup)
            idx_gpu = torch.randperm(B, device=device)  # GPUで作る

            # 画像はGPU上でブレンド
            images = lam * images + (1.0 - lam) * images[idx_gpu]

            # ラベルもGPU上で「元ラベル」から作る（CPUの data[...] は触らない）
            batch_B = {k: v[idx_gpu] for k, v in batch_A.items()}
        else:
            lam = 1.0
            batch_B = None

        # ---- forward & loss ----
        with torch.amp.autocast("cuda", enabled=use_amp):
            if use_coords:
                outputs = model(images, coords)
            else:
                outputs = model(images)
            lossA = criterion(outputs, batch_A)["loss"]
            loss = lossA
            if do_mix:
                lossB = criterion(outputs, batch_B)["loss"]
                loss = lam * lossA + (1.0 - lam) * lossB
            loss = loss / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss_hist.append(loss.item() * accum_steps)
        bar.set_description(f"train_loss:{np.mean(train_loss_hist[-50:]):.4f}  epoch:{epoch}")

    return float(np.mean(train_loss_hist))


@torch.no_grad()
def valid_func(
    model,
    loader_valid,
    metric_meters,
    criterion: nn.Module,
    device="cuda",
    use_coords=False,
    mode="image"
):
    model.eval()
    loss_hist = []
    loss_target_hist = []
    loss_mask_hist = []
    
    bar = tqdm(loader_valid)
    for data in bar:
        if mode=='image':
            images = data["image"].to(device, non_blocking=True).contiguous().float()
        elif mode=='feature':
            images = data["features"].to(device, non_blocking=True).contiguous().float()
        batch = _to_device_batch_for_loss(data, device)
        if use_coords:
            coords = data["grid_norm_center_zyx"].to(device, non_blocking=True).contiguous().float()
            outputs = model(images, coords)
        else:
            outputs = model(images)
        losses = criterion(outputs, batch)

        loss_hist.append(float(losses["loss"]))
        loss_target_hist.append(float(losses["loss_cls"]))
        loss_mask_hist.append(float(losses["loss_seg"]))

        if metric_meters is not None:
            #metric_meters.update(batch["target_aneurysm"], outputs["output"], batch["mask"], outputs["mask"])  # logitsそのまま渡す
            metric_meters.update(batch, outputs)

        bar.set_description(f"valid_loss:{np.mean(loss_hist[-50:]):.4f}")

    result = {
        "valid_loss": float(np.mean(loss_hist)),
        "valid_loss_target": float(np.mean(loss_target_hist)),
        "valid_loss_mask": float(np.mean(loss_mask_hist)),
    }

    if metric_meters is not None:
        scores = metric_meters.avg
        if 'auc' in scores:
            result.update({"auc": scores["auc"]})
        if 'dice' in scores:
            result.update({"dice": scores["dice"]})
        if "aneurysm_auc" in scores:
            result.update({"aneurysm_auc": scores["aneurysm_auc"]})
        if "per_class_auc" in scores:
            result.update({"per_class_auc": scores["per_class_auc"]})
        if "final_score" in scores:
            result.update({"final_score": scores["final_score"]})
        
    return result

def train_roi_func(
    model,
    loader_train,
    optimizer,
    epoch,
    criterion,
    device="cuda",
    scaler=None,
    accum_steps=1,
    p_mixup=0.0,
    alpha_mixup=0.4,
    mode='image'
):
    model.train()
    train_loss_hist = []
    bar = tqdm(loader_train)
    use_amp = scaler is not None

    optimizer.zero_grad(set_to_none=True)

    for it, data in enumerate(bar):
        if mode=='image':
            images = data["image"].to(device, non_blocking=True).contiguous().float()  # [B,S,H,W]
        elif mode=='feature':
            images = data["features"].to(device, non_blocking=True).contiguous().float()
        B = images.size(0)

        # まず“元ラベル”をGPUへ
        batch_A = _to_device_batch_for_loss(data, device)

        # ---- mixup ----
        do_mix = (p_mixup > 0.0) and (random.random() < p_mixup)
        if do_mix:
            lam = np.random.beta(alpha_mixup, alpha_mixup)
            idx_gpu = torch.randperm(B, device=device)  # GPUで作る

            # 画像はGPU上でブレンド
            images = lam * images + (1.0 - lam) * images[idx_gpu]

            # ラベルもGPU上で「元ラベル」から作る（CPUの data[...] は触らない）
            batch_B = {k: v[idx_gpu] for k, v in batch_A.items()}
        else:
            lam = 1.0
            batch_B = None

        # ---- forward & loss ----
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(images)
            lossA = criterion(outputs, batch_A)["loss"]
            loss = lossA
            if do_mix:
                lossB = criterion(outputs, batch_B)["loss"]
                loss = lam * lossA + (1.0 - lam) * lossB
            loss = loss / accum_steps

        if use_amp:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (it + 1) % accum_steps == 0:
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        train_loss_hist.append(loss.item() * accum_steps)
        bar.set_description(f"train_loss:{np.mean(train_loss_hist[-50:]):.4f}  epoch:{epoch}")

    return float(np.mean(train_loss_hist))


@torch.no_grad()
def valid_roi_func(
    model,
    loader_valid,
    metric_meters,
    criterion: nn.Module,
    device="cuda",
    mode="image"
):
    model.eval()
    loss_hist = []
    loss_hist_patch = []
    loss_hist_mod = []
    loss_hist_plane = []
    bar = tqdm(loader_valid)
    for data in bar:
        if mode=='image':
            images = data["image"].to(device, non_blocking=True).contiguous().float()
        elif mode=='feature':
            images = data["features"].to(device, non_blocking=True).contiguous().float()
        batch = _to_device_batch_for_loss(data, device)
        outputs = model(images)
        losses = criterion(outputs, batch)

        loss_hist.append(float(losses["loss"]))
        if "loss_patch" in losses:
            loss_hist_patch.append(float(losses["loss_patch"]))
        if "loss_modality" in losses:
            loss_hist_mod.append(float(losses["loss_modality"]))
        if "loss_plane" in losses:
            loss_hist_plane.append(float(losses["loss_plane"]))

        if metric_meters is not None:
            #metric_meters.update(batch["target_aneurysm"], outputs["output"], batch["mask"], outputs["mask"])  # logitsそのまま渡す
            metric_meters.update(batch, outputs)

        bar.set_description(f"valid_loss:{np.mean(loss_hist[-50:]):.4f}")

    result = {
        "valid_loss": float(np.mean(loss_hist)),
    }
    if "loss_patch" in losses:
        result.update({"valid_loss_patch": float(np.mean(loss_hist_patch))})
    if "loss_modality" in losses:
        result.update({"valid_loss_modality": float(np.mean(loss_hist_mod))})
    if "loss_plane" in losses:
        result.update({"valid_loss_plane": float(np.mean(loss_hist_plane))})

    if metric_meters is not None:
        scores = metric_meters.avg
        if 'auc' in scores:
            result.update({"auc": scores["auc"]})
        if 'dice' in scores:
            result.update({"dice": scores["dice"]})
        if "aneurysm_auc" in scores:
            result.update({"aneurysm_auc": scores["aneurysm_auc"]})
        if "per_class_auc" in scores:
            result.update({"per_class_auc": scores["per_class_auc"]})
        if "final_score" in scores:
            result.update({"final_score": scores["final_score"]})
        if "by_topk" in scores:
            result.update({**scores})
        
    return result

@torch.no_grad()
def valid_roi_volume_func(
    model,
    loader_valid,
    metric_meters,
    criterion: nn.Module,
    device="cuda",
    mode="image"
):
    model.eval()
    loss_hist = []
    loss_patch_hist = []
    loss_volume_hist = []
    bar = tqdm(loader_valid)
    for data in bar:
        if mode=='image':
            images = data["image"].to(device, non_blocking=True).contiguous().float()
        elif mode=='feature':
            images = data["features"].to(device, non_blocking=True).contiguous().float()
        batch = _to_device_batch_for_loss(data, device)
        outputs = model(images)
        losses = criterion(outputs, batch)

        loss_hist.append(float(losses["loss"]))
        loss_patch_hist.append(float(losses["loss_patch"]))
        loss_volume_hist.append(float(losses["loss_volume"]))
        
        if metric_meters is not None:
            #metric_meters.update(batch["target_aneurysm"], outputs["output"], batch["mask"], outputs["mask"])  # logitsそのまま渡す
            metric_meters.update(batch={"target_site": batch["target_site"]},  # (B, C)
                                 outputs={"volume_output": outputs["volume_output"].detach().cpu()},
                                 )

        bar.set_description(f"valid_loss:{np.mean(loss_hist[-50:]):.4f}")

    result = {
        "valid_loss": float(np.mean(loss_hist)),
        "valid_loss_patch": float(np.mean(loss_patch_hist)),
        "valid_loss_volume": float(np.mean(loss_volume_hist)),
    }

    if metric_meters is not None:
        scores = metric_meters.avg  # MultiROIAUCVolume.avg の返り値

        # 主要スコア（final / aneurysm / sites mean）
        if "final_score" in scores:
            result["final_score"] = scores["final_score"]
        if "aneurysm_auc" in scores:
            result["aneurysm_auc"] = scores["aneurysm_auc"]
        if "site_mean_auc" in scores:
            result["site_mean_auc"] = scores["site_mean_auc"]

        # AUCのバリエーション
        if "auc_macro" in scores:
            result["auc_macro"] = scores["auc_macro"]
            # 後方互換：旧コードが result["auc"] を参照している場合に備えて
            result["auc"] = scores["auc_macro"]
        if "auc_micro" in scores:
            result["auc_micro"] = scores["auc_micro"]
        if "auc_weighted" in scores:
            result["auc_weighted"] = scores["auc_weighted"]

        # クラス別 / サイト別
        if "per_class_auc" in scores:
            result["per_class_auc"] = scores["per_class_auc"]
        if "site_aucs_including_aneurysm" in scores:
            result["site_aucs_including_aneurysm"] = scores["site_aucs_including_aneurysm"]
        if "site_aucs_excluding_aneurysm" in scores:
            result["site_aucs_excluding_aneurysm"] = scores["site_aucs_excluding_aneurysm"]

        # 参考情報（必要なら）
        for k in ("num_samples", "num_classes", "num_valid_classes",
                "mean_pred_overall", "pos_total", "neg_total"):
            if k in scores:
                result[k] = scores[k]
        
    return result

@torch.no_grad()
def extract_features_and_save(
    model,
    loader_valid,
    save_feature_root,
    save_prediction_root,
    device="cuda",
    ):
    model.eval()
    bar = tqdm(loader_valid)
    for data in bar:
        images = data["image"].to(device, non_blocking=True).contiguous().float()
        coords = data["grid_norm_center_zyx"].to(device, non_blocking=True).contiguous().float()
        sids = data["series_id"]
        outs = model(images, coords)
        feats = outs['feature'].cpu().detach().numpy()
        preds = outs['output'].cpu().detach().numpy()
        for feat, pred, sid in zip(feats, preds, sids):
            filename_feat = save_feature_root / (sid + '.npy')
            filename_pred = save_prediction_root / (sid + '.npy')
            np.save(filename_feat, feat)
            np.save(filename_pred, pred)

 


def rand_bbox(size, lam):
    """Generate random bounding box for CutMix"""
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int32(W * cut_rat)
    cut_h = np.int32(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


class ModelCheckpoint:
    """Model checkpoint handler"""
    
    def __init__(
        self,
        save_dir: str,
        monitor: str = 'val_loss',
        mode: str = 'min',
        save_top_k: int = 3,
        verbose: bool = True,
    ):
        self.save_dir = save_dir
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = save_top_k
        self.verbose = verbose
        
        self.best_scores = []
        self.best_paths = []
        
        os.makedirs(save_dir, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        filename: Optional[str] = None,
    ):
        """Save model checkpoint"""
        if filename is None:
            filename = f"epoch_{epoch:03d}_{self.monitor}_{metrics[self.monitor]:.4f}.pth"
        
        filepath = os.path.join(self.save_dir, filename)
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
        }
        torch.save(checkpoint, filepath)
        
        # Update best scores
        score = metrics[self.monitor]
        if self.mode == 'min':
            score = -score
        
        if len(self.best_scores) < self.save_top_k:
            self.best_scores.append(score)
            self.best_paths.append(filepath)
        elif score > min(self.best_scores):
            # Remove worst checkpoint
            idx = self.best_scores.index(min(self.best_scores))
            old_path = self.best_paths[idx]
            if os.path.exists(old_path):
                os.remove(old_path)
            
            # Add new checkpoint
            self.best_scores[idx] = score
            self.best_paths[idx] = filepath
        else:
            # Not in top-k, remove the file
            if os.path.exists(filepath):
                os.remove(filepath)
            return
        
        if self.verbose:
            print(f"Saved checkpoint: {filepath} ({self.monitor}={metrics[self.monitor]:.4f})")
    
    def load_best(self, model: nn.Module, optimizer: Optional[torch.optim.Optimizer] = None):
        """Load best checkpoint"""
        if not self.best_paths:
            return None
        
        # Get best checkpoint
        idx = self.best_scores.index(max(self.best_scores))
        best_path = self.best_paths[idx]
        
        # Load checkpoint
        checkpoint = torch.load(best_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.verbose:
            print(f"Loaded best checkpoint: {best_path}")
        
        return checkpoint
