from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, Union, Sequence, Optional


import torch
import torch.nn as nn
import torch.nn.functional as F


_LOSS_REGISTRY: dict[str, Callable[..., nn.Module]] = {}


def register_loss(name: str) -> Callable[[Callable[..., nn.Module]], Callable[..., nn.Module]]:
    """
    使い方:
        @register_loss("multitask_logits01")
        def _make_loss(**kwargs) -> nn.Module:
            return MultiTaskLossLogits01(**kwargs)
    """

    def _decorator(factory: Callable[..., nn.Module]) -> Callable[..., nn.Module]:
        key = name.strip().lower()
        if key in _LOSS_REGISTRY:
            raise ValueError(f"loss '{name}' is already registered")
        _LOSS_REGISTRY[key] = factory
        return factory

    return _decorator


def _filter_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """factory のシグネチャにある引数だけを通す（余分は無視）"""
    sig = inspect.signature(factory)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def create_loss(name: str, **kwargs: Any) -> nn.Module:
    """
    例:
      criterion = create_loss(
          "multitask_logits01",
          lambda_case=1.0, lambda_slice=1.0, lambda_mask=2.0,
          alpha_mask=0.5, tversky_alpha=0.3, tversky_beta=0.7,
          pos_weight_case=None, pos_weight_mask=None, pos_weight_slice=None,
          slice_weights_positives_only=True,
      )
    """
    key = name.strip().lower()
    if key not in _LOSS_REGISTRY:
        # フォールバック: 同モジュールの globals() に同名があればそれを使う
        g = globals()
        if name in g and callable(g[name]):
            factory = g[name]  # type: ignore[index]
        else:
            known = ", ".join(sorted(_LOSS_REGISTRY.keys()))
            raise KeyError(f"Unknown loss '{name}'. Known: [{known}]")
    else:
        factory = _LOSS_REGISTRY[key]

    return factory(**_filter_kwargs(factory, kwargs))


def list_losses() -> list[str]:
    """登録済み Loss 名一覧"""
    return sorted(_LOSS_REGISTRY.keys())


@register_loss("multitask_logits01")
def _make_multitask_logits01(
    lambda_case: float = 1.0,
    lambda_slice: float = 1.0,
    lambda_mask: float = 2.0,
    alpha_mask: float = 0.5,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    pos_weight_case: float | torch.Tensor | None = None,
    pos_weight_mask: float | None = None,
    pos_weight_slice: float | None = None,
    slice_weights_positives_only: bool = True,
    eps: float = 1e-6,
) -> nn.Module:
    return MultiTaskLossLogits01(
        lambda_case=lambda_case,
        lambda_slice=lambda_slice,
        lambda_mask=lambda_mask,
        alpha_mask=alpha_mask,
        tversky_alpha=tversky_alpha,
        tversky_beta=tversky_beta,
        pos_weight_case=pos_weight_case,
        pos_weight_mask=pos_weight_mask,
        pos_weight_slice=pos_weight_slice,
        slice_weights_positives_only=slice_weights_positives_only,
        eps=eps,
    )


@register_loss("aneurysm_3d_combined")
def _make_aneurysm_3d_combined(
    loss_type: str = 'binary',
    lambda_cls: float = 1.0,
    lambda_seg: float = 1.0,
    cls_loss_type: str = "focal",  # "focal", "bce"
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    dice_smooth: float = 1.0,
    seg_loss_type: str = "dice_focal",  # "dice", "focal", "dice_focal", "bce_dice"
    alpha_seg: float = 0.5,  # for combined seg losses
    class_weights: list[float] | None = None,
    eps: float = 1e-6,
) -> nn.Module:
    return Aneurysm3DCombinedLoss(
        loss_type=loss_type,
        lambda_cls=lambda_cls,
        lambda_seg=lambda_seg,
        cls_loss_type=cls_loss_type,
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        dice_smooth=dice_smooth,
        seg_loss_type=seg_loss_type,
        alpha_seg=alpha_seg,
        class_weights=class_weights,
        eps=eps,
    )

@register_loss("aneurysm_3d_roi")
def _make_aneurysm_3d_roi(
    loss_type:str = 'binary', # binary or multi
    cls_loss_type: str = "bce",  # "focal", "bce"
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
) -> nn.Module:
    return Aneurysm3DROILoss(
        loss_type=loss_type, # binary or multi
        cls_loss_type=cls_loss_type,  # "focal", "bce"
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
    )

@register_loss("aneurysm_3d_roi_v2")
def _make_aneurysm_3d_roi_v2(
    loss_type:str = 'binary', # binary or multi
    cls_loss_type: str = "bce",  # "focal", "bce"
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=1 の重み（スカラ or (C,)）
    neg_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=0 の重み（スカラ or (C,)）
    channel_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # ch ごとのスケール（スカラ or (C,)）
) -> nn.Module:
    return Aneurysm3DROILossV2(
        loss_type=loss_type, # binary or multi
        cls_loss_type=cls_loss_type,  # "focal", "bce"
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,  
        neg_weight=neg_weight,  
        channel_weight=channel_weight,
    )

@register_loss("aneurysm_3d_roi_v3")
def _make_aneurysm_3d_roi_v3(
    loss_type:str = 'binary', # binary or multi
    cls_loss_type: str = "bce",  # "focal", "bce"
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=1 の重み（スカラ or (C,)）
    neg_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=0 の重み（スカラ or (C,)）
    channel_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # ch ごとのスケール（スカラ or (C,)）
    lambda_patch: float = 1.0,
    lambda_volume: float = 1.0,
    eps: float = 1e-6,
    reduction: str = "mean"
) -> nn.Module:
    return Aneurysm3DROILossV3(
        loss_type=loss_type, # binary or multi
        cls_loss_type=cls_loss_type,  # "focal", "bce"
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,  
        neg_weight=neg_weight,  
        channel_weight=channel_weight,
        lambda_patch=lambda_patch,
        lambda_volume=lambda_volume,
        eps=eps,
        reduction=reduction
    )

@register_loss("aneurysm_3d_roi_v4")
def _make_aneurysm_3d_roi_v4(
    cls_loss_type: str = "bce",  # "focal", "bce"
    focal_alpha: float = 0.25,
    focal_gamma: float = 2.0,
    pos_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=1 の重み（スカラ or (C,)）
    neg_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=0 の重み（スカラ or (C,)）
    channel_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # ch ごとのスケール（スカラ or (C,)）
    ce_label_smoothing: float = 0.0,
    plane_class_weight: Optional[Union[Sequence[float], torch.Tensor]] = None,    # (P,)
    modality_class_weight: Optional[Union[Sequence[float], torch.Tensor]] = None, # (M,)
    # mixing
    lambda_patch: float = 1.0,
    lambda_plane: float = 1.0,
    lambda_modality: float = 1.0,
    reduction: str = "mean",
    eps: float = 1e-6
) -> nn.Module:
    return Aneurysm3DROILossV4(
        cls_loss_type=cls_loss_type,  # "focal", "bce"
        focal_alpha=focal_alpha,
        focal_gamma=focal_gamma,
        pos_weight=pos_weight,  
        neg_weight=neg_weight,  
        channel_weight=channel_weight,
        ce_label_smoothing=ce_label_smoothing,
        plane_class_weight=plane_class_weight,    # (P,)
        modality_class_weight=modality_class_weight, # (M,)
        # mixing
        lambda_patch=lambda_patch,
        lambda_plane=lambda_plane,
        lambda_modality=lambda_modality,
        reduction=reduction,
        eps=eps
    )

@register_loss("patch_binary_focal")
def _make_patch_binary_focal(
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    label_smoothing: float = 0.0,
) -> nn.Module:
    return PatchBinaryFocalLoss(
        alpha=alpha,
        gamma=gamma,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
    )


@register_loss("patch_binary_focal_with_mask")
def _make_patch_binary_focal_with_mask(
    alpha: float = 0.25,
    gamma: float = 2.0,
    pos_weight: float | None = None,
    label_smoothing: float = 0.0,
    lambda_cls: float = 1.0,
    lambda_mask: float = 0.5,
    mask_loss_type: str = "focal",  # "focal", "dice", "bce"
    dice_smooth: float = 1.0,
) -> nn.Module:
    return PatchBinaryFocalLossWithMask(
        alpha=alpha,
        gamma=gamma,
        pos_weight=pos_weight,
        label_smoothing=label_smoothing,
        lambda_cls=lambda_cls,
        lambda_mask=lambda_mask,
        mask_loss_type=mask_loss_type,
        dice_smooth=dice_smooth,
    )


class MultiTaskLossLogits01(nn.Module):
    """
    outputs:
      out['output']        : [B, K]         (logits)
      out['output_slice']  : [B, S, K]      (logits)
      out['mask']          : [B, S, H, W]   (logits)

    labels:
      batch['targets']        : [B, K]         (0/1)
      batch['targets_slice']  : [B, S, K]      (0/1)
      batch['mask']           : [B, S, H, W]   (0/1)
      batch['slice_weights']  : [B, S]         (0..1, 任意)
    """

    def __init__(
        self,
        lambda_case: float = 1.0,
        lambda_slice: float = 1.0,
        lambda_mask: float = 2.0,
        # セグの合成
        alpha_mask: float = 0.5,  # seg = alpha*BCE + (1-alpha)*Tversky
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        # 不均衡（任意）
        pos_weight_case: float | torch.Tensor | None = None,  # [K] もOK
        pos_weight_mask: float | None = None,
        pos_weight_slice: float | None = None,  # sliceロスの正例倍率
        # slice_weights の適用方法
        slice_weights_positives_only: bool = True,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.l_case = lambda_case
        self.l_slice = lambda_slice
        self.l_mask = lambda_mask

        self.a_mask = alpha_mask
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.eps = eps

        # pos_weight（BCEWithLogits）
        if isinstance(pos_weight_case, torch.Tensor):
            self.posw_case = pos_weight_case
        elif pos_weight_case is None:
            self.posw_case = None
        else:
            self.posw_case = torch.tensor(pos_weight_case).reshape(-1)  # 1 or [K]

        self.posw_mask = None if pos_weight_mask is None else torch.tensor([pos_weight_mask], dtype=torch.float32)
        self.posw_slice = pos_weight_slice  # スカラーを陽性項に掛ける

        self.slice_weights_positives_only = slice_weights_positives_only

        # smp の Tversky（重みなし版は使わず、手書き重み付きで統一）
        # self.tversky_smp = smp.losses.TverskyLoss(mode='multilabel', alpha=tversky_alpha, beta=tversky_beta, log_loss=False)

    @staticmethod
    def _expand_weight_like(w, ref):
        while w.ndim < ref.ndim:
            w = w.unsqueeze(-1)
        return w

    def _bce_logits(self, logits, target, pos_weight_tensor=None, weight=None):
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=(pos_weight_tensor.to(logits.device) if pos_weight_tensor is not None else None),
        )
        if weight is not None:
            loss = loss * weight
        return loss.mean()

    def _bce_logits_slice(self, logits_bsk, target_bsk, w_bs=None, pos_weight_slice=None):
        """
        logits/target: [B,S,K], w_bs: [B,S]
        - slice_weights を陽性にのみ掛ける（既定）
        - pos_weight_slice は陽性項に倍率を掛ける
        """
        loss_map = F.binary_cross_entropy_with_logits(logits_bsk, target_bsk, reduction="none")  # [B,S,K]

        if pos_weight_slice is not None:
            loss_map = torch.where(target_bsk > 0.5, loss_map * float(pos_weight_slice), loss_map)

        if w_bs is None:
            return loss_map.mean()

        w = w_bs.unsqueeze(-1)  # [B,S,1]
        if self.slice_weights_positives_only:
            w_full = torch.where(target_bsk > 0.5, w, torch.ones_like(w))  # 陽性のみ強調
        else:
            w_full = w

        num = (loss_map * w_full).sum()
        den = w_full.sum()
        return (num / den) if den > 0 else loss_map.mean()

    def _tversky_weighted(self, logits, target, w_bs=None):
        """
        重み付きTversky（手書き）
        logits/target: [B,S,H,W], w_bs: [B,S] or None
        """
        p = torch.sigmoid(logits)
        tp = (p * target).sum(dim=(-1, -2))  # [B,S]
        fp = (p * (1 - target)).sum(dim=(-1, -2))
        fn = ((1 - p) * target).sum(dim=(-1, -2))
        t = (tp + self.eps) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + self.eps)
        loss_s = 1.0 - t  # [B,S]
        if w_bs is None:
            return loss_s.mean()
        w = w_bs.clamp_min(1e-6)
        return (loss_s * w).sum() / w.sum()

    def forward(self, out: dict, batch: dict):
        device = out["output"].device

        # ----- case: [B,K] -----
        y_case = batch["targets"].to(device).float()
        posw_case = None
        if self.posw_case is not None:
            # 1要素 or [K] どちらも可
            posw_case = self.posw_case.to(device)
        L_case = self._bce_logits(out["output"], y_case, pos_weight_tensor=posw_case)

        # ----- slice×position: [B,S,K] -----
        y_slice = batch["targets_slice"].to(device).float()
        w_slice = batch.get("slice_weights", None)
        if w_slice is not None:
            w_slice = w_slice.to(device).float()

        L_slice = self._bce_logits_slice(out["output_slice"], y_slice, w_bs=w_slice, pos_weight_slice=self.posw_slice)

        # ----- segmentation: BCE + Tversky（どちらも slice_weights 反映） -----
        y_mask = batch["mask"].to(device).float()
        bce_map = F.binary_cross_entropy_with_logits(out["mask"], y_mask, reduction="none")  # [B,S,H,W]

        if w_slice is None:
            L_mask_bce = bce_map.mean()
        else:
            bce_slice = bce_map.mean(dim=(-1, -2))  # [B,S]  ← 先に各スライスの画素平均
            w = w_slice.clamp_min(1e-6)  # [B,S]
            L_mask_bce = (bce_slice * w).sum() / w.sum()  # スライス重みで平均

        L_mask_tvs = self._tversky_weighted(out["mask"], y_mask, w_slice)
        L_mask = self.a_mask * L_mask_bce + (1 - self.a_mask) * L_mask_tvs

        total = self.l_case * L_case + self.l_slice * L_slice + self.l_mask * L_mask
        return {
            "loss": total,
            "loss_case": L_case.detach(),
            "loss_slice": L_slice.detach(),
            "loss_mask": L_mask.detach(),
        }


class FocalLoss(nn.Module):
    """Focal Loss implementation for addressing class imbalance"""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, eps: float = 1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Prediction logits (..., num_classes) or (...,) for binary
            targets: Ground truth labels (..., num_classes) or (...,) for binary
        """
        # Clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-100, max=100)
        
        # Compute probabilities with numerical stability
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Compute focal weight
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.where(targets == 1, probs, 1 - probs)
        pt = torch.clamp(pt, min=self.eps, max=1 - self.eps)  # Prevent zero which causes NaN
        focal_weight = (1 - pt) ** self.gamma
        
        # Apply alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * focal_weight * ce_loss
        
        # Check for NaN and replace with zero
        focal_loss = torch.where(torch.isnan(focal_loss), torch.zeros_like(focal_loss), focal_loss)
        
        return focal_loss.mean()


class DiceLoss(nn.Module):
    """Dice Loss for segmentation tasks"""
    
    def __init__(self, smooth: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.smooth = smooth
        self.eps = eps
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Prediction logits (B, 1, D, H, W) or (B, D, H, W)
            targets: Ground truth masks (B, 1, D, H, W) or (B, D, H, W)
        """
        # Clamp logits to prevent overflow
        logits = torch.clamp(logits, min=-100, max=100)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1 - self.eps)
        
        # Flatten
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        
        # Calculate dice coefficient with numerical stability
        intersection = (probs * targets).sum(dim=1)
        probs_sum = probs.sum(dim=1)
        targets_sum = targets.sum(dim=1)
        
        # Add small epsilon to prevent division by zero
        dice_coeff = (2.0 * intersection + self.smooth) / (
            probs_sum + targets_sum + self.smooth + self.eps
        )
        
        # Clamp dice coefficient to valid range
        dice_coeff = torch.clamp(dice_coeff, min=0.0, max=1.0)
        
        dice_loss = 1.0 - dice_coeff
        
        # Check for NaN and replace with 1.0 (maximum loss)
        dice_loss = torch.where(torch.isnan(dice_loss), torch.ones_like(dice_loss), dice_loss)
        
        return dice_loss.mean()

class TverskyLoss(nn.Module):
    """
    Tversky Loss for binary segmentation.
    alpha: weight for false positives (FP)
    beta : weight for false negatives (FN)
    """
    def __init__(self, alpha: float = 0.7, beta: float = 0.3, smooth: float = 1.0, eps: float = 1e-6):
        super().__init__()
        assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0, "alpha/beta must be in [0,1]"
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Prediction logits (B, 1, D, H, W) or (B, D, H, W)
            targets: Ground truth masks (B, 1, D, H, W) or (B, D, H, W), {0,1}
        """
        # Clamp logits and get probabilities
        logits = torch.clamp(logits, min=-100, max=100)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # Flatten (batch-wise)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        # Tversky components
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        # Tversky index with numerical stability
        denom = tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps
        tversky = (tp + self.smooth) / denom

        # Clamp index and convert to loss
        tversky = torch.clamp(tversky, min=0.0, max=1.0)
        loss = 1.0 - tversky

        # NaN guard
        loss = torch.where(torch.isnan(loss), torch.ones_like(loss), loss)

        return loss.mean()

class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky Loss for binary segmentation.
    alpha: weight for false positives (FP)
    beta : weight for false negatives (FN)
    gamma: focusing parameter (>1 focuses more on hard examples)
    """
    def __init__(
        self,
        alpha: float = 0.7,
        beta: float = 0.3,
        gamma: float = 0.75,
        smooth: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        assert 0.0 <= alpha <= 1.0 and 0.0 <= beta <= 1.0, "alpha/beta must be in [0,1]"
        assert gamma > 0.0, "gamma must be > 0"
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.eps = eps

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, 1, D, H, W) or (B, D, H, W)  — raw logits
            targets: (B, 1, D, H, W) or (B, D, H, W) — {0,1}
        """
        # numerics: clamp -> sigmoid -> clamp
        logits = torch.clamp(logits, min=-100, max=100)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        # flatten per-batch
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        # components
        tp = (probs * targets).sum(dim=1)
        fp = (probs * (1.0 - targets)).sum(dim=1)
        fn = ((1.0 - probs) * targets).sum(dim=1)

        # Tversky index
        denom = tp + self.alpha * fp + self.beta * fn + self.smooth + self.eps
        tversky = (tp + self.smooth) / denom
        tversky = torch.clamp(tversky, min=0.0, max=1.0)

        # Focal Tversky
        loss = torch.pow(1.0 - tversky, self.gamma)

        # NaN guard
        loss = torch.where(torch.isnan(loss), torch.ones_like(loss), loss)

        return loss.mean()
    
class Aneurysm3DCombinedLoss(nn.Module):
    """
    Combined loss for 3D aneurysm detection:
    - Classification loss for anatomical location prediction (Focal or BCE)
    - Segmentation loss for aneurysm mask prediction
    
    Expected model outputs:
        output['output']: (B, num_classes) - classification logits
        output['mask']: (B, 1, D, H, W) - segmentation logits
    
    Expected targets:
        targets['targets']: (B, num_classes) - classification labels
        targets['mask']: (B, 1, D, H, W) - segmentation masks
        
    Usage:
        # With BCE classification loss
        loss_fn = create_loss('aneurysm_3d_combined', cls_loss_type='bce')
        
        # With Focal classification loss (default)
        loss_fn = create_loss('aneurysm_3d_combined', cls_loss_type='focal')
    """
    
    def __init__(
        self,
        loss_type:str = 'binary', # binary or multi
        lambda_cls: float = 1.0,
        lambda_seg: float = 1.0,
        cls_loss_type: str = "bce",  # "focal", "bce"
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        dice_smooth: float = 1.0,
        seg_loss_type: str = "dice_focal",  # "dice", "focal", "dice_focal", "bce_dice"
        tvs_alpha: float   = 0.3,
        tvs_beta: float    = 0.7,
        tvs_focal_gamma: float = 2.0,
        alpha_seg: float = 0.5,  # for combined seg losses
        class_weights: list[float] | None = None,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.lambda_cls = lambda_cls
        self.lambda_seg = lambda_seg
        self.cls_loss_type = cls_loss_type
        self.seg_loss_type = seg_loss_type
        self.alpha_seg = alpha_seg
        self.eps = eps
        
        if cls_loss_type=='bce':
            # Classification loss (Focal Loss or BCE for class imbalance)
            self.loss_cls = nn.BCEWithLogitsLoss()
        elif cls_loss_type=='focal':
            self.loss_cls = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)
        else:
            raise ValueError(f"Unknown cls_loss_type: {self.cls_loss_type}")
        
        # Segmentation losses
        self.dice_loss = DiceLoss(smooth=dice_smooth, eps=eps)
        self.seg_focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)
        self.seg_tvs_focal_loss = FocalTverskyLoss(alpha=tvs_alpha, beta=tvs_beta, gamma=tvs_focal_gamma)
        
        # Class weights for classification
        self.class_weights = None
        if class_weights is not None:
            self.class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: Model outputs dict with 'output' and 'mask'
            targets: Target dict with 'targets' and 'mask'
        """
        device = outputs['output'].device
        
        # Move class weights to device
        if self.class_weights is not None:
            self.class_weights = self.class_weights.to(device)
        
        # === Classification Loss ===
        cls_logits = outputs['output']  # (B, num_classes)
        if self.loss_type=='binary':
            cls_targets = targets['target_aneurysm'].to(device).float()  # (B, num_classes)
        elif self.loss_type=='multi':
            cls_targets = targets['target_site'].to(device).float()  # (B, num_classes)
        else:
            raise ValueError('in valid type')
        # Choose classification loss type
        cls_loss = self.loss_cls(cls_logits, cls_targets)
        
        # === Segmentation Loss ===
        # Check if mask is provided in both outputs and targets
        seg_loss = torch.tensor(0.0, device=device)
        if 'mask' in outputs and 'mask' in targets:
            seg_logits = outputs['mask']  # (B, 1, D, H, W)
            seg_targets = targets['mask'].to(device).unsqueeze(1).float()  # (B, 1, D, H, W)
            
            if self.seg_loss_type == "dice":
                seg_loss = self.dice_loss(seg_logits, seg_targets)
            elif self.seg_loss_type == "focal":
                seg_loss = self.seg_focal_loss(seg_logits, seg_targets)
            elif self.seg_loss_type == "dice_focal":
                dice_loss = self.dice_loss(seg_logits, seg_targets)
                focal_loss = self.seg_focal_loss(seg_logits, seg_targets)
                seg_loss = self.alpha_seg * dice_loss + (1 - self.alpha_seg) * focal_loss
            elif self.seg_loss_type == "bce_dice":
                bce_loss = F.binary_cross_entropy_with_logits(seg_logits, seg_targets)
                dice_loss = self.dice_loss(seg_logits, seg_targets)
                seg_loss = self.alpha_seg * bce_loss + (1 - self.alpha_seg) * dice_loss
            elif self.seg_loss_type == "focal_focaltvs":
                focal_loss = self.seg_focal_loss(seg_logits, seg_targets)
                focal_tvs_loss = self.seg_tvs_focal_loss(seg_logits, seg_targets)
                seg_loss = self.alpha_seg * focal_tvs_loss + (1 - self.alpha_seg) * focal_loss
            else:
                raise ValueError(f"Unknown seg_loss_type: {self.seg_loss_type}")
        
        # === Total Loss ===
        total_loss = self.lambda_cls * cls_loss + self.lambda_seg * seg_loss
        
        return {
            "loss": total_loss,
            "loss_cls": cls_loss.detach(),
            "loss_seg": seg_loss.detach(),
        }


class Aneurysm3DROILoss(nn.Module):
    """
    Combined loss for 3D aneurysm detection:
    - Classification loss for anatomical location prediction (Focal or BCE)
    
    Expected model outputs:
        output['output']: (B, num_classes) - classification logits
        
    Expected targets:
        targets['mask']: (B, 1, D, H, W) - segmentation masks
        
    Usage:
        # With BCE classification loss
        loss_fn = create_loss('aneurysm_3d_combined', cls_loss_type='bce')
        
        # With Focal classification loss (default)
        loss_fn = create_loss('aneurysm_3d_combined', cls_loss_type='focal')
    """
    
    def __init__(
        self,
        loss_type:str = 'binary', # binary or multi
        cls_loss_type: str = "bce",  # "focal", "bce"
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.loss_type = loss_type
        self.cls_loss_type = cls_loss_type
        self.eps = eps
        
        if cls_loss_type=='bce':
            # Classification loss (Focal Loss or BCE for class imbalance)
            self.loss_cls = nn.BCEWithLogitsLoss()
        elif cls_loss_type=='focal':
            self.loss_cls = FocalLoss(alpha=focal_alpha, gamma=focal_gamma, eps=eps)
        else:
            raise ValueError(f"Unknown cls_loss_type: {self.cls_loss_type}")
        
        
    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: Model outputs dict with 'output' and 'mask'
            targets: Target dict with 'targets' and 'mask'
        """
        
        # === Classification Loss ===
        cls_logits = outputs['output']  # (B, num_classes)
        cls_target_map = targets['mask_map']
        B, C, D, H, W = cls_target_map.shape
        N = D * H * W
        cls_target_map = cls_target_map.view(B, C, N)
        cls_target_map = cls_target_map.transpose(1,2) # B, N, C

        # Choose classification loss type
        cls_loss = self.loss_cls(cls_logits, cls_target_map)
        
        
        return {
            "loss": cls_loss,
        }
    
class Aneurysm3DROILossV2(nn.Module):
    """
    Combined loss for 3D aneurysm detection with per-label(0/1) and per-channel weights.
    - outputs['output']: (B, C)  ※(B, C) は (B, N, C) へブロードキャストされる
    - targets['mask_map']: (B, C, D, H, W) → (B, N, C) に整形（max pooling はしない）
    """

    def __init__(
        self,
        loss_type: str = 'binary',             # 既存の引数はそのまま
        cls_loss_type: str = "bce",            # "focal", "bce"
        focal_alpha: Union[float, Sequence[float], torch.Tensor, None] = 0.25,
        focal_gamma: float = 2.0,
        # 追加: 0/1 と ch の重み
        pos_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=1 の重み（スカラ or (C,)）
        neg_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # y=0 の重み（スカラ or (C,)）
        channel_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # ch ごとのスケール（スカラ or (C,)）
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.cls_loss_type = cls_loss_type
        self.focal_gamma = focal_gamma
        self.eps = eps
        self.reduction = reduction

        # 既存の設計を踏襲：BCE は nn.BCEWithLogitsLoss を使う（ただし後段で重み付けするので reduction='none'）
        if cls_loss_type == 'bce':
            self.loss_cls = nn.BCEWithLogitsLoss(reduction='none')
        elif cls_loss_type == 'focal':
            self.loss_cls = None  # Focal は下で手計算（最小限の変更で要素別重みに対応するため）
        else:
            raise ValueError(f"Unknown cls_loss_type: {self.cls_loss_type}")

        # α は ch ごとにも受け取れる
        if focal_alpha is None:
            self.register_buffer("focal_alpha", None)
        else:
            self.register_buffer("focal_alpha", torch.as_tensor(focal_alpha, dtype=torch.float32))

        # 重みはバッファとして保持（学習しない）
        def to_buf(x):
            return torch.as_tensor(x, dtype=torch.float32)

        self.register_buffer("pos_w", to_buf(pos_weight))
        self.register_buffer("neg_w", to_buf(neg_weight))
        self.register_buffer("ch_w",  to_buf(channel_weight))

    def _view_as_channels(self, w: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """
        (C,) or () を like の最後次元 C にブロードキャストできる形へ。
        like: (..., C) 例：(B, N, C) or (B, C)
        """
        if w.numel() == 1:
            return w
        # 最後の次元が C になるように reshape
        shape = [1] * (like.dim() - 1) + [-1]
        return w.view(*shape)

    def _focal_bce_with_logits_elementwise(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        要素ごと focal BCE（reduction='none' 相当）
        """
        # 基本の BCE（要素ごと）
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')

        # p_t
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        focal_factor = (1.0 - pt).clamp_(min=0.0).pow(self.focal_gamma)

        # alpha_t（あれば）
        if self.focal_alpha is None:
            alpha_t = 1.0
        else:
            a = self._view_as_channels(self.focal_alpha.to(logits.device, logits.dtype), logits)
            if a.numel() == 1:
                alpha_t = torch.where(target > 0.5, a, 1.0 - a)
            else:
                alpha_t = torch.where(target > 0.5, a, 1.0 - a)

        return alpha_t * focal_factor * bce

    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: Model outputs dict with 'output'
            targets: Target dict with 'mask_map'
        """
        # === Classification Loss ===（既存の流れを維持）
        cls_logits = outputs['output']  # (B, C)
        cls_target_map = targets['mask_map']  # (B, C, D, H, W)

        B, C, D, H, W = cls_target_map.shape
        N = D * H * W
        cls_target_map = cls_target_map.view(B, C, N).transpose(1, 2)  # (B, N, C)

        # ログイット (B, C) は (B, N, C) へ自動ブロードキャストされる想定（元コードの挙動を維持）
        # 要素ごとの損失（reduction='none'）
        if self.cls_loss_type == 'bce':
            elem_loss = self.loss_cls(cls_logits, cls_target_map)        # (B, N, C)
        else:  # focal
            elem_loss = self._focal_bce_with_logits_elementwise(cls_logits, cls_target_map)  # (B, N, C)

        # ===== 重み付け =====
        # 0/1 重み + ch 重み（ターゲットに応じて pos/neg を選択）
        pw = self._view_as_channels(self.pos_w.to(cls_logits.device, cls_logits.dtype), elem_loss)
        nw = self._view_as_channels(self.neg_w.to(cls_logits.device, cls_logits.dtype), elem_loss)
        cw = self._view_as_channels(self.ch_w.to(cls_logits.device, cls_logits.dtype), elem_loss)

        elem_w = torch.where(cls_target_map > 0.5, pw, nw) * cw  # (B, N, C)
        elem_loss = elem_loss * elem_w

        # reduction
        if self.reduction == "mean":
            cls_loss = elem_loss.mean()
        elif self.reduction == "sum":
            cls_loss = elem_loss.sum()
        else:
            cls_loss = elem_loss  # 'none' など

        return {
            "loss": cls_loss,
        }
    
class Aneurysm3DROILossV3(nn.Module):
    """
    Combined loss for 3D aneurysm detection.

    Inputs:
      outputs:
        - 'patch_output': (B, N, C)  # パッチ/ボクセルごとのロジット（推奨）
        - 'output': (B, C)           # 旧仕様。存在する場合は (B, N, C) へブロードキャストしてpatch lossに使う
        - 'volume_output': (B, C)    # 追加: ボリューム全体のロジット
      targets:
        - 'mask_map': (B, C, D, H, W)  # パッチ用は (B, N, C) に整形（max pooling はしない）
                                        # ボリューム用ターゲットは max pooling: (B, C) に集約
    """

    def __init__(
        self,
        loss_type: str = 'binary',       # 既存引数（未使用だが維持）
        cls_loss_type: str = "bce",      # "focal" or "bce"
        focal_alpha: Union[float, Sequence[float], torch.Tensor, None] = 0.25,
        focal_gamma: float = 2.0,
        # 0/1 と ch の重み
        pos_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,   # y=1 の重み（スカラ or (C,)）
        neg_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,   # y=0 の重み（スカラ or (C,)）
        channel_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # ch ごとのスケール（スカラ or (C,)）
        # それぞれの項の重み
        lambda_patch: float = 1.0,
        lambda_volume: float = 1.0,
        eps: float = 1e-6,
        reduction: str = "mean",
    ):
        super().__init__()
        self.loss_type = loss_type
        self.cls_loss_type = cls_loss_type
        self.focal_gamma = focal_gamma
        self.eps = eps
        self.reduction = reduction
        self.lambda_patch = float(lambda_patch)
        self.lambda_volume = float(lambda_volume)

        if cls_loss_type == 'bce':
            self.loss_cls = nn.BCEWithLogitsLoss(reduction='none')
        elif cls_loss_type == 'focal':
            self.loss_cls = None
        else:
            raise ValueError(f"Unknown cls_loss_type: {self.cls_loss_type}")

        # α は ch ごとにも受け取れる
        if focal_alpha is None:
            self.register_buffer("focal_alpha", None)
        else:
            self.register_buffer("focal_alpha", torch.as_tensor(focal_alpha, dtype=torch.float32))

        def to_buf(x):
            return torch.as_tensor(x, dtype=torch.float32)

        self.register_buffer("pos_w", to_buf(pos_weight))
        self.register_buffer("neg_w", to_buf(neg_weight))
        self.register_buffer("ch_w",  to_buf(channel_weight))

    # ------- helpers -------
    def _view_as_channels(self, w: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        """
        (C,) or () を like の最後次元 C にブロードキャストできる形へ。
        like: (..., C) 例：(B, N, C) or (B, C)
        """
        if w.numel() == 1:
            return w
        shape = [1] * (like.dim() - 1) + [-1]
        return w.view(*shape)

    def _focal_bce_with_logits_elementwise(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """要素ごとの focal BCE（reduction='none' 相当）"""
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        focal_factor = (1.0 - pt).clamp_(min=0.0).pow(self.focal_gamma)

        if self.focal_alpha is None:
            alpha_t = 1.0
        else:
            a = self._view_as_channels(self.focal_alpha.to(logits.device, logits.dtype), logits)
            alpha_t = torch.where(target > 0.5, a, 1.0 - a)

        return alpha_t * focal_factor * bce

    def _elemwise_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """BCE/Focal を要素ごとに返す（reduction='none'）"""
        if self.cls_loss_type == 'bce':
            return self.loss_cls(logits, target)
        else:
            return self._focal_bce_with_logits_elementwise(logits, target)

    def _apply_weights_and_reduce(self, elem_loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        pos/neg + channel の重みを乗じてreduction
        elem_loss, target の形は一致（(..., C)）
        """
        pw = self._view_as_channels(self.pos_w.to(elem_loss.device, elem_loss.dtype), elem_loss)
        nw = self._view_as_channels(self.neg_w.to(elem_loss.device, elem_loss.dtype), elem_loss)
        cw = self._view_as_channels(self.ch_w.to(elem_loss.device, elem_loss.dtype), elem_loss)
        elem_w = torch.where(target > 0.5, pw, nw) * cw
        elem_loss = elem_loss * elem_w

        if self.reduction == "mean":
            return elem_loss.mean()
        elif self.reduction == "sum":
            return elem_loss.sum()
        else:
            return elem_loss  # 'none'

    # ------- forward -------
    def forward(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        outputs:
          - 'patch_output': (B, N, C)  ※推奨
          - 'output': (B, C)           ※旧：存在するなら (B, N, C) に拡張して patch loss に使用
          - 'volume_output': (B, C)    ※追加
        targets:
          - 'mask_map': (B, C, D, H, W)
        """
        mask_map = targets['mask_map']            # (B, C, D, H, W)
        B, C, D, H, W = mask_map.shape
        N = D * H * W

        # ---- Patch-wise loss ----
        patch_logits = None
        if 'patch_output' in outputs:
            patch_logits = outputs['patch_output']           # (B, N, C)
        elif 'output' in outputs:
            # 旧仕様: (B, C) を (B, N, C) へ拡張
            patch_logits = outputs['output'].unsqueeze(1).expand(B, N, C)
        # ターゲット（パッチ）は max pooling せずに整形
        patch_target = mask_map.view(B, C, N).transpose(1, 2)  # (B, N, C)

        loss_patch = None
        if patch_logits is not None:
            if patch_logits.shape[:2] != (B, N) or patch_logits.shape[-1] != C:
                raise ValueError(f"patch_output shape {tuple(patch_logits.shape)} must be (B={B}, N={N}, C={C})")
            elem_patch = self._elemwise_loss(patch_logits, patch_target)      # (B, N, C)
            loss_patch = self._apply_weights_and_reduce(elem_patch, patch_target)

        # ---- Volume-wise loss ----
        loss_volume = None
        if 'volume_output' in outputs:
            vol_logits = outputs['volume_output']           # (B, C)
            if vol_logits.shape != (B, C):
                raise ValueError(f"volume_output shape {tuple(vol_logits.shape)} must be (B={B}, C={C})")
            # ボリュームターゲットは max pooling で集約（ラベルがどこか1でもあれば1）
            vol_target = (mask_map.view(B, C, N) > 0).any(dim=2).float()     # (B, C)

            elem_vol = self._elemwise_loss(vol_logits, vol_target)           # (B, C)
            loss_volume = self._apply_weights_and_reduce(elem_vol, vol_target)

        # ---- 合算 ----
        if loss_patch is None and loss_volume is None:
            raise ValueError("outputs must contain 'patch_output' or 'output', or 'volume_output'.")

        loss = 0.0
        if loss_patch is not None:
            loss = loss + self.lambda_patch * loss_patch
        if loss_volume is not None:
            loss = loss + self.lambda_volume * loss_volume

        out = {"loss": loss}
        if loss_patch is not None:
            out["loss_patch"] = loss_patch
        if loss_volume is not None:
            out["loss_volume"] = loss_volume
        return out
    

class Aneurysm3DROILossV4(nn.Module):
    """
    Combined loss for 3-head setting:
      - Patch head (multi-label): outputs['output'] -> (B, N, C) or (B, C)
      - Plane head (multi-class):  outputs['output_plane'] -> (B, P)  (e.g., P=3)
      - Modality head (multi-class): outputs['output_modality'] -> (B, M) (e.g., M=4)

    Targets:
      - targets['mask_map']: (B, C, D, H, W)  # for patch loss (reshape to (B,N,C), no max pool)
      - targets['plane']: (B,) Long           # class index [0..P-1]
      - targets['modality']: (B,) Long        # class index [0..M-1]
    """

    def __init__(
        self,
        # patch loss
        cls_loss_type: str = "bce",              # "bce" or "focal"
        focal_alpha: Union[float, Sequence[float], torch.Tensor, None] = 0.25,
        focal_gamma: float = 2.0,
        pos_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,   # (scalar or (C,))
        neg_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,   # (scalar or (C,))
        channel_weight: Union[float, Sequence[float], torch.Tensor] = 1.0,  # (scalar or (C,))
        # plane/modality loss
        ce_label_smoothing: float = 0.0,
        plane_class_weight: Optional[Union[Sequence[float], torch.Tensor]] = None,    # (P,)
        modality_class_weight: Optional[Union[Sequence[float], torch.Tensor]] = None, # (M,)
        # mixing
        lambda_patch: float = 1.0,
        lambda_plane: float = 1.0,
        lambda_modality: float = 1.0,
        reduction: str = "mean",
        eps: float = 1e-6,
    ):
        super().__init__()
        # --- patch head config ---
        self.cls_loss_type = cls_loss_type
        self.focal_gamma = float(focal_gamma)
        self.reduction = reduction
        self.eps = eps
        if cls_loss_type == "bce":
            self.loss_cls = nn.BCEWithLogitsLoss(reduction="none")   # elementwise, we weight after
        elif cls_loss_type == "focal":
            self.loss_cls = None
        else:
            raise ValueError(f"Unknown cls_loss_type: {cls_loss_type}")

        # alpha for focal (scalar or per-class)
        if focal_alpha is None:
            self.register_buffer("focal_alpha", None)
        else:
            self.register_buffer("focal_alpha", torch.as_tensor(focal_alpha, dtype=torch.float32))

        def to_buf(x):
            return torch.as_tensor(x, dtype=torch.float32)

        self.register_buffer("pos_w", to_buf(pos_weight))
        self.register_buffer("neg_w", to_buf(neg_weight))
        self.register_buffer("ch_w",  to_buf(channel_weight))

        # --- plane/modality config ---
        self.ce_label_smoothing = float(ce_label_smoothing)
        self.register_buffer("plane_w", None, persistent=False)
        self.register_buffer("modality_w", None, persistent=False)
        if plane_class_weight is not None:
            self.register_buffer("plane_w", to_buf(plane_class_weight))
        if modality_class_weight is not None:
            self.register_buffer("modality_w", to_buf(modality_class_weight))

        # --- lambdas ---
        self.lambda_patch = float(lambda_patch)
        self.lambda_plane = float(lambda_plane)
        self.lambda_modality = float(lambda_modality)

    # ---------- helpers ----------
    def _view_as_channels(self, w: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        # reshape scalar or (C,) -> broadcast over like(..., C)
        if w.numel() == 1:
            return w
        shape = [1] * (like.dim() - 1) + [-1]
        return w.view(*shape)

    def _focal_bce_with_logits_elementwise(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        p = torch.sigmoid(logits)
        pt = p * target + (1 - p) * (1 - target)
        focal_factor = (1.0 - pt).clamp(min=0.0).pow(self.focal_gamma)

        if self.focal_alpha is None:
            alpha_t = 1.0
        else:
            a = self._view_as_channels(self.focal_alpha.to(logits.device, logits.dtype), logits)
            alpha_t = torch.where(target > 0.5, a, 1.0 - a)

        return alpha_t * focal_factor * bce

    def _elemwise_patch_loss(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.cls_loss_type == "bce":
            return self.loss_cls(logits, target)
        else:
            return self._focal_bce_with_logits_elementwise(logits, target)

    def _apply_weights_and_reduce(self, elem_loss: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # pos/neg + channel weights
        pw = self._view_as_channels(self.pos_w.to(elem_loss.device, elem_loss.dtype), elem_loss)
        nw = self._view_as_channels(self.neg_w.to(elem_loss.device, elem_loss.dtype), elem_loss)
        cw = self._view_as_channels(self.ch_w.to(elem_loss.device, elem_loss.dtype), elem_loss)
        elem_w = torch.where(target > 0.5, pw, nw) * cw
        elem_loss = elem_loss * elem_w

        if self.reduction == "mean":
            return elem_loss.mean()
        elif self.reduction == "sum":
            return elem_loss.sum()
        else:
            return elem_loss  # 'none'

    # ---------- forward ----------
    def forward(self, outputs: dict[str, torch.Tensor], targets: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        outputs:
          - 'output': (B,N,C) or (B,C)  (patch head, multilabel)
          - 'output_plane': (B,P)       (plane head, multiclass)
          - 'output_modality': (B,M)    (modality head, multiclass)
        targets:
          - 'mask_map': (B,C,D,H,W)
          - 'plane': (B,)  long
          - 'modality': (B,) long
        """
        loss_dict = {}

        # ----- Patch head (multi-label) -----
        assert 'mask_map' in targets, "targets must contain 'mask_map' for patch head"
        mask_map = targets['mask_map']                  # (B, C, D, H, W)
        B, C, D, H, W = mask_map.shape
        N = D * H * W
        patch_target = mask_map.view(B, C, N).transpose(1, 2).contiguous()  # (B, N, C)

        assert 'output' in outputs, "outputs must contain 'output' for patch head"
        patch_logits = outputs['output']
        # Accept both (B,N,C) and (B,C) (legacy)
        if patch_logits.dim() == 2:        # (B, C) -> expand to (B, N, C)
            patch_logits = patch_logits.unsqueeze(1).expand(B, N, C)
        elif patch_logits.dim() == 3:
            if patch_logits.shape[:2] != (B, N) or patch_logits.shape[-1] != C:
                raise ValueError(f"'output' shape {tuple(patch_logits.shape)} must be (B={B}, N={N}, C={C})")
        else:
            raise ValueError(f"'output' must be (B,C) or (B,N,C), got {tuple(patch_logits.shape)}")

        elem_patch = self._elemwise_patch_loss(patch_logits, patch_target)  # (B,N,C)
        loss_patch = self._apply_weights_and_reduce(elem_patch, patch_target)
        loss_dict["loss_patch"] = loss_patch

        # ----- Plane head (multi-class) -----
        loss_plane = torch.tensor(0.0, device=patch_logits.device)
        if "output_plane" in outputs and "plane_encoded" in targets and targets["plane_encoded"] is not None:
            plane_logits = outputs["output_plane"]   # (B, P)
            plane_target = targets["plane_encoded"].to(dtype=torch.long, device=plane_logits.device)  # (B,)
            w = None if self.plane_w is None else self.plane_w.to(plane_logits.device, plane_logits.dtype)
            loss_plane = F.cross_entropy(
                plane_logits, plane_target,
                weight=w, reduction="mean", label_smoothing=self.ce_label_smoothing
            )
        loss_dict["loss_plane"] = loss_plane

        # ----- Modality head (multi-class) -----
        loss_modality = torch.tensor(0.0, device=patch_logits.device)
        if "output_modality" in outputs and "modality_encoded" in targets and targets["modality_encoded"] is not None:
            modality_logits = outputs["output_modality"]  # (B, M)
            modality_target = targets["modality_encoded"].to(dtype=torch.long, device=modality_logits.device)  # (B,)
            w = None if self.modality_w is None else self.modality_w.to(modality_logits.device, modality_logits.dtype)
            loss_modality = F.cross_entropy(
                modality_logits, modality_target,
                weight=w, reduction="mean", label_smoothing=self.ce_label_smoothing
            )
        loss_dict["loss_modality"] = loss_modality

        # ----- total -----
        total = self.lambda_patch * loss_patch + self.lambda_plane * loss_plane + self.lambda_modality * loss_modality
        loss_dict["loss"] = total
        return loss_dict


class PatchBinaryFocalLoss(nn.Module):
    """
    Focal Loss for patch-based binary classification
    Useful for training with patch datasets where class imbalance is common
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        
        if pos_weight is not None:
            self.pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        else:
            self.pos_weight_tensor = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Prediction logits (B,) or (B, 1)
            targets: Ground truth labels (B,) or (B, 1)
        """
        # Ensure same shape
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        
        # Move pos_weight to device
        pos_weight = None
        if self.pos_weight_tensor is not None:
            pos_weight = self.pos_weight_tensor.to(logits.device)
        
        # Standard BCE loss
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none', pos_weight=pos_weight
        )
        
        # Focal loss components
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        focal_loss = alpha_t * focal_weight * ce_loss
        return focal_loss.mean()


class PatchBinaryFocalLossWithMask(nn.Module):
    """
    Enhanced Focal Loss for patch-based binary classification that incorporates mask information
    Combines classification focal loss with mask-aware losses for better lesion localization
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        pos_weight: float | None = None,
        label_smoothing: float = 0.0,
        lambda_cls: float = 1.0,
        lambda_mask: float = 0.5,
        mask_loss_type: str = "focal",  # "focal", "dice", "bce"
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.lambda_cls = lambda_cls
        self.lambda_mask = lambda_mask
        self.mask_loss_type = mask_loss_type
        self.dice_smooth = dice_smooth
        
        if pos_weight is not None:
            self.pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32)
        else:
            self.pos_weight_tensor = None
        
        # Initialize mask loss components
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice_loss = DiceLoss(smooth=dice_smooth)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor | None = None) -> dict:
        """
        Args:
            logits: Classification prediction logits (B,) or (B, 1) for binary classification
            targets: Ground truth labels (B,) or (B, 1) for binary classification  
            mask: Optional 3D mask tensor (B, Z, H, W) indicating lesion locations
            
        Returns:
            Dictionary with total loss and component losses
        """
        # Ensure same shape for classification
        if logits.dim() > 1 and logits.size(1) == 1:
            logits = logits.squeeze(1)
        if targets.dim() > 1 and targets.size(1) == 1:
            targets = targets.squeeze(1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            targets_smooth = targets * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            targets_smooth = targets
        
        # Move pos_weight to device
        pos_weight = None
        if self.pos_weight_tensor is not None:
            pos_weight = self.pos_weight_tensor.to(logits.device)
        
        # === Classification Loss ===
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets_smooth, reduction='none', pos_weight=pos_weight
        )
        
        # Focal loss components
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        
        # Alpha weighting
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        
        cls_loss = alpha_t * focal_weight * ce_loss
        cls_loss = cls_loss.mean()
        
        total_loss = self.lambda_cls * cls_loss
        loss_dict = {
            "loss": total_loss,
            "loss_cls": cls_loss.detach(),
        }
        
        # === Mask Loss (if mask is provided) ===
        if mask is not None and self.lambda_mask > 0:
            # Generate pseudo mask predictions from classification logits
            # For positive samples, we expect the mask to be informative
            # For negative samples, mask should be mostly empty
            
            batch_size = logits.size(0)
            mask_target = mask.float()  # (B, Z, H, W)
            
            # Create a simple mask prediction from classification probability
            # Higher classification confidence should correlate with mask presence
            cls_probs = torch.sigmoid(logits)  # (B,)
            
            # Expand classification probability to match mask dimensions
            # Use it as a baseline "attention" map
            cls_probs_expanded = cls_probs.view(batch_size, 1, 1, 1)  # (B, 1, 1, 1)
            
            # Create pseudo mask logits by combining classification confidence with spatial attention
            # For positive cases, use higher baseline activation
            # For negative cases, use lower baseline activation
            mask_shape = mask_target.shape[1:]  # (Z, H, W)
            pseudo_mask_logits = torch.zeros_like(mask_target)  # (B, Z, H, W)
            
            for b in range(batch_size):
                if targets[b] > 0.5:  # Positive case
                    # Use classification confidence as baseline, boosted in mask regions
                    # Clamp more conservatively to avoid numerical issues
                    clamped_prob = cls_probs[b].clamp(min=1e-6, max=1-1e-6)
                    base_logit = torch.logit(clamped_prob)
                    # Additional clamp to prevent extreme values
                    base_logit = torch.clamp(base_logit, min=-10, max=10)
                    pseudo_mask_logits[b] = base_logit.expand_as(pseudo_mask_logits[b])
                    # Boost regions where we expect lesions (could be enhanced with attention)
                    if mask_target[b].sum() > 0:
                        # Add spatial attention based on mask target regions
                        attention_boost = 2.0  # Boost factor for lesion regions
                        boost_values = attention_boost * mask_target[b]
                        # Clamp boost to prevent overflow
                        boost_values = torch.clamp(boost_values, min=-5, max=5)
                        pseudo_mask_logits[b] += boost_values
                else:  # Negative case
                    # Low baseline activation for negative cases
                    base_logit = -2.197  # logit(0.1), pre-computed to avoid numerical issues
                    pseudo_mask_logits[b] = torch.full_like(pseudo_mask_logits[b], base_logit)
            
            # Calculate mask loss based on selected type
            if self.mask_loss_type == "focal":
                mask_loss = self.focal_loss(pseudo_mask_logits, mask_target)
            elif self.mask_loss_type == "dice":
                mask_loss = self.dice_loss(pseudo_mask_logits, mask_target)
            elif self.mask_loss_type == "bce":
                mask_loss = F.binary_cross_entropy_with_logits(
                    pseudo_mask_logits, mask_target, reduction='mean'
                )
            else:
                raise ValueError(f"Unknown mask_loss_type: {self.mask_loss_type}")
            
            total_loss = total_loss + self.lambda_mask * mask_loss
            loss_dict["loss"] = total_loss
            loss_dict["loss_mask"] = mask_loss.detach()
        else:
            loss_dict["loss_mask"] = torch.tensor(0.0, device=logits.device)
        
        return loss_dict


if __name__ == "__main__":
    print("Testing loss functions...")
    B, S, K, H, W, D = 2, 30, 14, 128, 128, 32
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Test original multitask loss
    print("\n=== Testing MultiTaskLossLogits01 ===")
    out = {
        "output": torch.randn(B, K, device=device),  # logits
        "output_slice": torch.randn(B, S, K, device=device),  # logits
        "mask": torch.randn(B, S, H, W, device=device),  # logits
    }
    batch = {
        "targets": torch.randint(0, 2, (B, K), device=device, dtype=torch.float32),
        "targets_slice": torch.randint(0, 2, (B, S, K), device=device, dtype=torch.float32),
        "mask": torch.randint(0, 2, (B, S, H, W), device=device, dtype=torch.float32),
        "slice_weights": torch.rand(B, S, device=device),
    }

    crit = create_loss("multitask_logits01", lambda_case=1.0, lambda_slice=1.0, lambda_mask=2.0).to(device)
    crit.eval()
    with torch.no_grad():
        losses = crit(out, batch)
    print("Original loss results:", {k: float(v) for k, v in losses.items()})

    # Test 3D combined loss
    print("\n=== Testing Aneurysm3DCombinedLoss ===")
    outputs_3d = {
        "output": torch.randn(B, K, device=device),  # classification logits
        "mask": torch.randn(B, 1, D, H, W, device=device),  # 3D segmentation logits
    }
    targets_3d = {
        "targets": torch.randint(0, 2, (B, K), device=device, dtype=torch.float32),
        "mask": torch.randint(0, 2, (B, 1, D, H, W), device=device, dtype=torch.float32),
    }

    # Test different segmentation loss types
    seg_loss_types = ["dice", "focal", "dice_focal", "bce_dice"]
    for seg_loss_type in seg_loss_types:
        print(f"\nTesting seg_loss_type: {seg_loss_type}")
        crit_3d = create_loss(
            "aneurysm_3d_combined", 
            lambda_cls=1.0, 
            lambda_seg=1.0,
            seg_loss_type=seg_loss_type,
            alpha_seg=0.5
        ).to(device)
        crit_3d.eval()
        with torch.no_grad():
            losses_3d = crit_3d(outputs_3d, targets_3d)
        print(f"  Results: {losses_3d['loss']:.4f} (cls: {losses_3d['loss_cls']:.4f}, seg: {losses_3d['loss_seg']:.4f})")

    # Test patch binary focal loss
    print("\n=== Testing PatchBinaryFocalLoss ===")
    patch_logits = torch.randn(32, 1, device=device)  # batch of patches
    patch_targets = torch.randint(0, 2, (32, 1), device=device, dtype=torch.float32)
    
    patch_loss = create_loss(
        "patch_binary_focal",
        alpha=0.25,
        gamma=2.0,
        pos_weight=2.0,
        label_smoothing=0.1
    ).to(device)
    patch_loss.eval()
    
    with torch.no_grad():
        focal_loss_result = patch_loss(patch_logits, patch_targets)
    print(f"Patch focal loss: {focal_loss_result:.4f}")
    
    # Test individual components
    print("\n=== Testing Individual Components ===")
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    dice = DiceLoss(smooth=1.0)
    
    # Test focal loss
    cls_logits = torch.randn(8, 13, device=device)
    cls_targets = torch.randint(0, 2, (8, 13), device=device, dtype=torch.float32)
    with torch.no_grad():
        focal_result = focal(cls_logits, cls_targets)
    print(f"Focal loss: {focal_result:.4f}")
    
    # Test dice loss
    seg_logits = torch.randn(4, 1, 16, 64, 64, device=device)
    seg_targets = torch.randint(0, 2, (4, 1, 16, 64, 64), device=device, dtype=torch.float32)
    with torch.no_grad():
        dice_result = dice(seg_logits, seg_targets)
    print(f"Dice loss: {dice_result:.4f}")

    print("\n=== All tests completed! ===")
    print(f"Available losses: {list_losses()}")
