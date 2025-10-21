import random
import os
import logging
from typing import Any
from pathlib import Path

from sklearn import metrics
import numpy as np
import torch


def setup_logger(name: str, log_file: str, level: int = logging.INFO) -> logging.Logger:
    """
    同名ロガーを複数回呼んでもハンドラが増殖しない安全設計。
    画面（StreamHandler）とファイル（FileHandler）の両方に出力。
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
        sh = logging.StreamHandler()
        sh.setFormatter(fmt)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(sh)
        logger.addHandler(fh)

    return logger


def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore


def asign_filename_npy(row):
    sid = row["SeriesInstanceUID"]
    # ここを row.name ではなく row['fold'] や row['SomeCol'] に変えてもOK
    save_root = "/mnt/project/brain/aneurysm/tamoto/RSNA2025/data/npy"
    series_path = Path(save_root) / sid / (sid + ".npy")
    return series_path


def asign_filename_png(row):
    sid = row["SeriesInstanceUID"]
    # ここを row.name ではなく row['fold'] や row['SomeCol'] に変えてもOK
    save_root = "/mnt/project/brain/aneurysm/tamoto/RSNA2025/data/png"
    if isinstance(row["sorted_files"], list):
        return [
            x.replace("/kaggle/input/rsna-intracranial-aneurysm-detection/series", save_root).replace(".dcm", ".png")
            for x in row["sorted_files"]
        ]

    return row["sorted_files"]


class RSNAMultiLabelAUC:
    """
    y_true / y_pred_logits: [B, K_total]
      - 最後の列 (-1) が「症例ラベル」
      - 先頭〜(K_total-2) が「部位ラベル」

    case_mode:
      'last'     : 症例AUCは最後の列を使用
      'noisy_or' : 部位列から noisy-or 合成（最後の列は無視）
      'max'      : 部位列から max 合成（最後の列は無視）

    class_names:
      - None              : 返り値に index ベースで 'auc_class_{k}' を格納
      - 長さ K_total-1    : 部位名のみ（症例名なし）
      - 長さ K_total      : 末尾が症例名とみなし、部位名は先頭〜(K_total-2)
    """

    def __init__(self, case_mode: str = "last", eps: float = 1e-6, class_names: list[str] = None, 
                 compute_on_step: bool = True, dist_sync_on_step: bool = False):
        """
        Args:
            case_mode: How to compute case-level predictions ('last', 'noisy_or', 'max')
            eps: Small value to avoid numerical issues
            class_names: List of class names for output
            compute_on_step: Whether to compute metric on each step (for compatibility)
            dist_sync_on_step: Whether to sync across processes on each step (for compatibility)
        """
        assert case_mode in ("last", "noisy_or", "max")
        self.case_mode = case_mode
        self.eps = eps
        self.class_names = class_names
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.reset()

    def reset(self):
        self._y_true = []  # [N, K_total]
        self._y_pred_logit = []  # [N, K_total]

    def update(self, y_true: torch.Tensor, y_pred_logits: torch.Tensor):
        yt = y_true.detach().float().cpu().numpy()
        yp = y_pred_logits.detach().float().cpu().numpy()
        self._y_true.append(yt)
        self._y_pred_logit.append(yp)

    @staticmethod
    def _safe_auc(y_true_1d, y_prob_1d):
        if len(np.unique(y_true_1d)) < 2:
            return np.nan
        return metrics.roc_auc_score(y_true_1d, y_prob_1d)

    @property
    def avg(self):
        if len(self._y_true) == 0:
            return dict(auc_case=np.nan, auc_per_class_mean=np.nan, auc_per_class={}, final_score=np.nan)

        y_true = np.concatenate(self._y_true, axis=0)  # [N, K_total]
        logits = np.concatenate(self._y_pred_logit, axis=0)  # [N, K_total]
        probs = 1.0 / (1.0 + np.exp(-logits))  # sigmoid

        N, K_total = y_true.shape
        if K_total < 1:
            return dict(auc_case=np.nan, auc_per_class_mean=np.nan, auc_per_class={}, final_score=np.nan)

        # ----- 部位名の解決 -----
        region_names = None
        if self.class_names is not None:
            if len(self.class_names) == K_total - 1:
                region_names = list(self.class_names)
            elif len(self.class_names) == K_total:
                region_names = list(self.class_names[:-1])  # 末尾は症例名とみなす

        # ----- 列の分割 -----
        if K_total == 1:
            # 症例のみ
            y_case = y_true[:, -1]
            p_case = probs[:, -1]
            auc_case = self._safe_auc(y_case, p_case)
            return {
                "auc_case": float(auc_case) if not np.isnan(auc_case) else np.nan,
                "auc_per_class_mean": np.nan,
                "auc_per_class": {},
                "final_score": float(auc_case) if not np.isnan(auc_case) else np.nan,
                "num_samples": int(N),
                "num_classes": 0,
                "case_mode": self.case_mode,
            }

        y_cls_true = y_true[:, :-1]  # [N, K]   部位
        p_cls = probs[:, :-1]  # [N, K]
        K = y_cls_true.shape[1]

        # ----- 症例AUC -----
        if self.case_mode == "last":
            y_case = y_true[:, -1]
            p_case = probs[:, -1]
        elif self.case_mode == "noisy_or":
            p_clamped = np.clip(p_cls, self.eps, 1 - self.eps)
            log_prod = np.log1p(-p_clamped).sum(axis=1)
            p_case = 1.0 - np.exp(log_prod)
            y_case = (y_cls_true.sum(axis=1) > 0).astype(np.int32)
        else:  # 'max'
            p_case = p_cls.max(axis=1)
            y_case = (y_cls_true.sum(axis=1) > 0).astype(np.int32)

        auc_case = self._safe_auc(y_case, p_case)

        # ----- 部位ごとのAUC -----
        auc_per_class_list = []
        auc_per_class_map = {}
        for k in range(K):
            auc_k = self._safe_auc(y_cls_true[:, k], p_cls[:, k])
            auc_per_class_list.append(auc_k)
            key = region_names[k] if region_names is not None else f"class_{k}"
            auc_per_class_map[key] = float(auc_k) if not np.isnan(auc_k) else np.nan

        auc_per_class_mean = np.nanmean(auc_per_class_list) if K > 0 else np.nan

        # ----- 最終スコア -----
        final = np.nanmean([auc_case, auc_per_class_mean])

        # 返り値
        out = {
            "auc_case": float(auc_case) if not np.isnan(auc_case) else np.nan,
            "auc_per_class_mean": float(auc_per_class_mean) if not np.isnan(auc_per_class_mean) else np.nan,
            "auc_per_class": auc_per_class_map,  # ★ 部位ごとのAUC（名前付き）
            "final_score": float(final) if not np.isnan(final) else np.nan,
            "num_samples": int(N),
            "num_classes": int(K),
            "case_mode": self.case_mode,
        }
        # 互換: indexベースのキーも残す
        for k, v in enumerate(auc_per_class_list):
            out[f"auc_class_{k}"] = float(v) if not np.isnan(v) else np.nan
        if region_names is not None:
            out["region_names"] = region_names
        return out


class RSNAPatchBinaryAUC:
    """
    パッチデータ用の1クラスbinary分類メトリック
    症例毎のパッチの推論結果を受け取り、その最大値でAUCを計算する
    
    Usage:
        metric = RSNAPatchBinaryAUC()
        # Training loop
        for batch in dataloader:
            outputs = model(batch)
            # outputs: (batch_size, 1) or (batch_size,) の logits/probs
            # targets: (batch_size, 1) or (batch_size,) の binary labels
            # series_ids: (batch_size,) の series ID list
            metric.update(targets, outputs, series_ids)
        
        # Get AUC
        result = metric.avg
        print(f"Case-level AUC: {result['auc']:.4f}")
    """
    
    def __init__(self, compute_on_step: bool = True, dist_sync_on_step: bool = False):
        """
        Args:
            compute_on_step: Whether to compute metric on each step (for compatibility)
            dist_sync_on_step: Whether to sync across processes on each step (for compatibility)
        """
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.reset()
    
    def reset(self):
        """Reset accumulated data"""
        # 症例IDをキーにして、その症例のパッチ予測結果を蓄積
        self._case_predictions = {}  # {series_id: list of predictions}
        self._case_labels = {}       # {series_id: true label}
    
    def update(
        self, 
        y_true: torch.Tensor, 
        y_pred_logits: torch.Tensor, 
        series_ids: list[str]
    ):
        """
        パッチの予測結果を症例別に蓄積
        
        Args:
            y_true: Ground truth labels, shape (batch_size,) or (batch_size, 1)
            y_pred_logits: Prediction logits, shape (batch_size,) or (batch_size, 1)  
            series_ids: List of series IDs, length batch_size
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().float().cpu().numpy()
        if isinstance(y_pred_logits, torch.Tensor):
            y_pred_logits = y_pred_logits.detach().float().cpu().numpy()
        
        # Ensure 1D arrays
        if y_true.ndim > 1 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)
        if y_pred_logits.ndim > 1 and y_pred_logits.shape[1] == 1:
            y_pred_logits = y_pred_logits.squeeze(1)
        
        # Convert logits to probabilities
        y_pred_probs = 1.0 / (1.0 + np.exp(-y_pred_logits))  # sigmoid
        
        # Accumulate per case
        for i, series_id in enumerate(series_ids):
            label = int(y_true[i])
            prob = float(y_pred_probs[i])
            
            if series_id not in self._case_predictions:
                self._case_predictions[series_id] = []
                self._case_labels[series_id] = label
            
            # Add patch prediction
            self._case_predictions[series_id].append(prob)
            
            # Verify label consistency within the same case
            if self._case_labels[series_id] != label:
                # If inconsistent, take positive label (assume positive patches exist in positive cases)
                self._case_labels[series_id] = max(self._case_labels[series_id], label)
    
    def _safe_auc(self, y_true, y_pred):
        """Safe AUC calculation"""
        try:
            if len(np.unique(y_true)) < 2:
                return np.nan
            return metrics.roc_auc_score(y_true, y_pred)
        except Exception:
            return np.nan
    
    @property
    def avg(self):
        """
        症例レベルのAUCを計算
        各症例のパッチ予測の最大値を使用
        """
        if not self._case_predictions:
            return {
                'auc': np.nan,
                'num_cases': 0,
                'num_patches': 0,
                'case_max_probs': [],
                'case_labels': []
            }
        
        # 症例毎の最大予測値と真のラベルを取得
        case_max_probs = []
        case_labels = []
        total_patches = 0
        
        for series_id in self._case_predictions:
            # その症例のパッチ予測の最大値を取る
            max_prob = max(self._case_predictions[series_id])
            label = self._case_labels[series_id]
            
            case_max_probs.append(max_prob)
            case_labels.append(label)
            total_patches += len(self._case_predictions[series_id])
        
        case_max_probs = np.array(case_max_probs)
        case_labels = np.array(case_labels)
        
        # AUC計算
        auc = self._safe_auc(case_labels, case_max_probs)
        
        return {
            'auc': float(auc) if not np.isnan(auc) else np.nan,
            'num_cases': len(case_labels),
            'num_patches': total_patches,
            'case_max_probs': case_max_probs.tolist(),
            'case_labels': case_labels.tolist(),
            'pos_cases': int(np.sum(case_labels)),
            'neg_cases': int(len(case_labels) - np.sum(case_labels))
        }
    
    def get_case_predictions(self):
        """症例毎の詳細な予測結果を取得"""
        results = {}
        for series_id in self._case_predictions:
            results[series_id] = {
                'patches': self._case_predictions[series_id],
                'label': self._case_labels[series_id],
                'max_prob': max(self._case_predictions[series_id]),
                'mean_prob': np.mean(self._case_predictions[series_id]),
                'num_patches': len(self._case_predictions[series_id])
            }
        return results


class SimplePatchAUC:
    """
    - パッチレベルAUC（全パッチをそのまま評価）
    - 3DマスクDice（画像ごとに計算して平均）
    """
    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        smooth: float = 1.0,
        eps: float = 1e-6,
        dice_soft: bool = True,       # True: soft-Dice（推奨）, False: hard-Dice（thresholdで2値化）
        dice_threshold: float = 0.5,  # hard-Dice用の閾値
    ):
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.smooth = smooth
        self.eps = eps
        self.dice_soft = dice_soft
        self.dice_threshold = dice_threshold
        self.reset()

    def reset(self):
        """Reset accumulated data"""
        # AUC用
        self._predictions = []
        self._labels = []
        # Dice用（画像ごと）
        self._dice_values = []

    def update(
        self,
        batch: dict[torch.Tensor],
        outputs:dict[torch.Tensor]
    ):
        """
        パッチAUC: y_true / y_pred_logits を蓄積（従来通り）
        マスクDice: y_true_mask / y_pred_mask_logits が渡された場合のみ、その場で画像ごとにDice計算し値のみ蓄積

        """
        y_true = batch['target_aneurysm']
        y_pred_logits = outputs["output"]
        y_true_mask = batch["mask"]
        y_pred_mask_logits = outputs["mask"]
        # ===== AUCパート（従来通り） =====
        if isinstance(y_true, torch.Tensor):
            y_true_np = y_true.detach().float().cpu().numpy()
        else:
            y_true_np = y_true
        if isinstance(y_pred_logits, torch.Tensor):
            y_pred_logits_np = y_pred_logits.detach().float().cpu().numpy()
        else:
            y_pred_logits_np = y_pred_logits

        if y_true_np.ndim > 1 and y_true_np.shape[1] == 1:
            y_true_np = y_true_np.squeeze(1)
        if y_pred_logits_np.ndim > 1 and y_pred_logits_np.shape[1] == 1:
            y_pred_logits_np = y_pred_logits_np.squeeze(1)

        # ロジット→確率（数値安定化付）
        y_pred_probs_np = 1.0 / (1.0 + np.exp(-np.clip(y_pred_logits_np, -500, 500)))

        self._predictions.extend(y_pred_probs_np.tolist())
        self._labels.extend(y_true_np.tolist())

        # ===== Diceパート（メモリ節約：ここで画像ごとにDiceのみ計算） =====
        if (y_true_mask is not None) and (y_pred_mask_logits is not None):
            self._update_dice_per_image(y_true_mask, y_pred_mask_logits)

    @torch.no_grad()
    def _update_dice_per_image(self, y_true_mask: torch.Tensor, y_pred_mask_logits: torch.Tensor):
        """
        画像ごと（=バッチ内の各サンプルごと）にDiceを計算して self._dice_values にappend。
        - 入力は 3D: (B, [C|1|なし], D, H, W)
        - マルチチャネルはクラス平均（macro）Diceをその画像のDiceとして扱う
        - soft/hardはコンストラクタ設定による
        """
        # 形状合わせ
        # 目標: (B, C, D, H, W)
        if y_true_mask.dim() == 4:  # (B, D, H, W)
            y_true_mask = y_true_mask.unsqueeze(1)
        if y_pred_mask_logits.dim() == 4:
            y_pred_mask_logits = y_pred_mask_logits.unsqueeze(1)

        assert y_true_mask.shape[:2] == y_pred_mask_logits.shape[:2], \
            f"Shape mismatch: {y_true_mask.shape} vs {y_pred_mask_logits.shape}"

        # 数値安定化 + 確率化
        logits = torch.clamp(y_pred_mask_logits, min=-100, max=100)
        probs = torch.sigmoid(logits)
        probs = torch.clamp(probs, min=self.eps, max=1.0 - self.eps)

        targets = y_true_mask.float()

        B, C = probs.shape[:2]

        if not self.dice_soft:
            # hard-Dice用に2値化
            probs = (probs >= self.dice_threshold).float()

        # 画像ごとに計算（マルチチャネルはクラス平均）
        # メモリ節約のため、flatten→sumのみ使う
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)

        tp = (probs_flat * targets_flat).sum(dim=2)              # (B, C)
        p_sum = probs_flat.sum(dim=2)                             # (B, C)
        t_sum = targets_flat.sum(dim=2)                           # (B, C)

        dice_c = (2.0 * tp + self.smooth) / (p_sum + t_sum + self.smooth + self.eps)  # (B, C)
        dice_c = dice_c.clamp_(0.0, 1.0)

        # 画像ごとにクラス平均
        dice_per_image = dice_c.mean(dim=1)  # (B,)

        # NaNガード
        dice_per_image = torch.where(torch.isnan(dice_per_image),
                                     torch.zeros_like(dice_per_image),
                                     dice_per_image)

        self._dice_values.extend(dice_per_image.cpu().tolist())

    def _safe_auc(self, y_true, y_pred):
        """Safe AUC calculation"""
        try:
            if len(np.unique(y_true)) < 2:
                print("Warning: Only one class present in labels")
                return np.nan
            return metrics.roc_auc_score(y_true, y_pred)
        except Exception as e:
            print(f"Warning: AUC calculation failed: {e}")
            return np.nan

    @property
    def avg(self):
        """
        - Patch AUC
        - Dice（画像平均）
        を返す
        """
        # AUC
        if self._predictions:
            predictions = np.array(self._predictions)
            labels = np.array(self._labels)
            auc = self._safe_auc(labels, predictions)
            pos_indices = labels == 1
            neg_indices = labels == 0
            auc_part = {
                'auc': float(auc) if not np.isnan(auc) else np.nan,
                'num_patches': int(len(labels)),
                'num_cases': int(len(labels)),  # 互換性のため
                'mean_pred': float(np.mean(predictions)),
                'mean_label': float(np.mean(labels)),
                'num_pos': int(np.sum(pos_indices)),
                'num_neg': int(np.sum(neg_indices)),
            }
        else:
            auc_part = {
                'auc': np.nan,
                'num_patches': 0,
                'num_cases': 0,
                'mean_pred': np.nan,
                'mean_label': np.nan,
                'num_pos': 0,
                'num_neg': 0,
            }

        # Dice
        if self._dice_values:
            dice_vals = np.array(self._dice_values, dtype=float)
            dice_part = {
                'dice': float(np.mean(dice_vals)),
                'dice_std': float(np.std(dice_vals)),
                'num_images_dice': int(len(dice_vals)),
            }
        else:
            dice_part = {
                'dice': np.nan,
                'dice_std': np.nan,
                'num_images_dice': 0,
            }

        return {**auc_part, **dice_part}
    
class MultiPatchAUC:
    """
    - パッチレベル AUC（multilabel）
      * クラス別 AUC
      * macro / micro / weighted
      * 最終スコア = 0.5*Aneurysm AUC + 0.5*部位13種AUC平均
    - 3Dマスク Dice（画像ごとに計算して平均）
    """

    target_sites = {
        "Left Infraclinoid Internal Carotid Artery": 0,
        "Right Infraclinoid Internal Carotid Artery": 1,
        "Left Supraclinoid Internal Carotid Artery": 2,
        "Right Supraclinoid Internal Carotid Artery": 3,
        "Left Middle Cerebral Artery": 4,
        "Right Middle Cerebral Artery": 5,
        "Anterior Communicating Artery": 6,
        "Left Anterior Cerebral Artery": 7,
        "Right Anterior Cerebral Artery": 8,
        "Left Posterior Communicating Artery": 9,
        "Right Posterior Communicating Artery": 10,
        "Basilar Tip": 11,
        "Other Posterior Circulation": 12,
        "Aneurysm Present": 13,
    }

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        smooth: float = 1.0,
        eps: float = 1e-6,
        dice_soft: bool = True,
        dice_threshold: float = 0.5,
    ):
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.smooth = smooth
        self.eps = eps
        self.dice_soft = dice_soft
        self.dice_threshold = dice_threshold
        self.reset()

    def reset(self):
        self._predictions_batches = []  # list of (N,C)
        self._labels_batches = []       # list of (N,C)
        self._dice_values = []

    def update(
        self,
        batch: dict[torch.Tensor],
        outputs:dict[torch.Tensor]
    ):
        y_true = batch['target_site']
        y_pred_logits = outputs["output"]
        self.use_mask = False
        if ("mask" in batch)&('mask' in outputs):
            y_true_mask = batch["mask"]
            y_pred_mask_logits = outputs["mask"]
            self.use_mask = True
        # ---- multilabel AUC gather ----
        y_t = y_true.detach().float().cpu().numpy() if isinstance(y_true, torch.Tensor) else y_true
        y_l = y_pred_logits.detach().float().cpu().numpy() if isinstance(y_pred_logits, torch.Tensor) else y_pred_logits

        y_t = np.asarray(y_t)
        y_l = np.asarray(y_l)
        if y_t.ndim == 1: y_t = y_t[None, :]   # (C,) -> (1,C)
        if y_l.ndim == 1: y_l = y_l[None, :]   # (C,) -> (1,C)

        y_p = 1.0 / (1.0 + np.exp(-np.clip(y_l, -500, 500)))  # logits -> probs
        self._predictions_batches.append(y_p.astype(np.float32))
        self._labels_batches.append(y_t.astype(np.float32))
        # ---- Dice per image ----
        if self.use_mask:
            if (y_true_mask is not None) and (y_pred_mask_logits is not None):
                self._update_dice_per_image(y_true_mask, y_pred_mask_logits)

    @torch.no_grad()
    def _update_dice_per_image(self, y_true_mask: torch.Tensor, y_pred_mask_logits: torch.Tensor):
        if y_true_mask.dim() == 4: y_true_mask = y_true_mask.unsqueeze(1)
        if y_pred_mask_logits.dim() == 4: y_pred_mask_logits = y_pred_mask_logits.unsqueeze(1)
        assert y_true_mask.shape[:2] == y_pred_mask_logits.shape[:2], \
            f"Shape mismatch: {y_true_mask.shape} vs {y_pred_mask_logits.shape}"

        logits = torch.clamp(y_pred_mask_logits, min=-100, max=100)
        probs = torch.sigmoid(logits).clamp_(self.eps, 1.0 - self.eps)
        targets = y_true_mask.float()
        if not self.dice_soft:
            probs = (probs >= self.dice_threshold).float()

        B, C = probs.shape[:2]
        probs_flat = probs.view(B, C, -1)
        targets_flat = targets.view(B, C, -1)
        tp = (probs_flat * targets_flat).sum(dim=2)
        p_sum = probs_flat.sum(dim=2)
        t_sum = targets_flat.sum(dim=2)
        dice_c = (2.0 * tp + self.smooth) / (p_sum + t_sum + self.smooth + self.eps)
        dice_c = dice_c.clamp_(0.0, 1.0)
        dice_per_image = dice_c.mean(dim=1)
        dice_per_image = torch.where(torch.isnan(dice_per_image),
                                     torch.zeros_like(dice_per_image),
                                     dice_per_image)
        self._dice_values.extend(dice_per_image.cpu().tolist())

    # ---------- AUC helpers ----------
    def _per_class_auc(self, y_true: np.ndarray, y_prob: np.ndarray):
        N, C = y_true.shape
        aucs = np.full(C, np.nan, dtype=float)
        valid = np.zeros(C, dtype=bool)
        for c in range(C):
            y = y_true[:, c]
            p = y_prob[:, c]
            if len(np.unique(y)) < 2:
                continue
            try:
                aucs[c] = metrics.roc_auc_score(y, p)
                valid[c] = True
            except Exception:
                pass
        return aucs, valid

    def _micro_auc(self, y_true: np.ndarray, y_prob: np.ndarray):
        try:
            return float(metrics.roc_auc_score(y_true, y_prob, average="micro"))
        except Exception:
            y = y_true.reshape(-1); p = y_prob.reshape(-1)
            if len(np.unique(y)) < 2: return np.nan
            try:
                return float(metrics.roc_auc_score(y, p))
            except Exception:
                return np.nan

    def _weighted_auc(self, per_class_auc: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray):
        pos_counts = y_true.sum(axis=0).astype(float)
        w = np.where(valid_mask, pos_counts, 0.0)
        if w.sum() <= 0: return np.nan
        return float(np.nansum(per_class_auc * w) / (w.sum() + 1e-12))

    @staticmethod
    def _nan_weighted_mean(values, weights):
        vals = np.array(values, dtype=float)
        w = np.array(weights, dtype=float)
        m = ~np.isnan(vals)
        if not m.any():
            return np.nan
        w_eff = w[m]
        w_eff = w_eff / (w_eff.sum() + 1e-12)
        return float(np.sum(vals[m] * w_eff))

    @property
    def avg(self):
        # ===== AUC 集計 =====
        if self._predictions_batches:
            y_prob = np.concatenate(self._predictions_batches, axis=0)  # (N,C)
            y_true = np.concatenate(self._labels_batches, axis=0)       # (N,C)

            per_class_auc, valid_mask = self._per_class_auc(y_true, y_prob)

            auc_macro = float(np.nanmean(per_class_auc[valid_mask])) if valid_mask.any() else np.nan
            auc_micro = self._micro_auc(y_true, y_prob)
            auc_weighted = self._weighted_auc(per_class_auc, y_true, valid_mask)

            # --- Aneurysm / Sites ---
            aneurysm_idx = self.target_sites.get("Aneurysm Present", None)
            aneurysm_auc = float(per_class_auc[aneurysm_idx]) if (aneurysm_idx is not None and not np.isnan(per_class_auc[aneurysm_idx])) else np.nan
            site_indices = [i for k, i in self.target_sites.items() if k != "Aneurysm Present"]
            site_aucs = per_class_auc[site_indices] if len(site_indices) > 0 else np.array([], dtype=float)
            site_mean_auc = float(np.nanmean(site_aucs)) if np.isfinite(site_aucs).any() else np.nan

            # 最終スコア
            final_score = self._nan_weighted_mean([aneurysm_auc, site_mean_auc], [0.5, 0.5])

            # --- per-class dict（名前付き） ---
            inv_map = {v: k for k, v in self.target_sites.items()}
            C = y_true.shape[1]
            per_class_dict = {
                inv_map.get(c, str(c)): (None if np.isnan(per_class_auc[c]) else float(per_class_auc[c]))
                for c in range(C)
            }

            # --- 依頼の出力: Aneurysm含む部位ごとのAUC ---
            # 並び順は target_sites の index 昇順で固定
            site_order = sorted(self.target_sites.items(), key=lambda kv: kv[1])  # [(name, idx), ...]
            site_aucs_including_aneurysm = {
                name: (None if np.isnan(per_class_auc[idx]) else float(per_class_auc[idx]))
                for name, idx in site_order
            }
            # 参考: aneurysm を除いた13部位のみ
            site_aucs_excluding_aneurysm = {
                name: (None if np.isnan(per_class_auc[idx]) else float(per_class_auc[idx]))
                for name, idx in site_order if name != "Aneurysm Present"
            }

            mean_pred = float(np.mean(y_prob)); mean_label = float(np.mean(y_true))
            num_elements = int(y_true.size); pos_total = int(y_true.sum()); neg_total = num_elements - pos_total

            auc_part = {
                "auc_macro": auc_macro,
                "auc_micro": float(auc_micro) if not np.isnan(auc_micro) else np.nan,
                "auc_weighted": float(auc_weighted) if not np.isnan(auc_weighted) else np.nan,
                "per_class_auc": per_class_dict,  # 全クラス
                # 最終スコア関連
                "aneurysm_auc": (None if np.isnan(aneurysm_auc) else aneurysm_auc),
                "site_mean_auc": (None if np.isnan(site_mean_auc) else site_mean_auc),
                "final_score": (None if np.isnan(final_score) else final_score),
                # サマリ
                "mean_pred_overall": mean_pred,
                "mean_label_overall": mean_label,
                }
        else:
            auc_part = {
                "auc_macro": np.nan,
                "auc_micro": np.nan,
                "auc_weighted": np.nan,
                "per_class_auc": {},
                "site_aucs_including_aneurysm": {},
                "site_aucs_excluding_aneurysm": {},
                "aneurysm_auc": None,
                "site_mean_auc": None,
                "final_score": None,
                "num_samples": 0,
                "num_classes": 0,
                "num_valid_classes": 0,
                "mean_pred_overall": np.nan,
                "pos_total": 0,
                "neg_total": 0,
            }

        # ===== Dice 集計 =====
        if self._dice_values:
            dice_vals = np.array(self._dice_values, dtype=float)
            dice_part = {
                "dice": float(np.mean(dice_vals)),
                "dice_std": float(np.std(dice_vals)),
                "num_images_dice": int(len(dice_vals)),
            }
        else:
            dice_part = {"dice": np.nan, "dice_std": np.nan, "num_images_dice": 0}

        return {**auc_part, **dice_part}
    
import numpy as np
from sklearn import metrics
import torch
from typing import Dict, List, Optional, Sequence


class MultiROIAUC:
    """
    ROIレベルの multilabel AUC 指標 +（外部で集計する）Dice の器。

    **方針（簡潔版）**
    - Top-K は常に **複数K（リスト）**で評価します。単一Kパスは削除。
    - `topk_list` または `topk_ratio_list` のいずれか（または両方）を必ず指定してください。
    - 推論で得られない情報（GTマスク等）は使用しません。

    **集約**: B×N×C の logits（N = Dm*Hm*Wm）→ B×C の確率
      - `aggregate_op="topk_mean"`（推奨）: N軸で降順ソート→累積和→先頭k平均
      - `aggregate_op="mean"` / `"max"` は非推奨（ただし互換のため残すが、結果は K 非依存）

    **Top-K の基準**
      - `topk_basis="per_class"`（デフォルト）: クラスごとに独立して上位kセルを選ぶ
      - `topk_basis="anchor"` : `anchor_index`（例: 13=Aneurysm）の順位を全クラスに共通適用

    **返却（avg プロパティ）**
      - `by_topk[K]` : 各Kでのサマリ辞書
      - `representative_topk` : 代表K（昇順で最初のK）
      - `rep_*` : 代表Kの各指標（final_score, aneurysm_auc, など）
      - `best_topk` : `final_score` が最大のK
      - `best_*` : ベストKの各指標（site別AUCやAneurysm AUC も含む）
    """

    target_sites = {
        "Left Infraclinoid Internal Carotid Artery": 0,
        "Right Infraclinoid Internal Carotid Artery": 1,
        "Left Supraclinoid Internal Carotid Artery": 2,
        "Right Supraclinoid Internal Carotid Artery": 3,
        "Left Middle Cerebral Artery": 4,
        "Right Middle Cerebral Artery": 5,
        "Anterior Communicating Artery": 6,
        "Left Anterior Cerebral Artery": 7,
        "Right Anterior Cerebral Artery": 8,
        "Left Posterior Communicating Artery": 9,
        "Right Posterior Communicating Artery": 10,
        "Basilar Tip": 11,
        "Other Posterior Circulation": 12,
        "Aneurysm Present": 13,
    }

    def __init__(
        self,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        smooth: float = 1.0,
        eps: float = 1e-6,
        *,
        aggregate_op: str = "topk_mean",   # "topk_mean" | "max" | "mean"（topk_mean推奨）
        topk_list: Optional[Sequence[int]] = None,
        topk_ratio_list: Optional[Sequence[float]] = None,
        # Top-K の基準
        topk_basis: str = "per_class",     # "per_class" | "anchor"
        anchor_index: int = 13,             # topk_basis=="anchor" のときのアンカー（例: Aneurysm=13）
        use_sigmoid: bool = True,
    ):
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.smooth = smooth
        self.eps = eps

        self.aggregate_op = aggregate_op
        self.topk_list = list(topk_list) if topk_list is not None else None
        self.topk_ratio_list = list(topk_ratio_list) if topk_ratio_list is not None else None
        self.topk_basis = str(topk_basis)
        self.anchor_index = int(anchor_index)
        self.use_sigmoid = bool(use_sigmoid)

        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        # 複数K用: K → list of (B,C)
        self._predictions_batches_per_k: Dict[int, List[np.ndarray]] = {}
        self._labels_batches: List[np.ndarray] = []       # (B,C)
        self._dice_values: List[float] = []
        self.active_topk: Optional[int] = None

    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    def _decide_k_list(self, N: int) -> List[int]:
        Ks: List[int] = []
        if self.topk_ratio_list:
            Ks.extend([int(max(1, round(N * float(r)))) for r in self.topk_ratio_list])
        if self.topk_list:
            Ks.extend([int(k) for k in self.topk_list])
        Ks = [k for k in Ks if 1 <= k <= N]
        if not Ks:
            raise ValueError("MultiROIAUC: Please provide non-empty topk_list and/or topk_ratio_list.")
        Ks = sorted(sorted(set(Ks)))
        return Ks

    # ------------------------------------------------------------------
    def _aggregate_logits_topk_multi(
        self,
        logits_bnc: np.ndarray,                 # (B,N,C)
        Ks: Sequence[int],
    ) -> Dict[int, np.ndarray]:                 # K -> (B,C)
        """Top-K平均（複数K）。**maskは使わない**。
        - per_class: クラス毎に独立ソート
        - anchor   : anchor_index の並びを全クラスに共通適用
        """
        B, N, C = logits_bnc.shape
        probs = self._sigmoid_np(logits_bnc) if self.use_sigmoid else logits_bnc
        out_per_k: Dict[int, np.ndarray] = {int(k): np.zeros((B, C), dtype=np.float32) for k in Ks}

        if self.topk_basis == "anchor":
            if not (0 <= self.anchor_index < C):
                raise ValueError(f"anchor_index out of range: {self.anchor_index} for C={C}")
            # アンカー確率で N 軸並び替え（B,N）
            anchor = probs[:, :, self.anchor_index]
            order = np.argsort(-anchor, axis=1)              # (B,N) 降順 index
            # 全クラスに同じ order を適用
            sorted_all = np.take_along_axis(probs, order[:, :, None], axis=1)  # (B,N,C)
            cs = np.cumsum(sorted_all, axis=1)               # (B,N,C)
            for k in Ks:
                out_per_k[int(k)] = cs[:, k - 1, :] / float(k)
            return out_per_k

        # per_class: クラス毎に独立
        xs = np.sort(probs, axis=1)[:, ::-1, :]  # (B,N,C) 降順
        cs = np.cumsum(xs, axis=1)               # (B,N,C)
        for k in Ks:
            out_per_k[int(k)] = cs[:, k - 1, :] / float(k)
        return out_per_k

    # ------------------------------------------------------------------
    def update(self, batch: dict, outputs: dict):
        """
        batch['target_site'] : (B, C)
        outputs['output']    : (B, N, C) logits     # N = Dm*Hm*Wm
        ※ GTの mask_map は使用しません（無視されます）。
        """
        y_true = batch['target_site']                      # (B,C)
        logits_bnc = outputs['output']                     # (B,N,C)

        # to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().float().cpu().numpy()
        else:
            y_true = np.asarray(y_true, dtype=np.float32)
        if isinstance(logits_bnc, torch.Tensor):
            logits_bnc = logits_bnc.detach().float().cpu().numpy()
        else:
            logits_bnc = np.asarray(logits_bnc, dtype=np.float32)

        B, N, C = logits_bnc.shape
        Ks = self._decide_k_list(N)
        self.active_topk = Ks[0]

        out_per_k = self._aggregate_logits_topk_multi(logits_bnc, Ks)
        for k, y_prob in out_per_k.items():
            self._predictions_batches_per_k.setdefault(int(k), []).append(y_prob.astype(np.float32))

        self._labels_batches.append(y_true.astype(np.float32))

    # ---------- AUC helpers ----------
    def _per_class_auc(self, y_true: np.ndarray, y_prob: np.ndarray):
        N, C = y_true.shape
        aucs = np.full(C, np.nan, dtype=float)
        valid = np.zeros(C, dtype=bool)
        for c in range(C):
            y = y_true[:, c]
            p = y_prob[:, c]
            if len(np.unique(y)) < 2:
                continue
            try:
                aucs[c] = metrics.roc_auc_score(y, p)
                valid[c] = True
            except Exception:
                pass
        return aucs, valid

    def _micro_auc(self, y_true: np.ndarray, y_prob: np.ndarray):
        try:
            return float(metrics.roc_auc_score(y_true, y_prob, average="micro"))
        except Exception:
            y = y_true.reshape(-1); p = y_prob.reshape(-1)
            if len(np.unique(y)) < 2: return np.nan
            try:
                return float(metrics.roc_auc_score(y, p))
            except Exception:
                return np.nan

    def _weighted_auc(self, per_class_auc: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray):
        pos_counts = y_true.sum(axis=0).astype(float)
        w = np.where(valid_mask, pos_counts, 0.0)
        if w.sum() <= 0: return np.nan
        return float(np.nansum(per_class_auc * w) / (w.sum() + 1e-12))

    @staticmethod
    def _nan_weighted_mean(values, weights):
        vals = np.array(values, dtype=float)
        w = np.array(weights, dtype=float)
        m = ~np.isnan(vals)
        if not m.any():
            return np.nan
        w_eff = w[m]
        w_eff = w_eff / (w_eff.sum() + 1e-12)
        return float(np.sum(vals[m] * w_eff))

    # ------------------------------------------------------------------
    def _summarize_one(self, y_true: np.ndarray, y_prob: np.ndarray):
        per_class_auc, valid_mask = self._per_class_auc(y_true, y_prob)
        auc_macro = float(np.nanmean(per_class_auc[valid_mask])) if valid_mask.any() else np.nan
        auc_micro = self._micro_auc(y_true, y_prob)
        auc_weighted = self._weighted_auc(per_class_auc, y_true, valid_mask)
        aneurysm_idx = self.target_sites.get("Aneurysm Present", None)
        aneurysm_auc = float(per_class_auc[aneurysm_idx]) if (aneurysm_idx is not None and not np.isnan(per_class_auc[aneurysm_idx])) else np.nan
        site_indices = [i for k, i in self.target_sites.items() if k != "Aneurysm Present"]
        site_aucs = per_class_auc[site_indices] if len(site_indices) > 0 else np.array([], dtype=float)
        site_mean_auc = float(np.nanmean(site_aucs)) if np.isfinite(site_aucs).any() else np.nan
        final_score = self._nan_weighted_mean([aneurysm_auc, site_mean_auc], [0.5, 0.5])
        inv_map = {v: k for k, v in self.target_sites.items()}
        C = y_true.shape[1]
        per_class_dict = {inv_map.get(c, str(c)): (None if np.isnan(per_class_auc[c]) else float(per_class_auc[c])) for c in range(C)}
        site_order = sorted(self.target_sites.items(), key=lambda kv: kv[1])
        site_aucs_including_aneurysm = {name: (None if np.isnan(per_class_auc[idx]) else float(per_class_auc[idx])) for name, idx in site_order}
        site_aucs_excluding_aneurysm = {name: (None if np.isnan(per_class_auc[idx]) else float(per_class_auc[idx])) for name, idx in site_order if name != "Aneurysm Present"}
        mean_pred = float(np.mean(y_prob)); mean_label = float(np.mean(y_true))
        num_elements = int(y_true.size); pos_total = int(y_true.sum()); neg_total = num_elements - pos_total
        return {
            "auc_macro": auc_macro,
            "auc_micro": float(auc_micro) if not np.isnan(auc_micro) else np.nan,
            "auc_weighted": float(auc_weighted) if not np.isnan(auc_weighted) else np.nan,
            "per_class_auc": per_class_dict,
            "site_aucs_including_aneurysm": site_aucs_including_aneurysm,
            "site_aucs_excluding_aneurysm": site_aucs_excluding_aneurysm,
            "aneurysm_auc": (None if np.isnan(aneurysm_auc) else aneurysm_auc),
            "site_mean_auc": (None if np.isnan(site_mean_auc) else site_mean_auc),
            "final_score": (None if np.isnan(final_score) else final_score),
            "num_samples": int(y_true.shape[0]),
            "num_classes": int(y_true.shape[1]),
            "num_valid_classes": int(valid_mask.sum()),
            "mean_pred_overall": mean_pred,
            "pos_total": pos_total,
            "neg_total": neg_total,
        }

    @property
    def avg(self):
        # ラベル結合
        if self._labels_batches:
            y_true = np.concatenate(self._labels_batches, axis=0)  # (B,C)
        else:
            return {
                "by_topk": {},
                "representative_topk": None,
                "best_topk": None,
                "dice": np.nan,
                "dice_std": np.nan,
                "num_images_dice": 0,
            }

        
        # 複数K（必須）
        if not self._predictions_batches_per_k:
            return {"by_topk": {}, "representative_topk": None, "best_topk": None}

        by_topk: Dict[int, dict] = {}
        best_k = None
        best_score = -np.inf
        for k, chunks in sorted(self._predictions_batches_per_k.items()):
            y_prob = np.concatenate(chunks, axis=0)
            summ = self._summarize_one(y_true, y_prob)
            by_topk[int(k)] = summ
            sc = summ.get("final_score")
            if sc is not None and np.isfinite(sc) and sc > best_score:
                best_score = sc
                best_k = int(k)

        # 代表K（first）とベストK
        rep_k = min(by_topk.keys())
        rep = by_topk[rep_k]
        best = by_topk[best_k] if (best_k in by_topk) else rep

        return {
            # 代表Kの結果（rep_*）
            **{f"rep_{k}": v for k, v in rep.items()},
            "representative_topk": int(rep_k),
            # ベストKの結果（best_*）
            "best_topk": int(best_k) if best_k is not None else int(rep_k),
            "best_final_score": best.get("final_score"),
            "best_aneurysm_auc": best.get("aneurysm_auc"),
            "best_site_mean_auc": best.get("site_mean_auc"),
            "best_auc_macro": best.get("auc_macro"),
            "best_auc_micro": best.get("auc_micro"),
            "best_auc_weighted": best.get("auc_weighted"),
            "best_per_class_auc": best.get("per_class_auc", {}),
            "best_site_aucs_including_aneurysm": best.get("site_aucs_including_aneurysm", {}),
            "best_site_aucs_excluding_aneurysm": best.get("site_aucs_excluding_aneurysm", {}),
            # 全Kの辞書
            "by_topk": by_topk,
            }


class MultiROIAUCVolume:
    """
    Volumeレベル (B, C) の予測から multilabel AUC を算出する簡易クラス。

    入力:
      batch['target_site']: (B, C)  0/1
      outputs['volume_output'] or outputs['output']: (B, C)  ロジット or 確率

    オプション:
      use_sigmoid=True のとき、入力がロジット想定で sigmoid を適用。
      False のとき、そのまま確率として扱う。

    返却 (avg):
      - auc_macro, auc_micro, auc_weighted
      - aneurysm_auc, site_mean_auc, final_score (= 平均: aneurysm/site)
      - per_class_auc (dict), site_aucs_* (dict)
    """

    target_sites = {
        "Left Infraclinoid Internal Carotid Artery": 0,
        "Right Infraclinoid Internal Carotid Artery": 1,
        "Left Supraclinoid Internal Carotid Artery": 2,
        "Right Supraclinoid Internal Carotid Artery": 3,
        "Left Middle Cerebral Artery": 4,
        "Right Middle Cerebral Artery": 5,
        "Anterior Communicating Artery": 6,
        "Left Anterior Cerebral Artery": 7,
        "Right Anterior Cerebral Artery": 8,
        "Left Posterior Communicating Artery": 9,
        "Right Posterior Communicating Artery": 10,
        "Basilar Tip": 11,
        "Other Posterior Circulation": 12,
        "Aneurysm Present": 13,
    }

    def __init__(self, use_sigmoid: bool = True):
        self.use_sigmoid = bool(use_sigmoid)
        self.reset()

    # -------------------------
    def reset(self):
        self._labels_batches = []       # list[(B, C)]
        self._pred_batches = []         # list[(B, C)]

    # -------------------------
    @staticmethod
    def _sigmoid_np(x: np.ndarray) -> np.ndarray:
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    # -------------------------
    def update(self, batch: Dict, outputs: Dict):
        """
        batch['target_site'] : (B, C) 0/1
        outputs['volume_output'] or outputs['output'] : (B, C)  logits or probs
        """
        y_true = batch["target_site"]
        if "volume_output" in outputs:
            y_pred = outputs["volume_output"]
        elif "output" in outputs:
            y_pred = outputs["output"]
        else:
            raise ValueError("outputs must contain 'volume_output' or 'output' (both are (B, C)).")

        # -> numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().float().cpu().numpy()
        else:
            y_true = np.asarray(y_true, dtype=np.float32)

        if isinstance(y_pred, torch.Tensor):
            y_pred = y_pred.detach().float().cpu().numpy()
        else:
            y_pred = np.asarray(y_pred, dtype=np.float32)

        # logits -> probs
        if self.use_sigmoid:
            y_pred = self._sigmoid_np(y_pred)

        self._labels_batches.append(y_true.astype(np.float32))
        self._pred_batches.append(y_pred.astype(np.float32))

    # -------------------------
    def _per_class_auc(self, y_true: np.ndarray, y_prob: np.ndarray):
        N, C = y_true.shape
        aucs = np.full(C, np.nan, dtype=float)
        valid = np.zeros(C, dtype=bool)
        for c in range(C):
            y = y_true[:, c]
            p = y_prob[:, c]
            if len(np.unique(y)) < 2:
                continue
            try:
                aucs[c] = metrics.roc_auc_score(y, p)
                valid[c] = True
            except Exception:
                pass
        return aucs, valid

    def _micro_auc(self, y_true: np.ndarray, y_prob: np.ndarray):
        try:
            return float(metrics.roc_auc_score(y_true, y_prob, average="micro"))
        except Exception:
            y = y_true.reshape(-1); p = y_prob.reshape(-1)
            if len(np.unique(y)) < 2:
                return np.nan
            try:
                return float(metrics.roc_auc_score(y, p))
            except Exception:
                return np.nan

    def _weighted_auc(self, per_class_auc: np.ndarray, y_true: np.ndarray, valid_mask: np.ndarray):
        pos_counts = y_true.sum(axis=0).astype(float)
        w = np.where(valid_mask, pos_counts, 0.0)
        if w.sum() <= 0:
            return np.nan
        return float(np.nansum(per_class_auc * w) / (w.sum() + 1e-12))

    @staticmethod
    def _nan_weighted_mean(values, weights):
        vals = np.array(values, dtype=float)
        w = np.array(weights, dtype=float)
        m = ~np.isnan(vals)
        if not m.any():
            return np.nan
        w_eff = w[m]
        w_eff = w_eff / (w_eff.sum() + 1e-12)
        return float(np.sum(vals[m] * w_eff))

    # -------------------------
    @property
    def avg(self):
        if not self._labels_batches:
            return {}

        y_true = np.concatenate(self._labels_batches, axis=0)  # (B,C)
        y_prob = np.concatenate(self._pred_batches, axis=0)    # (B,C)

        per_class_auc, valid_mask = self._per_class_auc(y_true, y_prob)
        auc_macro = float(np.nanmean(per_class_auc[valid_mask])) if valid_mask.any() else np.nan
        auc_micro = self._micro_auc(y_true, y_prob)
        auc_weighted = self._weighted_auc(per_class_auc, y_true, valid_mask)

        aneurysm_idx = self.target_sites["Aneurysm Present"]
        aneurysm_auc = float(per_class_auc[aneurysm_idx]) if not np.isnan(per_class_auc[aneurysm_idx]) else np.nan

        site_indices = [i for k, i in self.target_sites.items() if k != "Aneurysm Present"]
        site_aucs = per_class_auc[site_indices] if len(site_indices) > 0 else np.array([], dtype=float)
        site_mean_auc = float(np.nanmean(site_aucs)) if np.isfinite(site_aucs).any() else np.nan

        final_score = self._nan_weighted_mean([aneurysm_auc, site_mean_auc], [0.5, 0.5])

        inv_map = {v: k for k, v in self.target_sites.items()}
        C = y_true.shape[1]
        per_class_dict = {inv_map.get(c, str(c)): (None if np.isnan(per_class_auc[c]) else float(per_class_auc[c])) for c in range(C)}
        site_order = sorted(self.target_sites.items(), key=lambda kv: kv[1])
        site_aucs_including_aneurysm = {name: (None if np.isnan(per_class_auc[idx]) else float(per_class_auc[idx])) for name, idx in site_order}
        site_aucs_excluding_aneurysm = {name: (None if np.isnan(per_class_auc[idx]) else float(per_class_auc[idx])) for name, idx in site_order if name != "Aneurysm Present"}

        mean_pred = float(np.mean(y_prob))
        pos_total = int(y_true.sum()); neg_total = int(y_true.size - pos_total)

        return {
            "auc_macro": auc_macro,
            "auc_micro": float(auc_micro) if not np.isnan(auc_micro) else np.nan,
            "auc_weighted": float(auc_weighted) if not np.isnan(auc_weighted) else np.nan,
            "per_class_auc": per_class_dict,
            "site_aucs_including_aneurysm": site_aucs_including_aneurysm,
            "site_aucs_excluding_aneurysm": site_aucs_excluding_aneurysm,
            "aneurysm_auc": (None if np.isnan(aneurysm_auc) else aneurysm_auc),
            "site_mean_auc": (None if np.isnan(site_mean_auc) else site_mean_auc),
            "final_score": (None if np.isnan(final_score) else final_score),
            "num_samples": int(y_true.shape[0]),
            "num_classes": int(y_true.shape[1]),
            "num_valid_classes": int(np.isfinite(per_class_auc).sum()),
            "mean_pred_overall": mean_pred,
            "pos_total": pos_total,
            "neg_total": neg_total,
        }
    
    
class RSNAPatchMultiAUC:
    """
    パッチデータ用の複数クラス分類メトリック
    症例毎のパッチの推論結果を受け取り、その最大値でクラス別AUCを計算する
    """
    
    def __init__(self, num_classes: int, class_names: list[str] = None, 
                 compute_on_step: bool = True, dist_sync_on_step: bool = False):
        """
        Args:
            num_classes: Number of classes
            class_names: List of class names for output
            compute_on_step: Whether to compute metric on each step (for compatibility)
            dist_sync_on_step: Whether to sync across processes on each step (for compatibility)
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"class_{i}" for i in range(num_classes)]
        self.compute_on_step = compute_on_step
        self.dist_sync_on_step = dist_sync_on_step
        self.reset()
    
    def reset(self):
        """Reset accumulated data"""
        # 症例IDをキーにして、その症例のパッチ予測結果を蓄積
        self._case_predictions = {}  # {series_id: list of predictions (num_patches, num_classes)}
        self._case_labels = {}       # {series_id: true labels (num_classes,)}
    
    def update(
        self, 
        y_true: torch.Tensor, 
        y_pred_logits: torch.Tensor, 
        series_ids: list[str]
    ):
        """
        パッチの予測結果を症例別に蓄積
        
        Args:
            y_true: Ground truth labels, shape (batch_size, num_classes)
            y_pred_logits: Prediction logits, shape (batch_size, num_classes)
            series_ids: List of series IDs, length batch_size
        """
        # Convert to numpy
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.detach().float().cpu().numpy()
        if isinstance(y_pred_logits, torch.Tensor):
            y_pred_logits = y_pred_logits.detach().float().cpu().numpy()
        
        # Convert logits to probabilities
        y_pred_probs = 1.0 / (1.0 + np.exp(-y_pred_logits))  # sigmoid
        
        # Accumulate per case
        for i, series_id in enumerate(series_ids):
            labels = y_true[i].astype(int)  # (num_classes,)
            probs = y_pred_probs[i]  # (num_classes,)
            
            if series_id not in self._case_predictions:
                self._case_predictions[series_id] = []
                self._case_labels[series_id] = labels.copy()
            
            # Add patch prediction
            self._case_predictions[series_id].append(probs)
            
            # Take union of positive labels (if any patch is positive, case is positive)
            self._case_labels[series_id] = np.maximum(self._case_labels[series_id], labels)
    
    def _safe_auc(self, y_true, y_pred):
        """Safe AUC calculation"""
        try:
            if len(np.unique(y_true)) < 2:
                return np.nan
            return metrics.roc_auc_score(y_true, y_pred)
        except Exception:
            return np.nan
    
    @property 
    def avg(self):
        """
        症例レベルのクラス別AUCを計算
        各症例のパッチ予測の最大値を使用
        """
        if not self._case_predictions:
            return {
                'auc_per_class': {name: np.nan for name in self.class_names},
                'auc_mean': np.nan,
                'num_cases': 0,
                'num_patches': 0
            }
        
        # 症例毎のクラス別最大予測値と真のラベルを取得
        case_max_probs = []  # (num_cases, num_classes)
        case_labels = []     # (num_cases, num_classes)
        total_patches = 0
        
        for series_id in self._case_predictions:
            # その症例のパッチ予測の最大値を各クラスについて取る
            patch_probs = np.array(self._case_predictions[series_id])  # (num_patches, num_classes)
            max_probs = np.max(patch_probs, axis=0)  # (num_classes,)
            labels = self._case_labels[series_id]  # (num_classes,)
            
            case_max_probs.append(max_probs)
            case_labels.append(labels)
            total_patches += len(self._case_predictions[series_id])
        
        case_max_probs = np.array(case_max_probs)  # (num_cases, num_classes)
        case_labels = np.array(case_labels)        # (num_cases, num_classes)
        
        # クラス別AUC計算
        auc_per_class = {}
        auc_list = []
        
        for i, class_name in enumerate(self.class_names):
            auc = self._safe_auc(case_labels[:, i], case_max_probs[:, i])
            auc_per_class[class_name] = float(auc) if not np.isnan(auc) else np.nan
            if not np.isnan(auc):
                auc_list.append(auc)
        
        # 平均AUC
        auc_mean = np.mean(auc_list) if auc_list else np.nan
        
        return {
            'auc_per_class': auc_per_class,
            'auc_mean': float(auc_mean) if not np.isnan(auc_mean) else np.nan,
            'num_cases': len(case_labels),
            'num_patches': total_patches,
            'pos_cases_per_class': {
                name: int(np.sum(case_labels[:, i])) 
                for i, name in enumerate(self.class_names)
            }
        }


def make_dummy_batch(B=6, C=14, Dm=4, Hm=8, Wm=8, seed=0):
    """ダミーの batch / outputs を生成します。
    - target_site: (B, C) の 0/1
    - outputs['output']: (B, N, C) の logits（N = Dm*Hm*Wm）
      * 正例のサンプル/クラスにだけ、上位セルに +bias を与えて AUC が出やすいようにしています。
    """
    rng = np.random.default_rng(seed)
    N = Dm * Hm * Wm

    # --- ラベル生成（全クラスで 1 が一度も出ないのを避けるため、確率を少し上げる）
    pos_prob = np.linspace(0.25, 0.45, C)  # クラスごとに 25-45% の陽性率
    y_true = (rng.random((B, C)) < pos_prob).astype(np.float32)

    # --- ロジット生成
    logits = rng.normal(0, 1, size=(B, N, C)).astype(np.float32)

    # 正例には上位セルにバイアスを付与（Aneurysm=13は少し強め）
    top_cells = max(1, N // 32)  # 上位約3%にシグナル
    for b in range(B):
        for c in range(C):
            if y_true[b, c] > 0:
                # 現在の確率順を見て上位セルを強化
                order = np.argsort(-logits[b, :, c])  # 降順 index
                k = top_cells
                bias = 2.0 if c != 13 else 3.0
                logits[b, order[:k], c] += bias

    batch = {
        "target_site": torch.from_numpy(y_true),   # (B,C)
        # NOTE: 推論で得られないので mask_map は入れない（本メトリクスは不要）
    }
    outputs = {
        "output": torch.from_numpy(logits),        # (B,N,C)
    }
    return batch, outputs


def main():
    # --- メトリクス作成：複数 Top-K を同時評価 ---
    metric = MultiROIAUC(
        aggregate_op="topk_mean",     # "mean" や "max" も可
        topk_list=[8, 16, 32, 64, 128],  # 複数Kを一括評価
        topk_basis="anchor",
        anchor_index=13,
        # topk_ratio_list=[0.01, 0.02],   # 比率で指定する場合はこちら（topk_listと併用可）
        use_sigmoid=True,
    )

    # --- ダミーバッチ生成＆更新 ---
    batch, outputs = make_dummy_batch(B=6, C=14, Dm=4, Hm=8, Wm=8, seed=42)
    metric.update(batch, outputs)

    # --- 集計 ---
    res = metric.avg

    # 代表K（active_topk もしくはベスト final_score）
    rep_k = res.get("representative_topk")
    print(f"Representative Top-K: {rep_k}")

    # 代表Kの概要
    print("=== Summary (representative) ===")
    for k in [
        "rep_auc_macro", "rep_auc_micro", "rep_auc_weighted",
        "rep_aneurysm_auc", "rep_site_mean_auc", "rep_final_score",
        "num_samples", "num_classes", "num_valid_classes",
    ]:
        if k in res:
            print(f"{k}: {res[k]}")

    # 各Kの最終スコアを一覧
    print("\n=== Final Score by Top-K ===")
    for k, d in sorted(res["by_topk"].items()):
        print(f"K={k:>4}: final_score={d['final_score']:.4f}, aneurysm_auc={d['aneurysm_auc']}, site_mean_auc={d['site_mean_auc']}")

    # （必要なら）クラス別AUCを表示
    inv = {v: k for k, v in MultiROIAUC.target_sites.items()}
    show_k = rep_k
    print(f"\n=== Per-class AUC (K={show_k}) ===")
    per_cls = res["by_topk"][show_k]["per_class_auc"]
    for idx in range(len(inv)):
        name = inv[idx]
        print(f"[{idx:02d}] {name}: {per_cls.get(name)}")


if __name__ == "__main__":
    main()

