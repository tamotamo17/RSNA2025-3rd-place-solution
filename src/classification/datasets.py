import os
import ast
import math
import random
from pathlib import Path
from typing import Any, Optional, Union, Sequence
from tqdm import tqdm
from collections import OrderedDict

import numpy as np
import pandas as pd
import cv2
from skimage.draw import disk
from scipy.ndimage import zoom
from skimage.draw import ellipse 
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import albumentations as A

Tuple3f = tuple[float, float, float]
Tuple3i = tuple[int, int, int]

class RSNADataset(Dataset):
    tp2idx = {
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
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        num_slice: int,
        num_group: int,  # input_ch=num_groupの時にtargets_sliceをまとめる
        mode: str,
        transform=None,
        offsets: list = [-1, 0, 1],
        r: int = 5,
        base_z_spacing: float = 1.0,
        z_sigma_mm: float = 0.25,
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.num_slice = num_slice
        self.num_group = num_group
        self.mode = mode
        self.transform = transform
        self.offsets = offsets
        self.r = r
        self.base_z_spacing = base_z_spacing
        self.z_sigma_mm = z_sigma_mm
        self.debug = debug

    # --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # --------------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid = row["SeriesInstanceUID"]
        mod = row["Modality"]
        plane = row["OrientationLabel"]
        filename = row["file_name"]

        # read metadata
        z_spacing_mm = row["z_spacing"]
        slices = row["sorted_files"]
        if self.mode == "train":
            offset = random.choice(self.offsets)
        else:
            offset = 0

        # which slices to load
        series_loc = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid].reset_index(drop=True)
        # only SOPInstanceUID
        sop_ids = [x.split("/")[-1][:-4] for x in slices]
        fname_to_idx = {p: i for i, p in enumerate(sop_ids)}
        # target_indices = sorted({fname_to_idx[f] for f in target_fnames if f in fname_to_idx})

        lin_indices = np.linspace(0, len(slices) - 1, self.num_slice).astype(int) + offset
        selected_indices = np.clip(lin_indices, 0, len(slices) - 1)

        # load pixel data
        try:
            volume = self.load_slices_blockwise(filename, selected_indices, out_dtype=np.float32)
            Z, H, W = volume.shape
        except:
            raise ValueError(f"Load failed: {sid}, {index}")

        # binary mask + slice_weights
        mask = np.zeros((Z, H, W), dtype=np.uint8)
        slice_weights = np.zeros((Z,), dtype=np.float32)
        targets_slice = np.zeros((Z, len(self.tp2idx.keys())), dtype=np.float32)
        targets = row[list(self.tp2idx.keys())].values.astype(np.float32)
        r_eff = max(1, int(round(self.r * z_spacing_mm / self.base_z_spacing)))
        t_max = math.ceil(3 * self.z_sigma_mm / z_spacing_mm)

        for _, row_loc in series_loc.iterrows():
            fname = row_loc["SOPInstanceUID"]
            location = row_loc["location"]
            idx_location = self.tp2idx[location]
            if fname not in fname_to_idx:
                continue
            orig_idx = fname_to_idx[fname]

            # --- ここで座標を取り出す ---
            coords = ast.literal_eval(row_loc["coordinates"])
            x, y = int(coords["x"]), int(coords["y"])

            # selected_indices に含まれる各スライスで減衰マスクを貼る
            for vidx, sel_orig in enumerate(selected_indices):
                dz = sel_orig - orig_idx
                if abs(dz) > t_max:
                    continue

                dz_mm = abs(dz) * z_spacing_mm
                coeff = math.exp(-(dz_mm**2) / (2 * self.z_sigma_mm**2))
                r_z = max(1, int(round(r_eff * coeff)))

                rr, cc = disk((y, x), radius=r_z, shape=(H, W))
                mask[vidx, rr, cc] = 1
                slice_weights[vidx] = max(slice_weights[vidx], coeff)
                targets_slice[vidx, idx_location] = 1
                targets_slice[vidx, 13] = 1

        if self.num_group != 1:
            Z, C = targets_slice.shape
            if Z % self.num_group != 0:
                raise ValueError("Z must be divisible by num_group")

            targets_slice = targets_slice.reshape(Z // self.num_group, self.num_group, C).max(axis=1)
            slice_weights = slice_weights.reshape(Z // self.num_group, self.num_group).max(axis=1)

        # optional transform
        if self.transform is not None:
            volume = volume.transpose(1, 2, 0)
            mask = mask.transpose(1, 2, 0)
            data = self.transform(image=volume, mask=mask)
            volume = data["image"].transpose(2, 0, 1)
            mask = data["mask"].transpose(2, 0, 1)
        if self.num_group != 1:
            Z, H, W = mask.shape
            mask = mask.reshape(Z // self.num_group, self.num_group, H, W).max(axis=1)
        volume = self._pct_normalize(volume)

        # output
        return {
            "image": torch.tensor(volume, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "slice_weights": torch.tensor(slice_weights, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "targets_slice": torch.tensor(targets_slice, dtype=torch.float32),
            "series_id": sid,
            "indices": np.array(selected_indices),
        }

    @staticmethod
    def load_slices_blockwise(npy_path: str, idx, out_dtype=np.float32):
        """
        npy(memmap)から指定Zインデックスを高速に読み出す。
        1) idxをソート
        2) 連続ランをまとめて mm[a:b] で一気に読む（連続I/O）
        3) 連結 → 元の順序に戻す
        """
        mm = np.load(npy_path, mmap_mode="r")  # shape: (Z,H,W)
        idx = np.asarray(idx, dtype=np.int64)

        # もとの順序を保存
        order = np.argsort(idx)
        inv_order = np.empty_like(order)
        inv_order[order] = np.arange(order.size)
        idx_sorted = idx[order]

        # 連続ランに分割
        if idx_sorted.size == 0:
            return np.empty((0,) + mm.shape[1:], dtype=out_dtype)

        splits = np.where(np.diff(idx_sorted) != 1)[0] + 1
        runs = np.split(idx_sorted, splits)  # 例: [array([10,11,12]), array([20,21]), ...]

        # 各ランをまとめ読みして連結
        parts = [mm[r[0] : r[-1] + 1] for r in runs]  # 連続領域を一括メモリマップ読み
        vol_sorted = np.concatenate(parts, axis=0, dtype=mm.dtype)

        # 元の順序に戻す
        vol = vol_sorted[inv_order]

        # 必要ならここで dtype 変換（最後に一回だけ）
        if out_dtype is not None and vol.dtype != out_dtype:
            vol = vol.astype(out_dtype, copy=False)
        return vol

    @staticmethod
    def _pct_normalize(arr: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
        """
        パーセンタイルで 0–1 正規化 (clamp)。arr は ndarray (任意 shape)。
        """
        lo = np.percentile(arr, p_low)
        hi = np.percentile(arr, p_high)
        if hi <= lo:  # 画一値対策
            return np.zeros_like(arr, dtype=np.float32)
        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0)
        return arr.astype(np.float32)


class RSNApngDataset(Dataset):
    tp2idx = {
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
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        num_slice: int,
        num_group: int,  # input_ch=num_groupの時にtargets_sliceをまとめる
        mode: str,
        transform=None,
        offsets: list = [-1, 0, 1],
        r: int = 5,
        base_z_spacing: float = 1.0,
        z_sigma_mm: float = 0.25,
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.num_slice = num_slice
        self.num_group = num_group
        self.mode = mode
        self.transform = transform
        self.offsets = offsets
        self.r = r
        self.base_z_spacing = base_z_spacing
        self.z_sigma_mm = z_sigma_mm
        self.debug = debug

    # --------------------------------------------------------
    def __len__(self) -> int:
        return len(self.df)

    # --------------------------------------------------------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid = row["SeriesInstanceUID"]
        mod = row["Modality"]
        plane = row["OrientationLabel"]

        # read metadata
        z_spacing_mm = row["z_spacing"]
        slices = row["sorted_files"]
        if self.mode == "train":
            offset = random.choice(self.offsets)
        else:
            offset = 0

        # which slices to load
        series_loc = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid].reset_index(drop=True)
        # only SOPInstanceUID
        sop_ids = [x.split("/")[-1][:-4] for x in slices]
        fname_to_idx = {p: i for i, p in enumerate(sop_ids)}
        # target_indices = sorted({fname_to_idx[f] for f in target_fnames if f in fname_to_idx})

        lin_indices = np.linspace(0, len(slices) - 1, self.num_slice).astype(int) + offset
        selected_indices = np.clip(lin_indices, 0, len(slices) - 1)
        selected_paths = [slices[i] for i in selected_indices]
        # load pixel data
        try:
            volume = np.stack([cv2.imread(p, cv2.IMREAD_GRAYSCALE).astype(np.float32) for p in selected_paths], axis=0)
            Z, H, W = volume.shape
        except:
            raise ValueError(f"Load failed: {sid}, {index}")

        # binary mask + slice_weights
        mask = np.zeros((Z, H, W), dtype=np.uint8)
        slice_weights = np.zeros((Z,), dtype=np.float32)
        targets_slice = np.zeros((Z, len(self.tp2idx.keys())), dtype=np.float32)
        targets = row[list(self.tp2idx.keys())].values.astype(np.float32)
        r_eff = max(1, int(round(self.r * z_spacing_mm / self.base_z_spacing)))
        t_max = math.ceil(3 * self.z_sigma_mm / z_spacing_mm)

        for _, row_loc in series_loc.iterrows():
            fname = row_loc["SOPInstanceUID"]
            location = row_loc["location"]
            idx_location = self.tp2idx[location]
            if fname not in fname_to_idx:
                continue
            orig_idx = fname_to_idx[fname]

            # --- ここで座標を取り出す ---
            coords = ast.literal_eval(row_loc["coordinates"])
            x, y = int(coords["x"]), int(coords["y"])

            # selected_indices に含まれる各スライスで減衰マスクを貼る
            for vidx, sel_orig in enumerate(selected_indices):
                dz = sel_orig - orig_idx
                if abs(dz) > t_max:
                    continue

                dz_mm = abs(dz) * z_spacing_mm
                coeff = math.exp(-(dz_mm**2) / (2 * self.z_sigma_mm**2))
                r_z = max(1, int(round(r_eff * coeff)))

                rr, cc = disk((y, x), radius=r_z, shape=(H, W))
                mask[vidx, rr, cc] = 1
                slice_weights[vidx] = max(slice_weights[vidx], coeff)
                targets_slice[vidx, idx_location] = 1
                targets_slice[vidx, 13] = 1

        if self.num_group != 1:
            Z, C = targets_slice.shape
            if Z % self.num_group != 0:
                raise ValueError(f"Z must be divisible by num_group, z={Z}, num_group={self.num_group}")

            targets_slice = targets_slice.reshape(Z // self.num_group, self.num_group, C).max(axis=1)
            slice_weights = slice_weights.reshape(Z // self.num_group, self.num_group).max(axis=1)

        # optional transform
        if self.transform is not None:
            volume = volume.transpose(1, 2, 0) # HWZ
            mask = mask.transpose(1, 2, 0) # HWZ
            data = self.transform(image=volume, mask=mask)
            volume = data["image"].transpose(2, 0, 1) # ZHW
            mask = data["mask"].transpose(2, 0, 1) # ZHW
        if self.num_group != 1:
            Z, H, W = mask.shape
            mask = mask.reshape(Z // self.num_group, self.num_group, H, W).max(axis=1)
        volume = self._pct_normalize(volume)

        # output
        return {
            "image": torch.tensor(volume, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32),
            "slice_weights": torch.tensor(slice_weights, dtype=torch.float32),
            "targets": torch.tensor(targets, dtype=torch.float32),
            "targets_slice": torch.tensor(targets_slice, dtype=torch.float32),
            "series_id": sid,
            "indices": np.array(selected_indices),
        }

    @staticmethod
    def _pct_normalize(arr: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
        """
        パーセンタイルで 0–1 正規化 (clamp)。arr は ndarray (任意 shape)。
        """
        lo = np.percentile(arr, p_low)
        hi = np.percentile(arr, p_high)
        if hi <= lo:  # 画一値対策
            return np.zeros_like(arr, dtype=np.float32)
        arr = (arr - lo) / (hi - lo)
        arr = np.clip(arr, 0.0, 1.0)
        return arr.astype(np.float32)
    

class RSNAPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, df_loc: pd.DataFrame, mode: str,
                 patch_size_mm: tuple[float, float, float],
                 out_size_zyx: tuple[int, int, int],
                 r: float = 5.0, r_unit: str = "mm",
                 p_lesion_crop: float = 0.7,
                 jitter_mm: tuple[float, float, float] = None,
                 choose_random_lesion_in_train: bool = True,
                 transform: A.Compose|None =None,
                 debug: bool = False):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.patch_size_mm = patch_size_mm
        self.out_size_zyx = out_size_zyx
        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_lesion_crop = float(p_lesion_crop)
        self.jitter_mm = jitter_mm
        self.choose_random_lesion_in_train = choose_random_lesion_in_train
        self.transform = transform
        self.debug = debug

    def __len__(self) -> int:
        return len(self.df)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0

    @staticmethod
    def _pad_and_crop(vol: np.ndarray, zc: int, yc: int, xc: int,
                      hz: int, hy: int, hx: int) -> tuple[np.ndarray, tuple[int, int, int]]:
        Z, H, W = vol.shape
        z1, z2 = zc - hz, zc + hz + 1
        y1, y2 = yc - hy, yc + hy + 1
        x1, x2 = xc - hx, xc + hx + 1

        pad_before = [max(0, -z1), max(0, -y1), max(0, -x1)]
        pad_after  = [max(0, z2 - Z), max(0, y2 - H), max(0, x2 - W)]

        z1_c, y1_c, x1_c = max(0, z1), max(0, y1), max(0, x1)
        z2_c, y2_c, x2_c = min(Z, z2), min(H, y2), min(W, x2)

        patch = vol[...,x1_c:x2_c]
        patch = patch[z1_c:z2_c,y1_c:y2_c]
        patch = np.asarray(patch)
        if any(pad_before) or any(pad_after):
            out = np.zeros((z2 - z1, y2 - y1, x2 - x1), dtype=vol.dtype)
            out[
                pad_before[0]: pad_before[0] + (z2_c - z1_c),
                pad_before[1]: pad_before[1] + (y2_c - y1_c),
                pad_before[2]: pad_before[2] + (x2_c - x1_c),
            ] = patch
            patch = out
        return patch, (max(z1, 0), max(y1, 0), max(x1, 0))

    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int, int, int], mode: str) -> np.ndarray:
        Z, H, W = vol_zyx.shape
        Z_out, H_out, W_out = out_zyx
        zoom_factors = (Z_out / Z, H_out / H, W_out / W)
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _pct_normalize(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
        lo = np.percentile(arr, p_low); hi = np.percentile(arr, p_high)
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        return arr
    
    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low=-100, v_high=600) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        arr = (arr - v_low) / (v_high - v_low + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        return arr

    # ---------- crop policy ----------
    @staticmethod
    def _randint_inclusive(a: int, b: int) -> int:
        return int(np.random.randint(a, b + 1)) if a <= b else int(a)

    def _choose_random_center_inside(self, Z: int, H: int, W: int, hz: int, hy: int, hx: int) -> tuple[int, int, int]:
        zc = self._randint_inclusive(hz, max(hz, Z - 1 - hz))
        yc = self._randint_inclusive(hy, max(hy, H - 1 - hy))
        xc = self._randint_inclusive(hx, max(hx, W - 1 - hx))
        return zc, yc, xc

    def _choose_center_with_lesion(self, zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx, jitter_vox):
        jz, jy, jx = jitter_vox

        # 病変が入るための中心範囲
        zc_min, zc_max = zc_l - hz, zc_l + hz
        yc_min, yc_max = yc_l - hy, yc_l + hy
        xc_min, xc_max = xc_l - hx, xc_l + hx

        # jitter を考慮
        zc_rng = (max(zc_min, zc_l - jz), min(zc_max, zc_l + jz))
        yc_rng = (max(yc_min, yc_l - jy), min(yc_max, yc_l + jy))
        xc_rng = (max(xc_min, xc_l - jx), min(xc_max, xc_l + jx))
        # zc_rng = (max(zc_min, zc_l), min(zc_max, zc_l))
        # yc_rng = (max(yc_min, yc_l), min(yc_max, yc_l))
        # xc_rng = (max(xc_min, xc_l), min(xc_max, xc_l))

        # さらに「パッチが画内に収まる」よう制限
        zc_rng = (max(zc_rng[0], hz), min(zc_rng[1], Z - 1 - hz))
        yc_rng = (max(yc_rng[0], hy), min(yc_rng[1], H - 1 - hy))
        xc_rng = (max(xc_rng[0], hx), min(xc_rng[1], W - 1 - hx))

        # もし範囲がつぶれたら（病変が端すぎ）→最も近い内側へクリップ
        if zc_rng[0] > zc_rng[1]:
            zc = int(np.clip(zc_l, hz, Z - 1 - hz))
        else:
            zc = self._randint_inclusive(int(math.ceil(zc_rng[0])), int(math.floor(zc_rng[1])))

        if yc_rng[0] > yc_rng[1]:
            yc = int(np.clip(yc_l, hy, H - 1 - hy))
        else:
            yc = self._randint_inclusive(int(math.ceil(yc_rng[0])), int(math.floor(yc_rng[1])))

        if xc_rng[0] > xc_rng[1]:
            xc = int(np.clip(xc_l, hx, W - 1 - hx))
        else:
            xc = self._randint_inclusive(int(math.ceil(xc_rng[0])), int(math.floor(xc_rng[1])))

        return zc, yc, xc
    
    def _choose_center_with_lesion_nojitter(self, zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx):
        # 病変が入るための中心範囲
        zc_rng = (zc_l - hz, zc_l + hz)
        yc_rng = (yc_l - hy, yc_l + hy)
        xc_rng = (xc_l - hx, xc_l + hx)

        # パッチが画内に収まるよう制限
        zc_lo = max(zc_rng[0], hz);           zc_hi = min(zc_rng[1], Z - 1 - hz)
        yc_lo = max(yc_rng[0], hy);           yc_hi = min(yc_rng[1], H - 1 - hy)
        xc_lo = max(xc_rng[0], hx);           xc_hi = min(xc_rng[1], W - 1 - hx)

        # 範囲が潰れたら病変近傍を内側にクリップ
        zc = int(np.clip(zc_l, hz, Z - 1 - hz)) if zc_lo > zc_hi else self._randint_inclusive(int(math.ceil(zc_lo)), int(math.floor(zc_hi)))
        yc = int(np.clip(yc_l, hy, H - 1 - hy)) if yc_lo > yc_hi else self._randint_inclusive(int(math.ceil(yc_lo)), int(math.floor(yc_hi)))
        xc = int(np.clip(xc_l, hx, W - 1 - hx)) if xc_lo > xc_hi else self._randint_inclusive(int(math.ceil(xc_lo)), int(math.floor(xc_hi)))

        return zc, yc, xc


    @staticmethod
    def _match_label_key(label_keys: list[str], site_name: str | None) -> str | None:
        if not site_name:
            return None
        s = site_name.strip().lower()
        for k in label_keys:
            if k.lower() == s:
                return k
        cand = [k for k in label_keys if s in k.lower()]
        return cand[0] if cand else None

    @staticmethod
    def _pick_sop_value(loc_row: pd.Series) -> str | None:
        if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]):
            return str(loc_row["SOPInstanceUID"])
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = row["SeriesInstanceUID"]
        npy_path: str = row["file_name"]
        sorted_files: list[str] = row["sorted_files"]
        mod = row["Modality"]
        plane = row["OrientationLabel"]
        vmin_mr = row['intensity_p0']
        vmax_mr = row['intensity_p100']

        # spacing
        z_sp: float = float(row["z_spacing"])
        y_sp, x_sp = self._parse_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)))

        # lesion rows
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        has_lesion = len(loc_rows) > 0

        # load memmap
        vol = np.load(npy_path, mmap_mode="r")  # (Z,H,W)
        Z, H, W = vol.shape

        # mm → ボクセル半サイズ
        z_mm, y_mm, x_mm = self.patch_size_mm
        hz = max(0, int(round((z_mm / z_sp) / 2.0)))
        hy = max(0, int(round((y_mm / y_sp) / 2.0)))
        hx = max(0, int(round((x_mm / x_sp) / 2.0)))

        # jitter のボクセル換算（指定なければパッチ半径まで）
        if self.jitter_mm is None:
            jz, jy, jx = hz, hy, hx
        else:
            jz = int(round(self.jitter_mm[0] / z_sp))
            jy = int(round(self.jitter_mm[1] / y_sp))
            jx = int(round(self.jitter_mm[2] / x_sp))

        do_lesion = has_lesion and (np.random.rand() < self.p_lesion_crop)

        mask_should_draw = False
        label_key = None

        if do_lesion:
            if self.mode == "train" and self.choose_random_lesion_in_train and len(loc_rows) != 0:
                loc_row = loc_rows.sample(n=1).iloc[0]
            else:
                loc_row = loc_rows.iloc[0]

            # df_loc['location'] は df の列名と一致する前提
            label_key = str(loc_row["location"]) if "location" in loc_row.index else None

            # z: SOP→index
            sop_value = self._pick_sop_value(loc_row)
            if sop_value is not None:
                names = [p.split("/")[-1][:-4] for p in sorted_files]
                try:
                    zc_l = names.index(sop_value)
                except ValueError:
                    zc_l = 0
            else:
                coords = ast.literal_eval(loc_row["coordinates"])
                zc_l = int(coords["z"]) if "z" in coords else 0

            coords = ast.literal_eval(loc_row["coordinates"])
            xc_l = int(coords["x"]); yc_l = int(coords["y"])
            if np.random.rand() < 0.7:
                # 小さめ jitter 範囲で一様
                jz, jy, jx = hz//3, hy//3, hx//3

                # 病変を必ず含む中心
                zc, yc, xc = self._choose_center_with_lesion(
                    zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx, (jz, jy, jx)
                    )
            else:
                zc, yc, xc = self._choose_center_with_lesion_nojitter(
                    zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx
                    )
            mask_should_draw = True
        else:
            zc, yc, xc = self._choose_random_center_inside(Z, H, W, hz, hy, hx)

        # ---------- パッチ & マスク ----------
        patch, (z1, y1, x1) = self._pad_and_crop(vol, zc, yc, xc, hz, hy, hx)
        mask = np.zeros_like(patch, dtype=np.uint8)

        if mask_should_draw:
            # 病変の相対座標（パッチ内）
            py = int((yc_l if do_lesion else yc) - y1)
            px = int((xc_l if do_lesion else xc) - x1)
            py = int(np.clip(py, 0, mask.shape[1] - 1))
            px = int(np.clip(px, 0, mask.shape[2] - 1))

            # --- 3D楕円体の半径（mm→px）---
            if self.r_unit == "mm":
                rx0 = max(1, int(round(self.r / x_sp)))  # X半径(px)
                ry0 = max(1, int(round(self.r / y_sp)))  # Y半径(px)
                rz0 = max(1, int(round(self.r / z_sp)))  # Z半径(スライス数)
            else:
                rx0 = ry0 = rz0 = max(1, int(round(self.r)))

            # 病変のZ位置（パッチ座標系）
            #   zc_l: 病変のZ（シリーズ座標）
            #   z1:   パッチ開始Z（シリーズ座標）
            pz = int(zc_l - z1)
            pz = int(np.clip(pz, 0, mask.shape[0] - 1))

            # Z方向に -rz0..+rz0 へ走査し、楕円半径を縮小して描く
            Zp = mask.shape[0]
            for dz in range(-rz0, rz0 + 1):
                zc_slice = pz + dz
                if zc_slice < 0 or zc_slice >= Zp:
                    continue
                # 楕円体の断面縮小係数（単位球の式 x^2/rx0^2 + y^2/ry0^2 + z^2/rz0^2 <= 1 から）
                scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2))
                rx = max(1, int(round(rx0 * scale)))
                ry = max(1, int(round(ry0 * scale)))
                rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=mask.shape[1:])
                mask[zc_slice, rr, cc] = 1

        # ---------- リサイズ & 正規化 ----------
        Z_out, H_out, W_out = self.out_size_zyx
        patch = self._resize3d(patch.astype(np.float32, copy=False), (Z_out, H_out, W_out), mode="trilinear")
        mask  = self._resize3d(mask.astype(np.float32),           (Z_out, H_out, W_out), mode="nearest")
        if mod == 'CTA':
            patch = self._vl_normalize(patch, v_low=-100, v_high=600)
        else:
            patch = self._vl_normalize(patch, vmin_mr, vmax_mr)
        if self.transform is not None:
            patch = patch.transpose(1, 2, 0) # HWZ
            mask = mask.transpose(1, 2, 0) # HWZ
            data = self.transform(image=patch, mask=mask)
            patch = data["image"].transpose(2, 0, 1) # ZHW
            mask = data["mask"].transpose(2, 0, 1) # ZHW
        patch = patch.astype(np.float32) / 255.
        
        # ---------- ラベル処理 ----------
        targets = None
        label_keys = [k for k in (
            "Left Infraclinoid Internal Carotid Artery",
            "Right Infraclinoid Internal Carotid Artery",
            "Left Supraclinoid Internal Carotid Artery",
            "Right Supraclinoid Internal Carotid Artery",
            "Left Middle Cerebral Artery",
            "Right Middle Cerebral Artery",
            "Anterior Communicating Artery",
            "Left Anterior Cerebral Artery",
            "Right Anterior Cerebral Artery",
            "Left Posterior Communicating Artery",
            "Right Posterior Communicating Artery",
            "Basilar Tip",
            "Other Posterior Circulation",
            "Aneurysm Present",
        ) if k in self.df.columns]

        if label_keys:
            row = self.df.iloc[index]
            targets = row[label_keys].values.astype(np.float32)

        # 部位限定スカラターゲット
        target_aneurysm = [float(row["Aneurysm Present"])]
            
        out = {
            "image": torch.from_numpy(patch).float(),
            "mask": torch.from_numpy(mask).float(),
            "series_id": sid,
            "center_zyx": (int(zc), int(yc), int(xc)),
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "patch_index_start_zyx": (int(z1), int(y1), int(x1)),
            "mode": self.mode,
            "used_lesion_crop": bool(mask_should_draw),
            "target_site": torch.tensor(targets).float(),
            "target_aneurysm": torch.tensor(target_aneurysm).float(),
        }
        return out

class RSNAPatchDatasetV2(Dataset):
    def __init__(self, df: pd.DataFrame, df_loc: pd.DataFrame, mode: str,
                 volume_size_mm:tuple[float, float, float],
                 patch_size_mm: tuple[float, float, float],
                 out_size_zyx: tuple[int, int, int],
                 r: float = 5.0, r_unit: str = "mm",
                 p_lesion_crop: float = 0.7,
                 jitter_mm: tuple[float, float, float] = None,
                 choose_random_lesion_in_train: bool = True,
                 p_flip: float = 0.,
                 transform: A.Compose|None =None,
                 debug: bool = False):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.volume_size_mm = volume_size_mm
        self.patch_size_mm = patch_size_mm
        self.out_size_zyx = out_size_zyx
        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_lesion_crop = float(p_lesion_crop)
        self.jitter_mm = jitter_mm
        self.choose_random_lesion_in_train = choose_random_lesion_in_train
        self.p_flip = p_flip
        self.transform = transform
        self.debug = debug

    def __len__(self) -> int:
        return len(self.df)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0
    
    @staticmethod
    def _pad_value_for_mod(mod, vmin_mr) -> float:
        # 'CT', 'CTA', 'CCTA' などを CT として扱う
        if mod == "CTA":
            return -100.0
        else:
            return vmin_mr
    
    @staticmethod
    def crop_roi_mm(
        vol: np.ndarray,               # (Z,H,W) memmap
        x_min, x_max, y_min, y_max, z_min, z_max,
        z_sp: float, y_sp: float, x_sp: float,
        size_mm_zyx: tuple[float, float, float],
        pad_value: float | str = -1024,    # ★CTなら既知値を推奨。'min'は全量走査になるので基本避ける
    ):
        import math
        Z, H, W = vol.shape

        def _center_or_mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if math.isnan(fa) or math.isnan(fb):
                    raise ValueError
                return 0.5 * (fa + fb)
            except Exception:
                return lim / 2.0

        # 中心（voxel）
        cx = int(round(_center_or_mid(x_min, x_max, W)))
        cy = int(round(_center_or_mid(y_min, y_max, H)))
        cz = int(round(_center_or_mid(z_min, z_max, Z)))

        # 半サイズ[voxel]（偶数幅: 2*hv）
        z_mm, y_mm, x_mm = size_mm_zyx
        hvz = max(1, int(round((z_mm / z_sp) / 2.0)))
        hvy = max(1, int(round((y_mm / y_sp) / 2.0)))
        hvx = max(1, int(round((x_mm / x_sp) / 2.0)))

        z1, z2 = cz - hvz, cz + hvz
        y1, y2 = cy - hvy, cy + hvy
        x1, x2 = cx - hvx, cx + hvx

        # 画像内の交差
        z1c, z2c = max(0, z1), min(Z, z2)
        y1c, y2c = max(0, y1), min(H, y2)
        x1c, x2c = max(0, x1), min(W, x2)

        # 出力を先に作る（pad_value で埋める）
        if pad_value == "min":
            # 本当に必要な時だけ使う（重い）
            padv = float(vol.min())
        else:
            padv = float(pad_value)
        roi = np.empty((z2 - z1, y2 - y1, x2 - x1), dtype=vol.dtype)
        roi.fill(padv)

        # 転写先オフセット
        dz0 = z1c - z1
        dy0 = y1c - y1
        dx0 = x1c - x1

        # ★memmap に優しい順: まず最後の軸で連続スライス → 残りを切る
        sub = vol[..., x1c:x2c]                     # 連続ブロック
        sub = sub[z1c:z2c, y1c:y2c]                 # 面とスライスを抜く

        roi[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub

        return roi, (int(z1), int(y1), int(x1))     # ROI と “元座標での開始点”

    @staticmethod
    def _pad_and_crop(vol: np.ndarray, zc: int, yc: int, xc: int,
                    hz: int, hy: int, hx: int,
                    pad_value: float = 0.0) -> tuple[np.ndarray, tuple[int, int, int]]:
        Z, H, W = vol.shape
        z1, z2 = zc - hz, zc + hz + 1
        y1, y2 = yc - hy, yc + hy + 1
        x1, x2 = xc - hx, xc + hx + 1

        z1c, z2c = max(0, z1), min(Z, z2)
        y1c, y2c = max(0, y1), min(H, y2)
        x1c, x2c = max(0, x1), min(W, x2)

        out = np.empty((z2 - z1, y2 - y1, x2 - x1), dtype=vol.dtype)
        out.fill(float(pad_value))   # ★ ここがゼロ埋め→指定値埋めに

        dz0 = z1c - z1
        dy0 = y1c - y1
        dx0 = x1c - x1

        # 連続ブロックを先に切る（memmap 向け）
        sub = vol[..., x1c:x2c]
        sub = sub[z1c:z2c, y1c:y2c]
        out[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub

        return out, (int(max(z1, 0)), int(max(y1, 0)), int(max(x1, 0)))

    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int, int, int], mode: str) -> np.ndarray:
        Z, H, W = vol_zyx.shape
        Z_out, H_out, W_out = out_zyx
        zoom_factors = (Z_out / Z, H_out / H, W_out / W)
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _pct_normalize(arr: np.ndarray, p_low=1, p_high=99) -> np.ndarray:
        lo = np.percentile(arr, p_low); hi = np.percentile(arr, p_high)
        if hi <= lo:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, lo, hi)
        arr = (arr - lo) / (hi - lo + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        return arr
    
    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low=-100, v_high=600) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        arr = (arr - v_low) / (v_high - v_low + 1e-6)
        arr = (arr * 255).astype(np.uint8)
        return arr

    # ---------- crop policy ----------
    @staticmethod
    def _randint_inclusive(a: int, b: int) -> int:
        return int(np.random.randint(a, b + 1)) if a <= b else int(a)

    def _choose_random_center_inside(self, Z: int, H: int, W: int, hz: int, hy: int, hx: int) -> tuple[int, int, int]:
        zc = self._randint_inclusive(hz, max(hz, Z - 1 - hz))
        yc = self._randint_inclusive(hy, max(hy, H - 1 - hy))
        xc = self._randint_inclusive(hx, max(hx, W - 1 - hx))
        return zc, yc, xc

    def _choose_center_with_lesion(self, zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx, jitter_vox):
        jz, jy, jx = jitter_vox

        # 病変が入るための中心範囲
        zc_min, zc_max = zc_l - hz, zc_l + hz
        yc_min, yc_max = yc_l - hy, yc_l + hy
        xc_min, xc_max = xc_l - hx, xc_l + hx

        # jitter を考慮
        zc_rng = (max(zc_min, zc_l - jz), min(zc_max, zc_l + jz))
        yc_rng = (max(yc_min, yc_l - jy), min(yc_max, yc_l + jy))
        xc_rng = (max(xc_min, xc_l - jx), min(xc_max, xc_l + jx))
        # zc_rng = (max(zc_min, zc_l), min(zc_max, zc_l))
        # yc_rng = (max(yc_min, yc_l), min(yc_max, yc_l))
        # xc_rng = (max(xc_min, xc_l), min(xc_max, xc_l))

        # さらに「パッチが画内に収まる」よう制限
        zc_rng = (max(zc_rng[0], hz), min(zc_rng[1], Z - 1 - hz))
        yc_rng = (max(yc_rng[0], hy), min(yc_rng[1], H - 1 - hy))
        xc_rng = (max(xc_rng[0], hx), min(xc_rng[1], W - 1 - hx))

        # もし範囲がつぶれたら（病変が端すぎ）→最も近い内側へクリップ
        if zc_rng[0] > zc_rng[1]:
            zc = int(np.clip(zc_l, hz, Z - 1 - hz))
        else:
            zc = self._randint_inclusive(int(math.ceil(zc_rng[0])), int(math.floor(zc_rng[1])))

        if yc_rng[0] > yc_rng[1]:
            yc = int(np.clip(yc_l, hy, H - 1 - hy))
        else:
            yc = self._randint_inclusive(int(math.ceil(yc_rng[0])), int(math.floor(yc_rng[1])))

        if xc_rng[0] > xc_rng[1]:
            xc = int(np.clip(xc_l, hx, W - 1 - hx))
        else:
            xc = self._randint_inclusive(int(math.ceil(xc_rng[0])), int(math.floor(xc_rng[1])))

        return zc, yc, xc
    
    def _choose_center_with_lesion_nojitter(self, zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx):
        # 病変が入るための中心範囲
        zc_rng = (zc_l - hz, zc_l + hz)
        yc_rng = (yc_l - hy, yc_l + hy)
        xc_rng = (xc_l - hx, xc_l + hx)

        # パッチが画内に収まるよう制限
        zc_lo = max(zc_rng[0], hz);           zc_hi = min(zc_rng[1], Z - 1 - hz)
        yc_lo = max(yc_rng[0], hy);           yc_hi = min(yc_rng[1], H - 1 - hy)
        xc_lo = max(xc_rng[0], hx);           xc_hi = min(xc_rng[1], W - 1 - hx)

        # 範囲が潰れたら病変近傍を内側にクリップ
        zc = int(np.clip(zc_l, hz, Z - 1 - hz)) if zc_lo > zc_hi else self._randint_inclusive(int(math.ceil(zc_lo)), int(math.floor(zc_hi)))
        yc = int(np.clip(yc_l, hy, H - 1 - hy)) if yc_lo > yc_hi else self._randint_inclusive(int(math.ceil(yc_lo)), int(math.floor(yc_hi)))
        xc = int(np.clip(xc_l, hx, W - 1 - hx)) if xc_lo > xc_hi else self._randint_inclusive(int(math.ceil(xc_lo)), int(math.floor(xc_hi)))

        return zc, yc, xc


    @staticmethod
    def _match_label_key(label_keys: list[str], site_name: str | None) -> str | None:
        if not site_name:
            return None
        s = site_name.strip().lower()
        for k in label_keys:
            if k.lower() == s:
                return k
        cand = [k for k in label_keys if s in k.lower()]
        return cand[0] if cand else None

    @staticmethod
    def _pick_sop_value(loc_row: pd.Series) -> str | None:
        if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]):
            return str(loc_row["SOPInstanceUID"])
        return None
    

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = row["SeriesInstanceUID"]
        npy_path: str = row["file_name"]
        sorted_files: list[str] = row["sorted_files"]
        mod = row["Modality"]
        plane = row["OrientationLabel"]
        vmin_mr = row['intensity_p0']
        vmax_mr = row['intensity_p100']
        x_min, x_max = row['x1'], row['x2']
        y_min, y_max = row['y1'], row['y2']
        z_min, z_max = row['z1'], row['z2']

        # spacing
        z_sp: float = float(row["z_spacing"])
        y_sp, x_sp = self._parse_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)))

        # lesion rows
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        has_lesion = len(loc_rows) > 0

        # load memmap (Z,H,W)
        vol = np.load(npy_path, mmap_mode="r")
        pad_val = self._pad_value_for_mod(mod, vmin_mr)

        # ====== ① ROI を実際に使う ======
        roi, (z0, y0, x0) = self.crop_roi_mm(
            vol,
            x_min, x_max, y_min, y_max, z_min, z_max,
            z_sp, y_sp, x_sp,
            size_mm_zyx=self.volume_size_mm,
            pad_value=pad_val,
        )
        Zr, Hr, Wr = roi.shape

        # 半サイズ[voxel]
        z_mm, y_mm, x_mm = self.patch_size_mm
        hz = max(0, int(round((z_mm / z_sp) / 2.0)))
        hy = max(0, int(round((y_mm / y_sp) / 2.0)))
        hx = max(0, int(round((x_mm / x_sp) / 2.0)))

        # jitter を voxel に
        if self.jitter_mm is None:
            jz, jy, jx = hz, hy, hx
        else:
            jz = int(round(self.jitter_mm[0] / z_sp))
            jy = int(round(self.jitter_mm[1] / y_sp))
            jx = int(round(self.jitter_mm[2] / x_sp))

        do_lesion = has_lesion and (np.random.rand() < self.p_lesion_crop)
        mask_should_draw = False
        label_key = None  # 未使用だが残す

        # ====== ② 中心点は ROI 基準で選ぶ ======
        if do_lesion:
            if self.mode == "train" and self.choose_random_lesion_in_train and len(loc_rows) != 0:
                loc_row = loc_rows.sample(n=1).iloc[0]
            else:
                loc_row = loc_rows.iloc[0]

            label_key = str(loc_row["location"]) if "location" in loc_row.index else None

            # z: SOP→index（シリーズ座標）
            sop_value = self._pick_sop_value(loc_row)
            if sop_value is not None:
                names = [p.split("/")[-1][:-4] for p in sorted_files]
                try:
                    zc_l = names.index(sop_value)
                except ValueError:
                    zc_l = 0
            else:
                coords = ast.literal_eval(loc_row["coordinates"])
                zc_l = int(coords["z"]) if "z" in coords else 0

            coords = ast.literal_eval(loc_row["coordinates"])
            xc_l = int(coords["x"]); yc_l = int(coords["y"])

            # --- シリーズ座標 → ROI座標へ変換 ---
            zc_l_r = zc_l - z0
            yc_l_r = yc_l - y0
            xc_l_r = xc_l - x0

            if np.random.rand() < 0.7:
                jz_s, jy_s, jx_s = hz//3, hy//3, hx//3
                zc, yc, xc = self._choose_center_with_lesion(
                    zc_l_r, yc_l_r, xc_l_r, Zr, Hr, Wr, hz, hy, hx, (jz_s, jy_s, jx_s)
                )
            else:
                zc, yc, xc = self._choose_center_with_lesion_nojitter(
                    zc_l_r, yc_l_r, xc_l_r, Zr, Hr, Wr, hz, hy, hx
                )
            mask_should_draw = True
        else:
            zc, yc, xc = self._choose_random_center_inside(Zr, Hr, Wr, hz, hy, hx)

        # ====== ②.5 グリッド座標（ROI内での位置を離散/連続で） ======
        # 期待するグリッド数（mm指定から計算）
        gZ = max(1, int(round(self.volume_size_mm[0] / self.patch_size_mm[0])))
        gY = max(1, int(round(self.volume_size_mm[1] / self.patch_size_mm[1])))
        gX = max(1, int(round(self.volume_size_mm[2] / self.patch_size_mm[2])))

        # ROI のボクセルサイズ（Zr,Hr,Wr）は crop_roi_mm の返り値 roi.shape
        bin_z = Zr / float(gZ)
        bin_y = Hr / float(gY)
        bin_x = Wr / float(gX)

        # パッチ中心（zc,yc,xc は ROI 座標）→ 属するグリッドのインデックス
        iz = min(gZ - 1, int((zc + 0.5) / bin_z))
        iy = min(gY - 1, int((yc + 0.5) / bin_y))
        ix = min(gX - 1, int((xc + 0.5) / bin_x))

        grid_index_zyx = (int(iz), int(iy), int(ix))  # 例: (1,2,0)

        # 正規化した“グリッド座標”（セル中心を [0,1] に正規化）
        grid_norm_center_zyx = ((iz + 0.5) / gZ, (iy + 0.5) / gY, (ix + 0.5) / gX)

        # 参考: ROI 内の連続座標（中心を [0,1] に正規化）— 必要なら使ってください
        grid_norm_continuous_zyx = ((zc + 0.5) / Zr, (yc + 0.5) / Hr, (xc + 0.5) / Wr)

        # ====== ③ パッチは ROI から切る ======
        patch, (z1r, y1r, x1r) = self._pad_and_crop(roi, zc, yc, xc, hz, hy, hx, pad_value=pad_val)

        # ====== ④ マスク（ROI座標で計算） ======
        mask = np.zeros_like(patch, dtype=np.uint8)
        if mask_should_draw:
            # 病変の相対座標（パッチ内, ROI基準）
            py = int(np.clip(int((yc_l_r) - y1r), 0, mask.shape[1] - 1))
            px = int(np.clip(int((xc_l_r) - x1r), 0, mask.shape[2] - 1))
            pz = int(np.clip(int((zc_l_r) - z1r), 0, mask.shape[0] - 1))

            # 3D楕円体半径
            if self.r_unit == "mm":
                rx0 = max(1, int(round(self.r / x_sp)))
                ry0 = max(1, int(round(self.r / y_sp)))
                rz0 = max(1, int(round(self.r / z_sp)))
            else:
                rx0 = ry0 = rz0 = max(1, int(round(self.r)))

            Zp = mask.shape[0]
            for dz in range(-rz0, rz0 + 1):
                zc_slice = pz + dz
                if zc_slice < 0 or zc_slice >= Zp:
                    continue
                scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2))
                rx = max(1, int(round(rx0 * scale)))
                ry = max(1, int(round(ry0 * scale)))
                rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=mask.shape[1:])
                mask[zc_slice, rr, cc] = 1

        # ====== ⑤ ラベル処理 ======
        targets = None
        label_keys = [k for k in (
            "Left Infraclinoid Internal Carotid Artery",
            "Right Infraclinoid Internal Carotid Artery",
            "Left Supraclinoid Internal Carotid Artery",
            "Right Supraclinoid Internal Carotid Artery",
            "Left Middle Cerebral Artery",
            "Right Middle Cerebral Artery",
            "Anterior Communicating Artery",
            "Left Anterior Cerebral Artery",
            "Right Anterior Cerebral Artery",
            "Left Posterior Communicating Artery",
            "Right Posterior Communicating Artery",
            "Basilar Tip",
            "Other Posterior Circulation",
            "Aneurysm Present",
        ) if k in self.df.columns]
        if label_keys:
            row = self.df.iloc[index]
            targets = row[label_keys].values.astype(np.float32)

        target_aneurysm = [float(row["Aneurysm Present"])]

        # ====== ⑥ リサイズ & 正規化 ======
        Z_out, H_out, W_out = self.out_size_zyx
        patch = self._resize3d(patch.astype(np.float32, copy=False), (Z_out, H_out, W_out), mode="trilinear")
        mask  = self._resize3d(mask.astype(np.float32),           (Z_out, H_out, W_out), mode="nearest")

        if mod == 'CTA':
            patch = self._vl_normalize(patch, v_low=-100, v_high=600)
        else:
            patch = self._vl_normalize(patch, v_low=vmin_mr, v_high=vmax_mr)

        # LR flip（targets がある場合のみ左右入替）
        if np.random.rand() < self.p_flip:
            if plane=='AXIAL':
                patch = patch[:,:,::-1]
                mask  = mask[:,:,::-1]
            elif plane=='SAGITTAL':
                patch = patch[::-1,:,:]
                mask  = mask[::-1,:,:]
            elif plane=='CORONAL':
                patch = patch[:,:,::-1]
                mask  = mask[:,:,::-1]
            else:
                raise ValueError(f'{plane} is not valid')
            if targets is not None:
                # Infraclinoid Internal Carotid Artery
                targets[0], targets[1] = targets[1], targets[0]
                # Supraclinoid Internal Carotid Artery
                targets[2], targets[3] = targets[3], targets[2]
                # Middle Cerebral Artery
                targets[4], targets[5] = targets[5], targets[4]
                # Anterior Cerebral Artery
                targets[7], targets[8] = targets[8], targets[7]
                # Posterior Communicating Artery
                targets[9], targets[10] = targets[10], targets[9]

        if self.transform is not None:
            patch = patch.transpose(1, 2, 0) # HWZ
            mask = mask.transpose(1, 2, 0)   # HWZ
            data = self.transform(image=patch, mask=mask)
            patch = data["image"].transpose(2, 0, 1) # ZHW
            mask = data["mask"].transpose(2, 0, 1)   # ZHW

        patch = patch.astype(np.float32) / 255.

        # ====== ⑦ 返却（座標はシリーズ座標で維持）=======
        center_series = (int(zc + z0), int(yc + y0), int(xc + x0))
        start_series  = (int(z1r + z0), int(y1r + y0), int(x1r + x0))

        out = {
            "image": torch.from_numpy(patch).float(),
            "mask": torch.from_numpy(mask).float(),
            "series_id": sid,
            "center_zyx": center_series,
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "patch_index_start_zyx": start_series,
            "mode": self.mode,
            "used_lesion_crop": bool(mask_should_draw),
            "target_site": torch.tensor(targets).float() if targets is not None else None,
            "target_aneurysm": torch.tensor(target_aneurysm).float(),
            "plane": plane,
            "grid_index_zyx": grid_index_zyx,  # (iz, iy, ix) in [0..g?-1]
            "grid_norm_center_zyx": torch.Tensor(grid_norm_center_zyx),      # 各軸 ∈ (0,1]
            "grid_norm_continuous_zyx": torch.Tensor(grid_norm_continuous_zyx),  # 各軸 ∈ (0,1]
        }
        return out


class RSNAPatchDatasetV3(Dataset):
    """
    1ケース=1サンプル。ROI から1つの3Dパッチを返す “ランダム/病変追従” データセット。

    ★ 特別ルール:
      - patch_size_mm と volume_size_mm が（各軸で）同じなら、パッチでの再クロップをスキップし、
        ROI 全体をそのままパッチとして扱う（高速パス）。

    返す dict（主要キー）:
      image: FloatTensor (Z_out, H_out, W_out)         # 0..1
      mask:  FloatTensor (Z_out, H_out, W_out)         # 0/1
      target_site:           FloatTensor (14,)         # ★パッチ用（パッチ陰性なら全0）
      target_aneurysm:       FloatTensor (1,)          # ★パッチ用（0/1）
      target_site_case:      FloatTensor (14,) | None  # 症例ラベル（左右反転適用後）
      target_aneurysm_case:  FloatTensor (1,)          # 症例ラベル（左右反転適用後）
      center_zyx:            tuple(int,int,int)        # シリーズ座標（パッチ中心）
      patch_index_start_zyx: tuple(int,int,int)        # シリーズ座標（パッチ開始）
      grid_index_zyx:        tuple(int,int,int)        # ROI内の等分グリッドindex
      grid_norm_center_zyx:  FloatTensor (3,)          # ((i+0.5)/n)
      grid_norm_continuous_zyx: FloatTensor (3,)       # ((c+0.5)/ROI_size_vox)
      lesion_inside_roi:     bool | None
      lesion_inside_patch:   bool | None
    """

    target_order = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present",
    ]
    _swap_pairs = [(0,1), (2,3), (4,5), (7,8), (9,10)]  # 左右入替対象

    def __init__(
        self,
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        mode: str,
        volume_size_mm: tuple[float, float, float],
        patch_size_mm: tuple[float, float, float],
        out_size_zyx: tuple[int, int, int],
        r: float = 5.0, r_unit: str = "mm",
        p_lesion_crop: float = 0.7,
        jitter_mm: Optional[tuple[float, float, float]] = None,
        choose_random_lesion_in_train: bool = True,
        p_flip: float = 0.,
        transform: Optional["A.Compose"] = None,
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.volume_size_mm = tuple(map(float, volume_size_mm))
        self.patch_size_mm  = tuple(map(float, patch_size_mm))
        self.out_size_zyx = out_size_zyx
        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_lesion_crop = float(p_lesion_crop)
        self.jitter_mm = jitter_mm
        self.choose_random_lesion_in_train = bool(choose_random_lesion_in_train)
        self.p_flip = float(p_flip)
        self.transform = transform
        self.debug = bool(debug)

    def __len__(self) -> int:
        return len(self.df)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0

    @staticmethod
    def _pad_value_for_mod(mod: str, vmin_mr: float) -> float:
        # CT系は固定近傍で、MR系はデータ駆動の下限を使用（例に合わせて実装）
        return -100.0 if str(mod).upper() in ("CT","CTA","CCTA") else float(vmin_mr)

    @staticmethod
    def _is_same_mm_size(a: Sequence[float], b: Sequence[float], rtol=1e-6, atol=1e-3) -> bool:
        # 各軸でほぼ同じ（数値誤差考慮）
        return all(abs(a[i] - b[i]) <= max(atol, rtol * max(abs(a[i]), abs(b[i]))) for i in range(3))

    @staticmethod
    def crop_roi_mm(
        vol: np.ndarray,               # (Z,H,W)
        x_min, x_max, y_min, y_max, z_min, z_max,
        z_sp: float, y_sp: float, x_sp: float,
        size_mm_zyx: tuple[float, float, float],
        pad_value: float | str = -1024,
    ):
        Z, H, W = vol.shape

        def _center_or_mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if math.isnan(fa) or math.isnan(fb):
                    raise ValueError
                return 0.5 * (fa + fb)
            except Exception:
                return lim / 2.0

        # 中心（voxel, シリーズ座標）
        cx = int(round(_center_or_mid(x_min, x_max, W)))
        cy = int(round(_center_or_mid(y_min, y_max, H)))
        cz = int(round(_center_or_mid(z_min, z_max, Z)))

        # ROI半サイズ[voxel]（偶数幅: 2*hv）
        z_mm, y_mm, x_mm = size_mm_zyx
        hvz = max(1, int(round((z_mm / max(1e-6, z_sp)) / 2.0)))
        hvy = max(1, int(round((y_mm / max(1e-6, y_sp)) / 2.0)))
        hvx = max(1, int(round((x_mm / max(1e-6, x_sp)) / 2.0)))

        z1, z2 = cz - hvz, cz + hvz
        y1, y2 = cy - hvy, cy + hvy
        x1, x2 = cx - hvx, cx + hvx

        # 画像内の交差
        z1c, z2c = max(0, z1), min(Z, z2)
        y1c, y2c = max(0, y1), min(H, y2)
        x1c, x2c = max(0, x1), min(W, x2)

        # 出力を pad で初期化
        padv = float(vol.min()) if (pad_value == "min") else float(pad_value)
        roi = np.empty((z2 - z1, y2 - y1, x2 - x1), dtype=vol.dtype)
        roi.fill(padv)

        # 転写（memmapに優しい順序）
        dz0 = z1c - z1; dy0 = y1c - y1; dx0 = x1c - x1
        sub = vol[..., x1c:x2c]
        sub = sub[z1c:z2c, y1c:y2c]
        roi[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub

        return roi, (int(z1), int(y1), int(x1))  # ROI と “元座標での開始点（シリーズ座標）”

    @staticmethod
    def _pad_and_crop(
        vol: np.ndarray, zc: int, yc: int, xc: int,
        hz: int, hy: int, hx: int,
        pad_value: float = 0.0
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        Z, H, W = vol.shape
        # パッチは奇数幅（2*hz+1）
        z1, z2 = zc - hz, zc + hz + 1
        y1, y2 = yc - hy, yc + hy + 1
        x1, x2 = xc - hx, xc + hx + 1

        z1c, z2c = max(0, z1), min(Z, z2)
        y1c, y2c = max(0, y1), min(H, y2)
        x1c, x2c = max(0, x1), min(W, x2)

        out = np.empty((z2 - z1, y2 - y1, x2 - x1), dtype=vol.dtype)
        out.fill(float(pad_value))

        dz0 = z1c - z1; dy0 = y1c - y1; dx0 = x1c - x1
        sub = vol[..., x1c:x2c]
        sub = sub[z1c:z2c, y1c:y2c]
        out[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub

        return out, (int(max(z1, 0)), int(max(y1, 0)), int(max(x1, 0)))

    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int, int, int], mode: str) -> np.ndarray:
        Z, H, W = vol_zyx.shape
        Zo, Ho, Wo = out_zyx
        zoom_factors = (Zo / max(1, Z), Ho / max(1, H), Wo / max(1, W))
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low: float, v_high: float) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        arr = (arr - v_low) / (v_high - v_low + 1e-6)
        arr = (arr * 255).astype(np.uint8)  # 0..255 にしてから /255. で Float 0..1
        return arr

    # ---------- crop policy ----------
    @staticmethod
    def _randint_inclusive(a: int, b: int) -> int:
        return int(np.random.randint(a, b + 1)) if a <= b else int(a)

    def _choose_random_center_inside(self, Z: int, H: int, W: int, hz: int, hy: int, hx: int) -> tuple[int, int, int]:
        zc = self._randint_inclusive(hz, max(hz, Z - 1 - hz))
        yc = self._randint_inclusive(hy, max(hy, H - 1 - hy))
        xc = self._randint_inclusive(hx, max(hx, W - 1 - hx))
        return zc, yc, xc

    def _choose_center_with_lesion(self, zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx, jitter_vox):
        jz, jy, jx = jitter_vox
        zc_min, zc_max = zc_l - hz, zc_l + hz
        yc_min, yc_max = yc_l - hy, yc_l + hy
        xc_min, xc_max = xc_l - hx, xc_l + hx

        # jitter を考慮
        zc_rng = (max(zc_min, zc_l - jz), min(zc_max, zc_l + jz))
        yc_rng = (max(yc_min, yc_l - jy), min(yc_max, yc_l + jy))
        xc_rng = (max(xc_min, xc_l - jx), min(xc_max, xc_l + jx))

        # さらに「パッチが画内に収まる」よう制限
        zc_rng = (max(zc_rng[0], hz), min(zc_rng[1], Z - 1 - hz))
        yc_rng = (max(yc_rng[0], hy), min(yc_rng[1], H - 1 - hy))
        xc_rng = (max(xc_rng[0], hx), min(xc_rng[1], W - 1 - hx))

        if zc_rng[0] > zc_rng[1]:
            zc = int(np.clip(zc_l, hz, Z - 1 - hz))
        else:
            zc = self._randint_inclusive(int(math.ceil(zc_rng[0])), int(math.floor(zc_rng[1])))

        if yc_rng[0] > yc_rng[1]:
            yc = int(np.clip(yc_l, hy, H - 1 - hy))
        else:
            yc = self._randint_inclusive(int(math.ceil(yc_rng[0])), int(math.floor(yc_rng[1])))

        if xc_rng[0] > xc_rng[1]:
            xc = int(np.clip(xc_l, hx, W - 1 - hx))
        else:
            xc = self._randint_inclusive(int(math.ceil(xc_rng[0])), int(math.floor(xc_rng[1])))

        return zc, yc, xc

    def _choose_center_with_lesion_nojitter(self, zc_l, yc_l, xc_l, Z, H, W, hz, hy, hx):
        zc_rng = (zc_l - hz, zc_l + hz)
        yc_rng = (yc_l - hy, yc_l + hy)
        xc_rng = (xc_l - hx, xc_l + hx)

        zc_lo, zc_hi = max(zc_rng[0], hz), min(zc_rng[1], Z - 1 - hz)
        yc_lo, yc_hi = max(yc_rng[0], hy), min(yc_rng[1], H - 1 - hy)
        xc_lo, xc_hi = max(xc_rng[0], hx), min(xc_rng[1], W - 1 - hx)

        zc = int(np.clip(zc_l, hz, Z - 1 - hz)) if zc_lo > zc_hi else self._randint_inclusive(int(math.ceil(zc_lo)), int(math.floor(zc_hi)))
        yc = int(np.clip(yc_l, hy, H - 1 - hy)) if yc_lo > yc_hi else self._randint_inclusive(int(math.ceil(yc_lo)), int(math.floor(yc_hi)))
        xc = int(np.clip(xc_l, hx, W - 1 - hx)) if xc_lo > xc_hi else self._randint_inclusive(int(math.ceil(xc_lo)), int(math.floor(xc_hi)))
        return zc, yc, xc

    @staticmethod
    def _pick_sop_value(loc_row: pd.Series) -> Optional[str]:
        if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]):
            return str(loc_row["SOPInstanceUID"])
        return None

    # ---------- main ----------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = str(row["SeriesInstanceUID"])
        npy_path: str = row["file_name"]
        sorted_files: Sequence[str] = row.get("sorted_files", [])
        mod = str(row.get("Modality", ""))
        plane = str(row.get("OrientationLabel", ""))

        vmin_mr = float(row.get("intensity_p0", 0))
        vmax_mr = float(row.get("intensity_p100", 1))

        # spacing
        z_sp: float = float(row.get("z_spacing", 1.0))
        y_sp, x_sp = self._parse_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)))

        # lesion rows
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        has_lesion = len(loc_rows) > 0

        # load memmap
        vol = np.load(npy_path, mmap_mode="r")  # (Z,H,W)
        pad_val = self._pad_value_for_mod(mod, vmin_mr)

        # ====== ① ROI ======
        x_min, x_max = row.get("x1", np.nan), row.get("x2", np.nan)
        y_min, y_max = row.get("y1", np.nan), row.get("y2", np.nan)
        z_min, z_max = row.get("z1", np.nan), row.get("z2", np.nan)

        roi, (z0, y0, x0) = self.crop_roi_mm(
            vol,
            x_min, x_max, y_min, y_max, z_min, z_max,
            z_sp, y_sp, x_sp,
            size_mm_zyx=self.volume_size_mm,
            pad_value=pad_val,
        )
        Zr, Hr, Wr = roi.shape

        # ====== ② “同サイズ”の判定（高速パス可否） ======
        same_size_mm = self._is_same_mm_size(self.volume_size_mm, self.patch_size_mm)

        # ====== ③ lesion の座標（必要なら） ======
        # （高速パスでもマスク生成に使うので、必要時だけ計算）
        do_lesion = has_lesion and (np.random.rand() < self.p_lesion_crop)
        mask_should_draw = False
        zc_l_r = yc_l_r = xc_l_r = None  # ROI座標の病変中心
        if do_lesion:
            if self.mode == "train" and self.choose_random_lesion_in_train and len(loc_rows) > 0:
                loc_row = loc_rows.sample(n=1, random_state=None).iloc[0]
            else:
                loc_row = loc_rows.iloc[0]
            bad_loc = False
            sop_value = self._pick_sop_value(loc_row)
            if sop_value:
                names = [p.split("/")[-1][:-4] for p in (sorted_files or [])]
                try:
                    zc_l = names.index(sop_value)
                    coords = ast.literal_eval(loc_row.get("coordinates", "{}"))
                    xc_l = int(coords.get("x", 0)); yc_l = int(coords.get("y", 0))
                except ValueError:
                    bad_loc = True
            else:
                try:
                    coords = ast.literal_eval(loc_row["coordinates"])
                    zc_l = int(coords["z"]); xc_l = int(coords["x"]); yc_l = int(coords["y"])
                except Exception:
                    bad_loc = True
            if not bad_loc:
                zc_l_r, yc_l_r, xc_l_r = zc_l - z0, yc_l - y0, xc_l - x0
                mask_should_draw = (0 <= zc_l_r < Zr) and (0 <= yc_l_r < Hr) and (0 <= xc_l_r < Wr)

        # ====== ④ パッチの作り方 ======
        if same_size_mm:
            # ---- 高速パス：パッチでの再クロップをしない ----
            patch = roi  # そのまま
            # ROI 中心（偶数幅でも floor 側に寄るが一貫性を優先）
            zc = Zr // 2; yc = Hr // 2; xc = Wr // 2
            z1r = y1r = x1r = 0  # ROIの先頭がパッチ先頭
            # グリッドは 1x1x1
            grid_index_zyx = (0, 0, 0)
            grid_norm_center_zyx = (0.5, 0.5, 0.5)
            grid_norm_continuous_zyx = ((zc + 0.5) / max(1, Zr),
                                        (yc + 0.5) / max(1, Hr),
                                        (xc + 0.5) / max(1, Wr))
        else:
            # ---- 従来パス：ROI 内で中心を選び、パッチでクロップ ----
            # パッチ半径と jitter
            z_mm, y_mm, x_mm = self.patch_size_mm
            hz = max(0, int(round((z_mm / max(1e-6, z_sp)) / 2.0)))
            hy = max(0, int(round((y_mm / max(1e-6, y_sp)) / 2.0)))
            hx = max(0, int(round((x_mm / max(1e-6, x_sp)) / 2.0)))

            if self.jitter_mm is None:
                jz, jy, jx = hz, hy, hx
            else:
                jz = int(round(self.jitter_mm[0] / max(1e-6, z_sp)))
                jy = int(round(self.jitter_mm[1] / max(1e-6, y_sp)))
                jx = int(round(self.jitter_mm[2] / max(1e-6, x_sp)))

            if mask_should_draw and (zc_l_r is not None):
                if np.random.rand() < 0.7:
                    jz_s, jy_s, jx_s = hz//3, hy//3, hx//3
                    zc, yc, xc = self._choose_center_with_lesion(
                        zc_l_r, yc_l_r, xc_l_r, Zr, Hr, Wr, hz, hy, hx, (jz_s, jy_s, jx_s)
                    )
                else:
                    zc, yc, xc = self._choose_center_with_lesion_nojitter(
                        zc_l_r, yc_l_r, xc_l_r, Zr, Hr, Wr, hz, hy, hx
                    )
            else:
                zc, yc, xc = self._choose_random_center_inside(Zr, Hr, Wr, hz, hy, hx)

            # ROI内グリッド
            gZ = max(1, int(round(self.volume_size_mm[0] / self.patch_size_mm[0])))
            gY = max(1, int(round(self.volume_size_mm[1] / self.patch_size_mm[1])))
            gX = max(1, int(round(self.volume_size_mm[2] / self.patch_size_mm[2])))
            bin_z = Zr / float(gZ); bin_y = Hr / float(gY); bin_x = Wr / float(gX)
            iz = min(gZ - 1, int((zc + 0.5) / bin_z))
            iy = min(gY - 1, int((yc + 0.5) / bin_y))
            ix = min(gX - 1, int((xc + 0.5) / bin_x))
            grid_index_zyx = (int(iz), int(iy), int(ix))
            grid_norm_center_zyx = ((iz + 0.5) / gZ, (iy + 0.5) / gY, (ix + 0.5) / gX)
            grid_norm_continuous_zyx = ((zc + 0.5) / max(1, Zr),
                                        (yc + 0.5) / max(1, Hr),
                                        (xc + 0.5) / max(1, Wr))

            patch, (z1r, y1r, x1r) = self._pad_and_crop(roi, zc, yc, xc, hz, hy, hx, pad_value=pad_val)

        # ====== ⑤ マスク（ROI座標で計算） ======
        mask = np.zeros_like(patch, dtype=np.uint8)
        lesion_in_patch = False
        if mask_should_draw and (zc_l_r is not None):
            # 同サイズ高速パスでは z1r=y1r=x1r=0 なので “ROI 内かどうか” == “パッチ内かどうか”
            lesion_in_patch = (
                (z1r <= zc_l_r < z1r + patch.shape[0]) and
                (y1r <= yc_l_r < y1r + patch.shape[1]) and
                (x1r <= xc_l_r < x1r + patch.shape[2])
            )
            if lesion_in_patch:
                py = int(np.clip(int(yc_l_r - y1r), 0, patch.shape[1] - 1))
                px = int(np.clip(int(xc_l_r - x1r), 0, patch.shape[2] - 1))
                pz = int(np.clip(int(zc_l_r - z1r), 0, patch.shape[0] - 1))

                if self.r_unit == "mm":
                    rx0 = max(1, int(round(self.r / max(1e-6, x_sp))))
                    ry0 = max(1, int(round(self.r / max(1e-6, y_sp))))
                    rz0 = max(1, int(round(self.r / max(1e-6, z_sp))))
                else:
                    rx0 = ry0 = rz0 = max(1, int(round(self.r)))

                Zp = mask.shape[0]
                for dz in range(-rz0, rz0 + 1):
                    zc_slice = pz + dz
                    if not (0 <= zc_slice < Zp):
                        continue
                    scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2))
                    rx = max(1, int(round(rx0 * scale)))
                    ry = max(1, int(round(ry0 * scale)))
                    rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=mask.shape[1:])
                    mask[zc_slice, rr, cc] = 1

        # ====== ⑥ ラベル（症例 / パッチ） ======
        label_keys = [k for k in self.target_order if k in self.df.columns]
        targets_case = None
        if label_keys:
            targets_case = row[label_keys].values.astype(np.float32, copy=True)
        target_ane_case = np.array([float(row.get("Aneurysm Present", 0.0))], dtype=np.float32)

        # ====== ⑦ リサイズ & 正規化 ======
        Z_out, H_out, W_out = self.out_size_zyx
        patch = self._resize3d(patch.astype(np.float32, copy=False), (Z_out, H_out, W_out), mode="trilinear")
        mask  = self._resize3d(mask.astype(np.float32),               (Z_out, H_out, W_out), mode="nearest")

        if str(mod).upper() in ("CT","CTA","CCTA"):
            patch = self._vl_normalize(patch, v_low=-100, v_high=600)
        else:
            patch = self._vl_normalize(patch, v_low=vmin_mr, v_high=vmax_mr)

        # ====== ⑧ 左右反転（画像/マスク/症例ラベル） ======
        did_flip = False
        if np.random.rand() < self.p_flip:
            did_flip = True
            if plane in ("AXIAL", "CORONAL"):
                patch = patch[:, :, ::-1]
                mask  = mask[:,  :, ::-1]
            elif plane in ("SAGITTAL", "SAGITAL"):
                patch = patch[::-1, :, :]
                mask  = mask[ ::-1, :, :]
            else:
                raise ValueError(f'Invalid plane: {plane}')
            if targets_case is not None:
                for a, b in self._swap_pairs:
                    targets_case[a], targets_case[b] = targets_case[b], targets_case[a]

        # Albumentations
        if self.transform is not None:
            phwz = patch.transpose(1, 2, 0).astype(np.uint8)       # 0..255
            mhwz = (mask > 0.5).astype(np.uint8).transpose(1, 2, 0)
            data = self.transform(image=phwz, mask=mhwz)
            patch = data["image"].transpose(2, 0, 1).astype(np.float32)
            mask  = data["mask"].transpose(2, 0, 1).astype(np.float32)

        # 0..1 へ
        patch = patch.astype(np.float32) / 255.0
        mask  = (mask > 0.5).astype(np.float32)

        # ====== ⑨ ターゲットを最終決定（パッチ陰性なら0） ======
        is_positive_patch = bool((mask > 0.5).any()) and bool(mask_should_draw and lesion_in_patch)
        if targets_case is not None:
            targets_patch = targets_case.copy()
            if not is_positive_patch:
                targets_patch[:] = 0.0
        else:
            targets_patch = None
        target_ane_patch = np.array([1.0 if is_positive_patch else 0.0], dtype=np.float32)

        # ====== ⑩ 返却（座標はシリーズ座標で維持） ======
        # center は same_size_mm の場合 ROI 中心、従来パスでは選んだ中心
        center_series = (int((Zr // 2) + z0), int((Hr // 2) + y0), int((Wr // 2) + x0)) if same_size_mm \
                        else (int(zc + z0), int(yc + y0), int(xc + x0))
        start_series  = (int(0 + z0), int(0 + y0), int(0 + x0)) if same_size_mm \
                        else (int(z1r + z0), int(y1r + y0), int(x1r + x0))

        out = {
            "image": torch.from_numpy(patch).float(),
            "mask":  torch.from_numpy(mask).float(),
            "series_id": sid,
            "center_zyx": center_series,
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "patch_index_start_zyx": start_series,
            "mode": self.mode,
            "used_lesion_crop": bool(mask_should_draw),

            # --- パッチ用ターゲット（陰性なら全0） ---
            "target_site": torch.tensor(targets_patch).float() if targets_patch is not None else None,
            "target_aneurysm": torch.from_numpy(target_ane_patch).float(),

            # --- 症例ターゲット（左右反転後の一貫表現） ---
            "target_site_case": torch.tensor(targets_case).float() if targets_case is not None else None,
            "target_aneurysm_case": torch.from_numpy(target_ane_case).float(),

            "plane": plane,

            # --- ROI基準のグリッド情報 ---
            "grid_index_zyx": (0,0,0) if same_size_mm else grid_index_zyx,
            "grid_norm_center_zyx": torch.tensor((0.5,0.5,0.5) if same_size_mm else grid_norm_center_zyx, dtype=torch.float32),
            "grid_norm_continuous_zyx": torch.tensor(
                (( (Zr//2)+0.5)/max(1,Zr), ((Hr//2)+0.5)/max(1,Hr), ((Wr//2)+0.5)/max(1,Wr))
                if same_size_mm else grid_norm_continuous_zyx,
                dtype=torch.float32
            ),

            # --- デバッグ補助 ---
            "lesion_inside_roi": bool(mask_should_draw) if has_lesion else False,
            "lesion_inside_patch": bool(is_positive_patch) if has_lesion else False,
            "did_flip": bool(did_flip),
            "same_size_fast_path": bool(same_size_mm),
        }
        return out


class RSNAROIDatasetV1(Dataset):
    """
    ROI だけを切り出すデータセット（パッチ再クロップはしない）。

    - 入力: 1ケース=1サンプル
    - 出力: ROI を `out_size_zyx` にリサイズした画像/マスク。
    - マスクは **target（血管部位 13種 + Aneurysm Present）に対応する多クラス**（one-hot チャンネル形式）を返す。
      背景は全チャネル 0。Aneurysm Present チャンネルは空間クラス（部位チャネル）の **union** に、
      さらに **座標に site 指定のない注釈も union に含める**。
    - さらに、特徴マップサイズ `map_size_zyx` へ any-pooling で縮約した多クラス・マスク `mask_map`
      （形状: (K, Z_map, H_map, W_map)）を返せる。

    返す dict（主要キー）:
      image: FloatTensor (Z_out, H_out, W_out)                 # 0..1
      mask:  FloatTensor (K, Z_out, H_out, W_out)              # 多クラス one-hot（K=14; 最後が Aneurysm Present=union）
      mask_map: FloatTensor (K, Z_map, H_map, W_map) | None    # any-pooling による縮約

      # 座標メタ（ROI データセットなので、座標をすべて返します）
      roi_start_series_zyx / roi_end_series_* / roi_shape_vox_zyx / roi_size_mm_zyx
      roi_center_roi_zyx / roi_center_series_zyx / roi_to_out_scale_zyx / out_size_zyx / map_size_zyx
      lesion_centers_roi_zyx / lesion_centers_series_zyx / lesion_centers_norm_roi_zyx / lesion_centers_out_zyx

      target_site:           FloatTensor (14,)         # 症例ラベル（左右反転適用後）
      target_aneurysm:       FloatTensor (1,)          # ROI 内に描画があれば 1、それ以外は 0
      target_site_case:      FloatTensor (14,) | None
      target_aneurysm_case:  FloatTensor (1,)

      grid_index_zyx=(0,0,0), grid_norm_* は互換出力
    """

    target_order = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present",
    ]
    # 空間チャネル（部位）は Aneurysm Present を除外
    spatial_class_names: list[str] = target_order[:-1]
    # 返却用のクラス一覧（Aneurysm Present を含む全14）
    classes: list[str] = target_order
    aneurysm_idx: int = len(target_order) - 1

    _swap_pairs = [(0,1), (2,3), (4,5), (7,8), (9,10)]  # 左右入替対象（空間クラスの index 基準）

    def __init__(
        self,
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        mode: str,
        volume_size_mm: tuple[float, float, float],   # ROI の物理サイズ (Z,Y,X) [mm]
        out_size_zyx: tuple[int, int, int],           # ネット入力 (Z,Y,X)
        map_size_zyx: Optional[tuple[int, int, int]] = None,  # 特徴マップ (Z,Y,X) へ縮約した mask を作る場合に指定
        r: float = 5.0,
        r_unit: str = "mm",
        p_flip: float = 0.0,
        align_plane: bool = False,
        cta_min: float = -100,
        cta_max: float = 600,
        transform: Optional["A.Compose"] = None,
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.volume_size_mm = tuple(map(float, volume_size_mm))
        self.out_size_zyx = out_size_zyx
        self.map_size_zyx = map_size_zyx
        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_flip = float(p_flip)
        self.align_plane = align_plane
        self.cta_min = cta_min
        self.cta_max = cta_max
        self.transform = transform  # 3D多クラスへの厳密適用は未対応（下記注記）
        self.debug = bool(debug)

    def __len__(self) -> int:
        return len(self.df)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 0.5, 0.5

    def _pad_value_for_mod(self, mod: str, vmin_mr: float) -> float:
        return self.cta_min if str(mod).upper() in ("CT","CTA","CCTA") else float(vmin_mr)

    @staticmethod
    def crop_roi_mm(
        vol: np.ndarray,               # (Z,H,W)
        x_min, x_max, y_min, y_max, z_min, z_max,
        z_sp: float, y_sp: float, x_sp: float,
        size_mm_zyx: tuple[float, float, float],
        pad_value: float | str = -1024,
    ):
        Z, H, W = vol.shape

        def _center_or_mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if math.isnan(fa) or math.isnan(fb):
                    raise ValueError
                return 0.5 * (fa + fb)
            except Exception:
                return lim / 2.0

        # 中心（voxel, シリーズ座標）
        cx = int(round(_center_or_mid(x_min, x_max, W)))
        cy = int(round(_center_or_mid(y_min, y_max, H)))
        cz = int(round(_center_or_mid(z_min, z_max, Z)))

        # ROI 半サイズ[voxel]
        z_mm, y_mm, x_mm = size_mm_zyx
        hvz = max(1, int(round((z_mm / max(1e-6, z_sp)) / 2.0)))
        hvy = max(1, int(round((y_mm / max(1e-6, y_sp)) / 2.0)))
        hvx = max(1, int(round((x_mm / max(1e-6, x_sp)) / 2.0)))

        z1, z2 = cz - hvz, cz + hvz
        y1, y2 = cy - hvy, cy + hvy
        x1, x2 = cx - hvx, cx + hvx

        # 画像内の交差
        z1c, z2c = max(0, z1), min(Z, z2)
        y1c, y2c = max(0, y1), min(H, y2)
        x1c, x2c = max(0, x1), min(W, x2)

        # 出力を pad で初期化
        padv = float(vol.min()) if (pad_value == "min") else float(pad_value)
        roi = np.empty((z2 - z1, y2 - y1, x2 - x1), dtype=vol.dtype)
        roi.fill(padv)

        # 転写（memmap に優しい順序）
        dz0 = z1c - z1; dy0 = y1c - y1; dx0 = x1c - x1
        sub = vol[..., x1c:x2c]
        sub = sub[z1c:z2c, y1c:y2c]
        roi[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub

        return roi, (int(z1), int(y1), int(x1))

    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int, int, int], mode: str) -> np.ndarray:
        Z, H, W = vol_zyx.shape
        Zo, Ho, Wo = out_zyx
        zoom_factors = (Zo / max(1, Z), Ho / max(1, H), Wo / max(1, W))
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low: float, v_high: float) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        arr = (arr - v_low) / (v_high - v_low + 1e-6)
        arr = (arr * 255).astype(np.uint8)  # 0..255 にしてから /255.
        return arr

    @staticmethod
    def _pick_sop_value(loc_row: pd.Series) -> Optional[str]:
        if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]):
            return str(loc_row["SOPInstanceUID"])
        return None

    def _site_index_from_loc(self, loc_row: pd.Series) -> Optional[int]:
        """loc_row からサイト名/one-hot を読み取り、**空間クラス**内の index を返す（Aneurysm Present は除外）。
        優先順:
          1) loc_row["site"/"Site"/"label"...] が spatial_class_names に一致
          2) loc_row に spatial_class_names のカラムがあり ==1 の最初の index
          3) targets_case のうち spatial 部分が 1 の最初の index
        見つからなければ None。
        """
        # 1) 文字列での指定
        loc_name = loc_row['location']
        idx = self.target_order.index(loc_name)
        return idx

    @staticmethod
    def _mask_to_map_any_multi(mask_kzyx: np.ndarray, map_zyx: tuple[int,int,int]) -> np.ndarray:
        # mask: (K,Z,H,W) 0/1
        t = torch.from_numpy(mask_kzyx.astype(np.float32))[None, ...]  # (1,K,Z,H,W)
        pooled = F.adaptive_max_pool3d(t, output_size=map_zyx)         # (1,K,MZ,MH,MW)
        return pooled.squeeze(0).numpy().astype(np.float32)

    def _apply_albu_slicewise(
        self,
        img_zyx_u8: np.ndarray,                 # (Z, H, W) uint8
        mask_kzyx_u8: Optional[np.ndarray],     # (K, Z, H, W) uint8 0/255 or None
    ):
        """
        Albumentations を 2D スライスに対して、同一パラメータで Z 全体へ適用。
        transform は Compose でも ReplayCompose でもOK。
        """
        if self.transform is None:
            return img_zyx_u8, mask_kzyx_u8

        # Compose を ReplayCompose に包む（同一replayを使うため）
        t = self.transform
        if not isinstance(t, A.ReplayCompose):
            t = A.ReplayCompose(t.transforms)

        Z, H, W = img_zyx_u8.shape
        out_img = np.empty_like(img_zyx_u8)
        out_mask = None
        K = 0
        if mask_kzyx_u8 is not None:
            K = mask_kzyx_u8.shape[0]
            out_mask = np.zeros_like(mask_kzyx_u8)

        # 中央スライスでパラメータをサンプリング
        mid = Z // 2
        base = t(image=img_zyx_u8[mid], masks=[mask_kzyx_u8[k, mid] for k in range(K)] if K else None)
        replay = base["replay"]

        # 全スライスに同一replayを適用
        for z in range(Z):
            res = A.ReplayCompose.replay(
                replay,
                image=img_zyx_u8[z],
                masks=[mask_kzyx_u8[k, z] for k in range(K)] if K else None
            )
            out_img[z] = res["image"]
            if K:
                # Albumentations は mask を uint8 で返すので 0/255 → 0/1 に戻しやすい
                for k in range(K):
                    out_mask[k, z] = res["masks"][k]

        return out_img, out_mask


    # ---------- main ----------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = str(row["SeriesInstanceUID"])
        npy_path: str = row["file_name"]
        sorted_files: Sequence[str] = row.get("sorted_files", [])
        mod = str(row.get("Modality", ""))
        plane = str(row.get("OrientationLabel", ""))
        mod_enc = row["ModalityEncoded"]
        plane_enc = row["OrientationLabelEncoded"]

        vmin_mr = float(row.get("intensity_p0", 0))
        vmax_mr = float(row.get("intensity_p100", 1))

        # spacing
        z_sp: float = float(row.get("z_spacing", 1.0))
        y_sp, x_sp = self._parse_pixel_spacing(row.get("PixelSpacing", (0.5, 0.5)))

        # lesion rows（このシリーズに紐づく "全て" を使用）
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        has_lesion = len(loc_rows) > 0

        # load memmap
        vol = np.load(npy_path, mmap_mode="r")  # (Z,H,W)
        pad_val = self._pad_value_for_mod(mod, vmin_mr)

        # ====== ① ROI ======
        x_min, x_max = row.get("x1", np.nan), row.get("x2", np.nan)
        y_min, y_max = row.get("y1", np.nan), row.get("y2", np.nan)
        z_min, z_max = row.get("z1", np.nan), row.get("z2", np.nan)

        roi, (z0, y0, x0) = self.crop_roi_mm(
            vol,
            x_min, x_max, y_min, y_max, z_min, z_max,
            z_sp, y_sp, x_sp,
            size_mm_zyx=self.volume_size_mm,
            pad_value=pad_val,
        )
        Zr, Hr, Wr = roi.shape

        roi_center_roi = (int(Zr // 2), int(Hr // 2), int(Wr // 2))
        
        # ====== ② 症例ラベル（左右反転用にも使う） ======
        label_keys = [k for k in self.target_order if k in self.df.columns]
        targets_case = None
        if label_keys:
            targets_case = row[label_keys].values.astype(np.float32, copy=True)
        
        # ====== ③ 多クラス・マスク（ROI 座標で描画：シリーズ中の "全ての" 座標を反映） ======
        K_sp = len(self.spatial_class_names)
        mask_roi_sp = np.zeros((K_sp, Zr, Hr, Wr), dtype=np.uint8)
        union_roi = np.zeros((Zr, Hr, Wr), dtype=np.uint8)  # site 不明も含めた union
        draw_any = False
        lesion_centers_roi: list[tuple[int,int,int]] = []
        lesion_centers_series: list[tuple[int,int,int]] = []

        if has_lesion:
            # sorted_files から z を引くための辞書を先に用意
            name_to_z = None
            if sorted_files:
                names = [p.split("/")[-1][:-4] for p in (sorted_files or [])]
                name_to_z = {nm: i for i, nm in enumerate(names)}

            for _, loc_row in loc_rows.iterrows():
                # 1) 座標の抽出
                zc_l = xc_l = yc_l = None
                sop_value = self._pick_sop_value(loc_row)
                if sop_value and name_to_z is not None and sop_value in name_to_z:
                    zc_l = int(name_to_z[sop_value])
                    coords = ast.literal_eval(loc_row.get("coordinates", "{}")) if ("coordinates" in loc_row.index) else {}
                    xc_l = int(coords.get("x", coords.get("X", 0)))
                    yc_l = int(coords.get("y", coords.get("Y", 0)))
                elif "coordinates" in loc_row.index and pd.notna(loc_row["coordinates"]):
                    try:
                        coords = ast.literal_eval(loc_row["coordinates"])  # {x,y,z}
                        zc_l = int(coords.get("z", coords.get("Z", 0)))
                        xc_l = int(coords.get("x", coords.get("X", 0)))
                        yc_l = int(coords.get("y", coords.get("Y", 0)))
                    except Exception:
                        zc_l = xc_l = yc_l = None
                if zc_l is None or xc_l is None or yc_l is None:
                    continue  # 座標が不明

                # 2) ROI 座標へ変換
                zc_l_r, yc_l_r, xc_l_r = zc_l - z0, yc_l - y0, xc_l - x0
                inside = (0 <= zc_l_r < Zr) and (0 <= yc_l_r < Hr) and (0 <= xc_l_r < Wr)
                if not inside:
                    continue

                # 3) 半径（mm/px）→ voxel
                if self.r_unit == "mm":
                    rx0 = max(1, int(round(self.r / max(1e-6, x_sp))))
                    ry0 = max(1, int(round(self.r / max(1e-6, y_sp))))
                    rz0 = max(1, int(round(self.r / max(1e-6, z_sp))))
                else:
                    rx0 = ry0 = rz0 = max(1, int(round(self.r)))

                py = int(np.clip(int(yc_l_r), 0, Hr - 1))
                px = int(np.clip(int(xc_l_r), 0, Wr - 1))
                pz = int(np.clip(int(zc_l_r), 0, Zr - 1))

                # 4) union とクラス別に描画
                for dz in range(-rz0, rz0 + 1):
                    zc_slice = pz + dz
                    if not (0 <= zc_slice < Zr):
                        continue
                    scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2)) if rz0 > 0 else 1.0
                    rx = max(1, int(round(rx0 * scale)))
                    ry = max(1, int(round(ry0 * scale)))
                    rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=(Hr, Wr))
                    union_roi[zc_slice, rr, cc] = 1

                k = self._site_index_from_loc(loc_row)
                if k is not None:
                    for dz in range(-rz0, rz0 + 1):
                        zc_slice = pz + dz
                        if not (0 <= zc_slice < Zr):
                            continue
                        scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2)) if rz0 > 0 else 1.0
                        rx = max(1, int(round(rx0 * scale)))
                        ry = max(1, int(round(ry0 * scale)))
                        rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=(Hr, Wr))
                        mask_roi_sp[k, zc_slice, rr, cc] = 1
                else:
                    pass
                    # print("site not found")

                draw_any = True
                lesion_centers_roi.append((int(pz), int(py), int(px)))
                lesion_centers_series.append((int(pz + z0), int(py + y0), int(px + x0)))

        # ====== ④ リサイズ & 正規化 ======
        Z_out, H_out, W_out = self.out_size_zyx
        img_out = self._resize3d(roi.astype(np.float32, copy=False), (Z_out, H_out, W_out), mode="trilinear")
        # 空間クラスをリサイズ
        mask_kzo_sp = np.zeros((K_sp, Z_out, H_out, W_out), dtype=np.uint8)
        for k in range(K_sp):
            if mask_roi_sp[k].any():
                mask_kzo_sp[k] = self._resize3d(mask_roi_sp[k].astype(np.float32), (Z_out, H_out, W_out), mode="nearest").astype(np.uint8)
        # union（site不明も含む）をリサイズ
        union_out = self._resize3d(union_roi.astype(np.float32), (Z_out, H_out, W_out), mode="nearest").astype(np.uint8)
        # Aneurysm Present = 空間クラスの union OR union_out
        aneurysm_ch = (np.maximum(union_out, mask_kzo_sp.any(axis=0).astype(np.uint8)))[None, ...]
        # 全チャネル結合
        mask_kzo = np.concatenate([mask_kzo_sp, aneurysm_ch], axis=0)

        if str(mod).upper() in ("CT","CTA","CCTA"):
            img_out = self._vl_normalize(img_out, v_low=self.cta_min, v_high=self.cta_max)
        else:
            img_out = self._vl_normalize(img_out, v_low=vmin_mr, v_high=vmax_mr)
        if self.transform is not None:
            img_aug_u8, mask_aug_u8 = self._apply_albu_slicewise(
                img_out.astype(np.uint8),                      # (Z,H,W)
                (mask_kzo * 255).astype(np.uint8)             # (K,Z,H,W) 0/255
            )
            # 画像はuint8のまま後段の正規化へ、マスクは0/1へ戻す
            img_out = img_aug_u8
            mask_kzo = (mask_aug_u8 > 127).astype(np.uint8)
        # ====== ⑤ 左右反転（画像/マスク/症例ラベル & 座標） ======
        did_flip = False
        if np.random.rand() < self.p_flip:
            did_flip = True
            if plane in ("AXIAL", "CORONAL"):
                img_out = img_out[:, :, ::-1]
                mask_kzo = mask_kzo[:, :, :, ::-1]
                lesion_centers_roi = [(z, y, int(Wr - 1 - x)) for (z, y, x) in lesion_centers_roi]
            elif plane in ("SAGITTAL", "SAGITAL"):
                img_out = img_out[::-1, :, :]
                mask_kzo = mask_kzo[:, ::-1, :, :]
                lesion_centers_roi = [(int(Zr - 1 - z), y, x) for (z, y, x) in lesion_centers_roi]
            else:
                raise ValueError(f'Invalid plane: {plane}')
            # チャンネル（左右）を入れ替え：空間クラスのみ
            for a, b in self._swap_pairs:
                mask_kzo[[a, b]] = mask_kzo[[b, a]]
            if targets_case is not None:
                for a, b in self._swap_pairs:
                    targets_case[a], targets_case[b] = targets_case[b], targets_case[a]
            # series 座標を更新
            lesion_centers_series = [(int(z + z0), int(y + y0), int(x + x0)) for (z, y, x) in lesion_centers_roi]

        # 0..1 & 0/1 化
        img_out = img_out.astype(np.float32) / 255.0
        mask_kzo = (mask_kzo > 0).astype(np.float32)
        if self.align_plane:
            img_out = to_axial(img_out, plane=plane.lower())
            mask_kzo = to_axial(mask_kzo, plane=plane.lower(), channels='first')
        
        # ====== ⑥ 症例/ROI ラベル ======
        target_ane_case = np.array([float(row.get("Aneurysm Present", 0.0))], dtype=np.float32)
        is_positive_roi = bool(mask_kzo.any()) and bool(draw_any)
        target_ane_patch = np.array([1.0 if is_positive_roi else 0.0], dtype=np.float32)

        # ====== ⑦ マスクを map_size まで潰す（any-pooling, 多クラス） ======
        mask_map = None
        if self.map_size_zyx is not None:
            mask_map = self._mask_to_map_any_multi(mask_kzo.astype(np.uint8), self.map_size_zyx)
            mask_map = torch.from_numpy(mask_map.astype(np.float32))

        

        # ====== ⑨ 返却 ======
        out = {
            "image": torch.from_numpy(img_out).float(),                       # (Z,H,W)
            #"mask":  torch.from_numpy(mask_kzo).float(),                      # (K=14,Z,H,W)
            "mask_map": mask_map,                                            # (K,MZ,MH,MW) or None

            # --- メタ/座標情報 ---
            "series_id": sid,
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            # "out_size_zyx": tuple(int(x) for x in self.out_size_zyx),
            "map_size_zyx": tuple(int(x) for x in self.map_size_zyx) if self.map_size_zyx is not None else None,
            
            # ターゲット
            "target_site": torch.tensor(targets_case).float() if targets_case is not None else None,
            "target_aneurysm": torch.from_numpy(target_ane_patch).float(),
            
            "plane": plane,
            "plane_encoded": torch.tensor(plane_enc, dtype=torch.long),
            "modality_encoded": torch.tensor(mod_enc, dtype=torch.long)
        }
        return out


class RSNAROIDatasetV2(Dataset):
    """
    ROI だけを切り出すデータセット（パッチ再クロップはしない）。

    - 入力: 1ケース=1サンプル
    - 出力: ROI を `out_size_zyx` にリサイズした画像/マスク。
    - マスクは **target（血管部位 13種 + Aneurysm Present）に対応する多クラス**（one-hot チャンネル形式）を返す。
      背景は全チャネル 0。Aneurysm Present チャンネルは空間クラス（部位チャネル）の **union** に、
      さらに **座標に site 指定のない注釈も union に含める**。
    - さらに、特徴マップサイズ `map_size_zyx` へ any-pooling で縮約した多クラス・マスク `mask_map`
      （形状: (K, Z_map, H_map, W_map)）を返せる。

    返す dict（主要キー）:
      image: FloatTensor (Z_out, H_out, W_out)                 # 0..1
      mask:  FloatTensor (K, Z_out, H_out, W_out)              # 多クラス one-hot（K=14; 最後が Aneurysm Present=union）
      mask_map: FloatTensor (K, Z_map, H_map, W_map) | None    # any-pooling による縮約

      ほかメタはコード末尾参照
    """

    target_order = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present",
    ]
    spatial_class_names: list[str] = target_order[:-1]
    classes: list[str] = target_order
    aneurysm_idx: int = len(target_order) - 1

    # 左右入替対象（空間クラス index 基準）
    _swap_pairs = [(0, 1), (2, 3), (4, 5), (7, 8), (9, 10)]

    def __init__(
        self,
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        mode: str,
        volume_size_mm: tuple[float, float, float],   # ROI 物理サイズ (Z,Y,X) [mm]
        out_size_zyx: tuple[int, int, int],           # ネット入力 (Z,Y,X)
        map_size_zyx: Optional[tuple[int, int, int]] = None,  # 特徴マップ (Z,Y,X)
        r: float = 5.0,
        r_unit: str = "mm",
        p_flip: float = 0.0,
        align_plane: bool = False,
        transform: Optional["A.Compose"] = None,
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.volume_size_mm = tuple(map(float, volume_size_mm))
        self.out_size_zyx = out_size_zyx
        self.map_size_zyx = map_size_zyx
        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_flip = float(p_flip)
        self.align_plane = align_plane
        self.transform = transform
        self.debug = bool(debug)

    def __len__(self) -> int:
        return len(self.df)

    # ---------- plane helpers ----------
    @staticmethod
    def _plane_axis_maps(plane: str) -> tuple[tuple[int, int, int], tuple[str, str, str]]:
        """
        series座標 (Z,Y,X) を、保存済み vol(N,H,W) の軸へ写すためのインデックス/ラベルを返す。
        想定（ネイティブ配列形状）:
          axial   : (N,H,W)=(Z,Y,X)
          coronal : (N,H,W)=(Y,X,Z)
          sagittal: (N,H,W)=(X,Y,Z)
        戻り値: (idx_map, sp_map)
          idx_map: (iz,iy,ix)->(in,ih,iw) 例 coronal -> (1,2,0)
          sp_map : ("z","y","x")->(n,h,w) に対応 例 coronal -> ("y","x","z")
        """
        p = str(plane).lower()
        if p == "axial":
            return (0, 1, 2), ("z", "y", "x")
        elif p == "coronal":
            return (1, 2, 0), ("y", "x", "z")
        elif p == "sagittal":
            return (2, 1, 0), ("x", "y", "z")
        raise ValueError(f"unknown plane: {plane}")

    @staticmethod
    def _series_dims_zyx_from_vol(vol_shape: tuple[int, int, int], plane: str) -> tuple[int, int, int]:
        """
        vol.shape=(N,H,W) と plane から、シリーズ座標 (Z,Y,X) 方向の長さを返す。
        axial   : (Z,Y,X)=(N,H,W)
        coronal : (Z,Y,X)=(W,N,H)
        sagittal: (Z,Y,X)=(W,H,N)
        """
        N, H, W = vol_shape
        p = str(plane).upper()
        if p == "AXIAL":
            return (N, H, W)
        elif p == "CORONAL":
            return (W, N, H)
        elif p in ("SAGITTAL", "SAGITAL"):
            return (W, H, N)
        else:
            raise ValueError(f"Invalid plane: {plane}")

    # ---------- basic helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[", "").replace("]", "").replace("(", "").replace(")", "").replace(" ", "")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0

    @staticmethod
    def _pad_value_for_mod(mod: str, vmin_mr: float) -> float:
        return -100.0 if str(mod).upper() in ("CT", "CTA", "CCTA") else float(vmin_mr)

    # ---------- plane-aware crop (配列は転置しない) ----------
    @staticmethod
    def _crop_roi_mm_plane(
        vol_nhw: np.ndarray,                         # (N,H,W) ネイティブ平面のまま
        center_zyx_vox: tuple[float, float, float],  # シリーズ座標 (Z,Y,X) の中心[voxel]
        size_mm_zyx: tuple[float, float, float],     # ROI 物理サイズ (Z,Y,X)[mm]
        spacing_zyx_mm: tuple[float, float, float],  # (z_sp,y_sp,x_sp)[mm]
        plane: str,
        pad_value: float,
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        idx_map, sp_map = RSNAROIDatasetV1._plane_axis_maps(plane)

        # center を (N,H,W) へ並べ替え
        cz, cy, cx = map(float, center_zyx_vox)
        ctr_list = [cz, cy, cx]
        cn = int(round(ctr_list[idx_map[0]]))
        ch = int(round(ctr_list[idx_map[1]]))
        cw = int(round(ctr_list[idx_map[2]]))

        # spacing を (N,H,W) へ
        zsp, ysp, xsp = spacing_zyx_mm
        sp_dict = {"z": float(zsp), "y": float(ysp), "x": float(xsp)}
        sn = sp_dict[sp_map[0]]
        sh = sp_dict[sp_map[1]]
        sw = sp_dict[sp_map[2]]

        # mm サイズを (N,H,W) へ
        zmm, ymm, xmm = map(float, size_mm_zyx)
        mm_list = [zmm, ymm, xmm]
        mn = mm_list[idx_map[0]]
        mh = mm_list[idx_map[1]]
        mw = mm_list[idx_map[2]]

        # 半サイズ [voxel]
        hvn = max(1, int(round((mn / max(1e-6, sn)) / 2.0)))
        hvh = max(1, int(round((mh / max(1e-6, sh)) / 2.0)))
        hvw = max(1, int(round((mw / max(1e-6, sw)) / 2.0)))

        N, H, W = vol_nhw.shape
        n1, n2 = cn - hvn, cn + hvn
        h1, h2 = ch - hvh, ch + hvh
        w1, w2 = cw - hvw, cw + hvw

        n1c, n2c = max(0, n1), min(N, n2)
        h1c, h2c = max(0, h1), min(H, h2)
        w1c, w2c = max(0, w1), min(W, w2)

        roi = np.empty((n2 - n1, h2 - h1, w2 - w1), dtype=vol_nhw.dtype)
        roi.fill(float(pad_value))

        dn0, dh0, dw0 = n1c - n1, h1c - h1, w1c - w1
        roi[dn0:dn0 + (n2c - n1c), dh0:dh0 + (h2c - h1c), dw0:dw0 + (w2c - w1c)] = \
            vol_nhw[n1c:n2c, h1c:h2c, w1c:w2c]

        # origin を (N,H,W)->(Z,Y,X) へ逆写像
        nhw_origin = [n1, h1, w1]
        origin_series = [None, None, None]
        origin_series[idx_map[0]] = nhw_origin[0]
        origin_series[idx_map[1]] = nhw_origin[1]
        origin_series[idx_map[2]] = nhw_origin[2]
        origin_series_zyx = (int(origin_series[0]), int(origin_series[1]), int(origin_series[2]))

        return roi, origin_series_zyx

    # ---------- misc ops ----------
    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int, int, int], mode: str) -> np.ndarray:
        Z, H, W = vol_zyx.shape
        Zo, Ho, Wo = out_zyx
        zoom_factors = (Zo / max(1, Z), Ho / max(1, H), Wo / max(1, W))
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low: float, v_high: float) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        arr = (arr - v_low) / (v_high - v_low + 1e-6)
        arr = (arr * 255).astype(np.uint8)  # 0..255
        return arr

    @staticmethod
    def _pick_sop_value(loc_row: pd.Series) -> Optional[str]:
        if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]):
            return str(loc_row["SOPInstanceUID"])
        return None

    def _site_index_from_loc(self, loc_row: pd.Series) -> Optional[int]:
        loc_name = loc_row['location']
        try:
            return self.target_order.index(loc_name)
        except ValueError:
            return None

    @staticmethod
    def _mask_to_map_any_multi(mask_kzyx: np.ndarray, map_zyx: tuple[int, int, int]) -> np.ndarray:
        t = torch.from_numpy(mask_kzyx.astype(np.float32))[None, ...]  # (1,K,Z,H,W)
        pooled = F.adaptive_max_pool3d(t, output_size=map_zyx)         # (1,K,MZ,MH,MW)
        return pooled.squeeze(0).numpy().astype(np.float32)

    def _apply_albu_slicewise(
        self,
        img_zyx_u8: np.ndarray,                 # (Z,H,W) uint8
        mask_kzyx_u8: Optional[np.ndarray],     # (K,Z,H,W) uint8 or None
    ):
        if self.transform is None:
            return img_zyx_u8, mask_kzyx_u8

        t = self.transform
        if not isinstance(t, A.ReplayCompose):
            t = A.ReplayCompose(t.transforms)

        Z, H, W = img_zyx_u8.shape
        out_img = np.empty_like(img_zyx_u8)
        out_mask = None
        K = 0
        if mask_kzyx_u8 is not None:
            K = mask_kzyx_u8.shape[0]
            out_mask = np.zeros_like(mask_kzyx_u8)

        mid = Z // 2
        base = t(image=img_zyx_u8[mid], masks=[mask_kzyx_u8[k, mid] for k in range(K)] if K else None)
        replay = base["replay"]

        for z in range(Z):
            res = A.ReplayCompose.replay(
                replay,
                image=img_zyx_u8[z],
                masks=[mask_kzyx_u8[k, z] for k in range(K)] if K else None
            )
            out_img[z] = res["image"]
            if K:
                for k in range(K):
                    out_mask[k, z] = res["masks"][k]

        return out_img, out_mask

    # ---------- main ----------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = str(row["SeriesInstanceUID"])
        npy_path: str = row["file_name"]
        sorted_files: Sequence[str] = row.get("sorted_files", [])
        mod = str(row.get("Modality", ""))
        plane = str(row.get("OrientationLabel", ""))  # "AXIAL"/"CORONAL"/"SAGITTAL"

        vmin_mr = float(row.get("intensity_p0", 0))
        vmax_mr = float(row.get("intensity_p100", 1))

        # spacing (シリーズ座標系の (Z,Y,X))
        z_sp: float = float(row.get("z_spacing", 1.0))
        y_sp, x_sp = self._parse_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)))

        # lesion rows
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        has_lesion = len(loc_rows) > 0

        # load memmap (ネイティブ平面の (N,H,W) で保存されている前提)
        vol = np.load(npy_path, mmap_mode="r")
        pad_val = self._pad_value_for_mod(mod, vmin_mr)

        # ====== ① ROI（planeネイティブのままクロップ） ======
        x_min, x_max = row.get("x1", np.nan), row.get("x2", np.nan)
        y_min, y_max = row.get("y1", np.nan), row.get("y2", np.nan)
        z_min, z_max = row.get("z1", np.nan), row.get("z2", np.nan)

        def _mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if math.isnan(fa) or math.isnan(fb):
                    raise ValueError
                return 0.5 * (fa + fb)
            except Exception:
                return lim / 2.0

        # bbox 未指定時のデフォルト中心＝シリーズ座標の中央
        Z_len, Y_len, X_len = self._series_dims_zyx_from_vol(vol.shape, plane)
        cz = _mid(z_min, z_max, Z_len)
        cy = _mid(y_min, y_max, Y_len)
        cx = _mid(x_min, x_max, X_len)

        roi, (z0, y0, x0) = self._crop_roi_mm_plane(
            vol_nhw=vol,
            center_zyx_vox=(cz, cy, cx),
            size_mm_zyx=self.volume_size_mm,           # (Z,Y,X)[mm]
            spacing_zyx_mm=(z_sp, y_sp, x_sp),         # (Z,Y,X)[mm]
            plane=plane,
            pad_value=pad_val,
        )
        Zr, Hr, Wr = roi.shape
        roi_center_roi = (int(Zr // 2), int(Hr // 2), int(Wr // 2))

        # ====== ② 症例ラベル ======
        label_keys = [k for k in self.target_order if k in self.df.columns]
        targets_case = None
        if label_keys:
            targets_case = row[label_keys].values.astype(np.float32, copy=True)

        # ====== ③ マスク描画（ROI座標） ======
        K_sp = len(self.spatial_class_names)
        mask_roi_sp = np.zeros((K_sp, Zr, Hr, Wr), dtype=np.uint8)
        union_roi = np.zeros((Zr, Hr, Wr), dtype=np.uint8)
        draw_any = False
        lesion_centers_roi: list[tuple[int, int, int]] = []
        lesion_centers_series: list[tuple[int, int, int]] = []

        if has_lesion:
            name_to_z = None
            if sorted_files:
                names = [p.split("/")[-1][:-4] for p in (sorted_files or [])]
                name_to_z = {nm: i for i, nm in enumerate(names)}

            for _, loc_row in loc_rows.iterrows():
                zc_l = xc_l = yc_l = None
                sop_value = self._pick_sop_value(loc_row)
                if sop_value and name_to_z is not None and sop_value in name_to_z:
                    zc_l = int(name_to_z[sop_value])
                    coords = ast.literal_eval(loc_row.get("coordinates", "{}")) if ("coordinates" in loc_row.index) else {}
                    xc_l = int(coords.get("x", coords.get("X", 0)))
                    yc_l = int(coords.get("y", coords.get("Y", 0)))
                elif "coordinates" in loc_row.index and pd.notna(loc_row["coordinates"]):
                    try:
                        coords = ast.literal_eval(loc_row["coordinates"])  # {x,y,z}
                        zc_l = int(coords.get("z", coords.get("Z", 0)))
                        xc_l = int(coords.get("x", coords.get("X", 0)))
                        yc_l = int(coords.get("y", coords.get("Y", 0)))
                    except Exception:
                        zc_l = xc_l = yc_l = None
                if zc_l is None or xc_l is None or yc_l is None:
                    continue

                # ROI 座標へ
                zc_l_r, yc_l_r, xc_l_r = zc_l - z0, yc_l - y0, xc_l - x0
                inside = (0 <= zc_l_r < Zr) and (0 <= yc_l_r < Hr) and (0 <= xc_l_r < Wr)
                if not inside:
                    continue

                # 半径（mm or px）→ voxel
                if self.r_unit == "mm":
                    rx0 = max(1, int(round(self.r / max(1e-6, x_sp))))
                    ry0 = max(1, int(round(self.r / max(1e-6, y_sp))))
                    rz0 = max(1, int(round(self.r / max(1e-6, z_sp))))
                else:
                    rx0 = ry0 = rz0 = max(1, int(round(self.r)))

                py = int(np.clip(int(yc_l_r), 0, Hr - 1))
                px = int(np.clip(int(xc_l_r), 0, Wr - 1))
                pz = int(np.clip(int(zc_l_r), 0, Zr - 1))

                # union
                for dz in range(-rz0, rz0 + 1):
                    zc_slice = pz + dz
                    if not (0 <= zc_slice < Zr):
                        continue
                    scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2)) if rz0 > 0 else 1.0
                    rx = max(1, int(round(rx0 * scale)))
                    ry = max(1, int(round(ry0 * scale)))
                    rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=(Hr, Wr))
                    union_roi[zc_slice, rr, cc] = 1

                # クラス別
                k = self._site_index_from_loc(loc_row)
                if k is not None:
                    for dz in range(-rz0, rz0 + 1):
                        zc_slice = pz + dz
                        if not (0 <= zc_slice < Zr):
                            continue
                        scale = math.sqrt(max(0.0, 1.0 - (dz / float(rz0)) ** 2)) if rz0 > 0 else 1.0
                        rx = max(1, int(round(rx0 * scale)))
                        ry = max(1, int(round(ry0 * scale)))
                        rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=(Hr, Wr))
                        mask_roi_sp[k, zc_slice, rr, cc] = 1

                draw_any = True
                lesion_centers_roi.append((int(pz), int(py), int(px)))
                lesion_centers_series.append((int(pz + z0), int(py + y0), int(px + x0)))

        # ====== ④ リサイズ & 正規化 ======
        Z_out, H_out, W_out = self.out_size_zyx
        img_out = self._resize3d(roi.astype(np.float32, copy=False), (Z_out, H_out, W_out), mode="trilinear")

        mask_kzo_sp = np.zeros((len(self.spatial_class_names), Z_out, H_out, W_out), dtype=np.uint8)
        for k in range(len(self.spatial_class_names)):
            if mask_roi_sp[k].any():
                mask_kzo_sp[k] = self._resize3d(mask_roi_sp[k].astype(np.float32), (Z_out, H_out, W_out), mode="nearest").astype(np.uint8)

        union_out = self._resize3d(union_roi.astype(np.float32), (Z_out, H_out, W_out), mode="nearest").astype(np.uint8)
        aneurysm_ch = (np.maximum(union_out, mask_kzo_sp.any(axis=0).astype(np.uint8)))[None, ...]
        mask_kzo = np.concatenate([mask_kzo_sp, aneurysm_ch], axis=0)

        if str(mod).upper() in ("CT", "CTA", "CCTA"):
            img_out = self._vl_normalize(img_out, v_low=-100, v_high=600)
        else:
            img_out = self._vl_normalize(img_out, v_low=vmin_mr, v_high=vmax_mr)

        if self.transform is not None:
            img_aug_u8, mask_aug_u8 = self._apply_albu_slicewise(
                img_out.astype(np.uint8),
                (mask_kzo * 255).astype(np.uint8)
            )
            img_out = img_aug_u8
            mask_kzo = (mask_aug_u8 > 127).astype(np.uint8)

        # ====== ⑤ 左右/前後の確率的反転（平面ごとに定義） ======
        did_flip = False
        if np.random.rand() < self.p_flip:
            did_flip = True
            if plane in ("AXIAL", "CORONAL"):
                img_out = img_out[:, :, ::-1]
                mask_kzo = mask_kzo[:, :, :, ::-1]
                lesion_centers_roi = [(z, y, int(Wr - 1 - x)) for (z, y, x) in lesion_centers_roi]
            elif plane in ("SAGITTAL", "SAGITAL"):
                img_out = img_out[::-1, :, :]
                mask_kzo = mask_kzo[:, ::-1, :, :]
                lesion_centers_roi = [(int(Zr - 1 - z), y, x) for (z, y, x) in lesion_centers_roi]
            else:
                raise ValueError(f"Invalid plane: {plane}")

            # 左右クラス入替
            for a, b in self._swap_pairs:
                mask_kzo[[a, b]] = mask_kzo[[b, a]]
            if targets_case is not None:
                for a, b in self._swap_pairs:
                    targets_case[a], targets_case[b] = targets_case[b], targets_case[a]

            lesion_centers_series = [(int(z + z0), int(y + y0), int(x + x0))
                                     for (z, y, x) in lesion_centers_roi]

        # 0..1 & 0/1
        img_out = img_out.astype(np.float32) / 255.0
        mask_kzo = (mask_kzo > 0).astype(np.float32)

        # 必要なら axial に整列
        if self.align_plane:
            img_out = to_axial(img_out, plane=plane.lower())
            mask_kzo = to_axial(mask_kzo, plane=plane.lower(), channels='first')

        # ---- torch 変換前に負ストライド解消（重要）----
        img_out = np.ascontiguousarray(img_out)
        # mask_kzo を使うなら同様に:
        # mask_kzo = np.ascontiguousarray(mask_kzo)

        # ====== ⑥ 症例/ROI ラベル ======
        target_ane_case = np.array([float(row.get("Aneurysm Present", 0.0))], dtype=np.float32)
        is_positive_roi = bool(mask_kzo.any()) and bool(draw_any)
        target_ane_patch = np.array([1.0 if is_positive_roi else 0.0], dtype=np.float32)

        # ====== ⑦ マスクを map_size まで any-pooling ======
        mask_map = None
        if self.map_size_zyx is not None:
            mask_map = self._mask_to_map_any_multi(mask_kzo.astype(np.uint8), self.map_size_zyx)
            mask_map = torch.from_numpy(mask_map.astype(np.float32))

        # ====== ⑨ 返却 ======
        out = {
            "image": torch.from_numpy(img_out).float(),                         # (Z,H,W)
            # "mask":  torch.from_numpy(np.ascontiguousarray(mask_kzo)).float(), # (K,Z,H,W) を返すなら有効化
            "mask_map": mask_map,                                              # (K,MZ,MH,MW) or None
            "series_id": sid,
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "map_size_zyx": tuple(int(x) for x in self.map_size_zyx) if self.map_size_zyx is not None else None,
            "target_site": torch.tensor(targets_case).float() if targets_case is not None else None,
            "target_aneurysm": torch.from_numpy(target_ane_patch).float(),
            "plane": plane,
        }
        return out


class RSNASlidingBagDataset(Dataset):
    """
    __len__ : シリーズ数
    __getitem__(i) : i番目シリーズの全パッチをまとめて返す

    返す dict:
      image: FloatTensor (P, Z_out, H_out, W_out)  # 0..1
      mask:  FloatTensor (P, Z_out, H_out, W_out)  # 0/1
      # --- ターゲット（パッチ単位：マスク対応） ---
      target_site_patch:      FloatTensor (P, 14)
      target_aneurysm_patch:  FloatTensor (P, 1)
      # --- ターゲット（症例＝シリーズ単位） ---
      target_site_case:       FloatTensor (14,)
      target_aneurysm_case:   FloatTensor (1,)
      # --- 位置情報（0-1 正規化） ---
      patch_pos_norm_zyx:       FloatTensor (P, 3)  # ROI連続座標（中心/ROIサイズ）
      patch_pos_norm_zyx_grid:  FloatTensor (P, 3)  # グリッド正規化（i/(n-1); n==1→0.5）
      # メタ
      series_id, spacing_zyx_mm, plane, grid_counts_zyx,
      grid_indices_zyx (P,3), patch_starts_zyx (P,3)
    """

    target_order = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present",
    ]
    _swap_pairs = [(0,1), (2,3), (4,5), (7,8), (9,10)]  # 左右入替

    def __init__(
        self,
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        mode: str,
        volume_size_mm: tuple[float, float, float],
        patch_size_mm: tuple[float, float, float],
        out_size_zyx: tuple[int, int, int],
        overlap_size_mm: Union[float, tuple[float, float, float]] = 0.0,
        r: float = 5.0,
        r_unit: str = "mm",
        p_flip: float = 0.0,
        transform: Optional[A.Compose] = None,         # Albumentations
        same_aug_per_bag: bool = False,                # ★ 追加：バッグ内で同一Aug
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.volume_size_mm = tuple(map(float, volume_size_mm))
        self.patch_size_mm  = tuple(map(float, patch_size_mm))
        self.out_size_zyx = out_size_zyx

        if isinstance(overlap_size_mm, (int, float)):
            overlap_size_mm = (float(overlap_size_mm),) * 3
        self.overlap_size_mm = tuple(
            min(float(o), float(w) - 1e-6) for o, w in zip(overlap_size_mm, self.patch_size_mm)
        )

        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_flip = float(p_flip)
        self.debug = debug

        self.transform = transform
        self.same_aug_per_bag = bool(same_aug_per_bag)

        # Albumentations: same_aug_per_bag を使う場合は ReplayCompose にラップ
        self.replay_tfm: Optional[A.ReplayCompose] = None
        if self.transform is not None and self.same_aug_per_bag:
            if isinstance(self.transform, A.ReplayCompose):
                self.replay_tfm = self.transform
            elif isinstance(self.transform, A.Compose):
                self.replay_tfm = A.ReplayCompose(self.transform.transforms)
            else:
                self.replay_tfm = A.ReplayCompose([self.transform])

        self.site_to_idx = {name: i for i, name in enumerate(self.target_order)}

    def __len__(self) -> int:
        return len(self.df)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[","").replace("]","").replace("(","").replace(")","").replace(" ","")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0

    @staticmethod
    def _safe_spacing_1d(v, default=1.0):
        try:
            x = float(v)
        except Exception:
            return float(default)
        if not np.isfinite(x) or x <= 0:
            return float(default)
        return float(x)

    def _safe_pixel_spacing(self, v, default=(0.5, 0.5)):
        dy_raw, dx_raw = self._parse_pixel_spacing(v)
        dy = self._safe_spacing_1d(dy_raw, default=default[0])
        dx = self._safe_spacing_1d(dx_raw, default=default[1])
        return dy, dx

    @staticmethod
    def _pad_value_for_mod(mod: str, vmin_mr: float) -> float:
        return -100.0 if str(mod).upper() in ("CT","CTA","CCTA") else float(vmin_mr)

    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int,int,int], mode: str) -> np.ndarray:
        Z,H,W = vol_zyx.shape
        Zo,Ho,Wo = out_zyx
        zoom_factors = (Zo/max(1,Z), Ho/max(1,H), Wo/max(1,W))
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low: float, v_high: float) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        return (arr - v_low) / (v_high - v_low + 1e-6)

    @staticmethod
    def _build_targets_14(row: pd.Series, keys_14: Sequence[str]) -> np.ndarray:
        return np.array([float(row[k]) if (k in row.index and pd.notna(row[k])) else 0.0
                         for k in keys_14], dtype=np.float32)

    @staticmethod
    def _roi_center_from_bbox(xmin, xmax, ymin, ymax, zmin, zmax, W, H, Z):
        def mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if np.isnan(fa) or np.isnan(fb): raise ValueError
                return 0.5*(fa+fb)
            except Exception:
                return lim/2.0
        cx = int(round(mid(xmin, xmax, W)))
        cy = int(round(mid(ymin, ymax, H)))
        cz = int(round(mid(zmin, zmax, Z)))
        return cz, cy, cx

    @staticmethod
    def _crop_from_start(vol: np.ndarray, z0: int, y0: int, x0: int,
                         sz: int, sy: int, sx: int, pad_value: float):
        Z,H,W = vol.shape
        z1,y1,x1 = z0,y0,x0
        z2,y2,x2 = z0+sz, y0+sy, x0+sx

        z1c,z2c = max(0,z1), min(Z,z2)
        y1c,y2c = max(0,y1), min(H,y2)
        x1c,x2c = max(0,x1), min(W,x2)

        out = np.empty((sz,sy,sx), dtype=vol.dtype)
        out.fill(float(pad_value))

        dz0 = z1c - z1; dy0 = y1c - y1; dx0 = x1c - x1
        sub = vol[..., x1c:x2c]
        sub = sub[z1c:z2c, y1c:y2c]
        out[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub
        return out

    @staticmethod
    def _nwin(L, W, S):
        if L <= W: return 1
        return int(math.floor((L - W) / S) + 1)

    def _match_site_index(self, site_name: str | None) -> int | None:
        if not site_name: return None
        s = site_name.strip().lower()
        for k, i in self.site_to_idx.items():
            if k.lower() == s: return i
        for k, i in self.site_to_idx.items():
            if s in k.lower(): return i
        return None

    # ---------- main ----------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = row["SeriesInstanceUID"]
        npy_path: str = row["file_name"]
        sorted_files: list[str] = row.get("sorted_files", [])
        mod = row.get("Modality", "")
        plane = row.get("OrientationLabel", "")
        vmin_mr = float(row.get("intensity_p0", 0))
        vmax_mr = float(row.get("intensity_p100", 1))

        # spacing（安全化）
        z_sp: float = self._safe_spacing_1d(row.get("z_spacing", 1.0), default=1.0)
        y_sp, x_sp = self._safe_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)), default=(1.0, 1.0))

        # ROI & ストライド（mm -> voxel）
        Vmm = self.volume_size_mm
        Wmm = self.patch_size_mm
        Omm = self.overlap_size_mm
        Smm = (max(1e-6, Wmm[0]-Omm[0]), max(1e-6, Wmm[1]-Omm[1]), max(1e-6, Wmm[2]-Omm[2]))

        half_roi_z = max(1, int(round((Vmm[0]/max(1e-6,z_sp))/2.0)))
        half_roi_y = max(1, int(round((Vmm[1]/max(1e-6,y_sp))/2.0)))
        half_roi_x = max(1, int(round((Vmm[2]/max(1e-6,x_sp))/2.0)))
        roi_sz = 2*half_roi_z; roi_sy = 2*half_roi_y; roi_sx = 2*half_roi_x

        psz = max(1, int(round(Wmm[0]/max(1e-6,z_sp))))
        psy = max(1, int(round(Wmm[1]/max(1e-6,y_sp))))
        psx = max(1, int(round(Wmm[2]/max(1e-6,x_sp))))
        stz = max(1, int(round(Smm[0]/max(1e-6,z_sp))))
        sty = max(1, int(round(Smm[1]/max(1e-6,y_sp))))
        stx = max(1, int(round(Smm[2]/max(1e-6,x_sp))))

        # ボリューム
        vol = np.load(npy_path, mmap_mode="r")
        Z,H,W = vol.shape

        # ROI 中心（bboxがあれば使用）
        x_min, x_max = row.get("x1", np.nan), row.get("x2", np.nan)
        y_min, y_max = row.get("y1", np.nan), row.get("y2", np.nan)
        z_min, z_max = row.get("z1", np.nan), row.get("z2", np.nan)
        cz, cy, cx = self._roi_center_from_bbox(x_min, x_max, y_min, y_max, z_min, z_max, W, H, Z)

        # ROI 開始座標（シリーズ座標, pad対応）
        pad_val = self._pad_value_for_mod(mod, vmin_mr)
        roi_z0 = int(cz - half_roi_z)
        roi_y0 = int(cy - half_roi_y)
        roi_x0 = int(cx - half_roi_x)

        # ROI 抽出
        roi = self._crop_from_start(vol, roi_z0, roi_y0, roi_x0, roi_sz, roi_sy, roi_sx, pad_value=pad_val)
        
        # パッチ数
        nz = self._nwin(Vmm[0], Wmm[0], Smm[0])
        ny = self._nwin(Vmm[1], Wmm[1], Smm[1])
        nx = self._nwin(Vmm[2], Wmm[2], Smm[2])
        P = nz*ny*nx

        # ---- 病変マスク（ROI座標）----
        mask_roi = np.zeros_like(roi, dtype=np.uint8)
        site_mask_roi = np.zeros((13,)+roi.shape, dtype=np.uint8)
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        for _, loc_row in loc_rows.iterrows():
            site_name = str(loc_row.get("location", "") or "")
            site_idx = self._match_site_index(site_name)
            site_ch = site_idx if (site_idx is not None and site_idx < 13) else None

            sop = str(loc_row["SOPInstanceUID"]) if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]) else None
            if sop and isinstance(sorted_files, (list, tuple)) and len(sorted_files) > 0:
                names = [p.split("/")[-1][:-4] for p in sorted_files]
                try:
                    zc_l = names.index(sop)
                except ValueError:
                    coords = ast.literal_eval(loc_row["coordinates"])
                    zc_l = int(coords.get("z", 0))
            else:
                coords = ast.literal_eval(loc_row["coordinates"])
                zc_l = int(coords.get("z", 0))
            coords = ast.literal_eval(loc_row["coordinates"])
            yc_l = int(coords.get("y", 0)); xc_l = int(coords.get("x", 0))

            pz = int(np.clip(zc_l - roi_z0, 0, roi.shape[0]-1))
            py = int(np.clip(yc_l - roi_y0, 0, roi.shape[1]-1))
            px = int(np.clip(xc_l - roi_x0, 0, roi.shape[2]-1))

            if self.r_unit == "mm":
                rx0 = max(1, int(round(self.r / x_sp)))
                ry0 = max(1, int(round(self.r / y_sp)))
                rz0 = max(1, int(round(self.r / z_sp)))
            else:
                rx0 = ry0 = rz0 = max(1, int(round(self.r)))

            Zr = roi.shape[0]
            for dz in range(-rz0, rz0+1):
                zc_sl = pz + dz
                if not (0 <= zc_sl < Zr): continue
                scale = math.sqrt(max(0.0, 1.0 - (dz/float(rz0))**2))
                rx = max(1, int(round(rx0 * scale)))
                ry = max(1, int(round(ry0 * scale)))
                rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=roi.shape[1:])
                mask_roi[zc_sl, rr, cc] = 1
                if site_ch is not None:
                    site_mask_roi[site_ch, zc_sl, rr, cc] = 1

        # ---- 全パッチ切り出し＆位置ベクトル作成 ----
        Zo,Ho,Wo = self.out_size_zyx
        imgs, msks = [], []
        grid_idx, starts = [], []

        # 位置ベクトル（連続 / グリッド）
        grid_norm_continuous = []  # (P,3) = ROI 内の連続座標: (center / roi_size)
        grid_norm_center     = []  # (P,3) = グリッド中心の正規化: ((i + 0.5)/n)

        def norm_index(i, n):
            return 0.5 if n <= 1 else float(i) / float(n - 1)

        # パッチ target（マスク由来）
        site_vec_list, ane_vec_list = [], []

        do_flip = (self.mode == "train") and (np.random.rand() < self.p_flip)

        # Albumentations の Replay を共有するための一時変数
        replay_params = None

        for iz in range(nz):
            z0 = iz*stz
            for iy in range(ny):
                y0 = iy*sty
                for ix in range(nx):
                    x0 = ix*stx

                    patch = roi[z0:z0+psz, y0:y0+psy, x0:x0+psx]
                    mpatch = mask_roi[z0:z0+psz, y0:y0+psy, x0:x0+psx]
                    mpatch_sites = site_mask_roi[:, z0:z0+psz, y0:y0+psy, x0:x0+psx]

                    # 位置ベクトル（ROI連続 / グリッド）
                    cz_roi = (z0 + 0.5*psz) / max(1, roi_sz)
                    cy_roi = (y0 + 0.5*psy) / max(1, roi_sy)
                    cx_roi = (x0 + 0.5*psx) / max(1, roi_sx)
                    grid_norm_continuous.append([
                        float(np.clip(cz_roi, 0.0, 1.0)),
                        float(np.clip(cy_roi, 0.0, 1.0)),
                        float(np.clip(cx_roi, 0.0, 1.0)),
                    ])

                    # セル中心の正規化（n==1 でも (0+0.5)/1 = 0.5 になる）
                    grid_norm_center.append([
                        float((iz + 0.5) / max(1, nz)),
                        float((iy + 0.5) / max(1, ny)),
                        float((ix + 0.5) / max(1, nx)),
                    ])

                    # 出力解像度へ
                    patch = self._resize3d(patch.astype(np.float32, copy=False), (Zo,Ho,Wo), mode="trilinear")
                    mpatch = self._resize3d(mpatch.astype(np.float32), (Zo,Ho,Wo), mode="nearest")

                    # 0..1 正規化
                    if str(mod).upper() in ("CT","CTA","CCTA"):
                        patch = self._vl_normalize(patch, v_low=-100, v_high=600)
                    else:
                        patch = self._vl_normalize(patch, v_low=vmin_mr, v_high=vmax_mr)
                    
                    # ---- Albumentations: バッグ内で同一乱数を使う ----
                    if self.replay_tfm is not None:
                        phwz = (patch * 255.0).astype(np.uint8).transpose(1,2,0)  # HWZ
                        mhwz = (mpatch > 0.5).astype(np.uint8).transpose(1,2,0)   # HWZ, uint8
                        if replay_params is None:
                            data = self.replay_tfm(image=phwz, mask=mhwz)
                            replay_params = data["replay"]
                        else:
                            data = A.ReplayCompose.replay(replay_params, image=phwz, mask=mhwz)
                        patch  = (data["image"].transpose(2,0,1).astype(np.float32) / 255.0)
                        mpatch =  data["mask"].transpose(2,0,1).astype(np.float32)

                    elif self.transform is not None:
                        phwz = (patch * 255.0).astype(np.uint8).transpose(1,2,0)
                        mhwz = (mpatch > 0.5).astype(np.uint8).transpose(1,2,0)
                        data = self.transform(image=phwz, mask=mhwz)
                        patch  = (data["image"].transpose(2,0,1).astype(np.float32) / 255.0)
                        mpatch =  data["mask"].transpose(2,0,1).astype(np.float32)

                    imgs.append(patch.astype(np.float32))
                    msks.append(mpatch.astype(np.float32))
                    grid_idx.append((iz,iy,ix))
                    starts.append((roi_z0+z0, roi_y0+y0, roi_x0+x0))

                    # パッチターゲット（マスク対応）
                    present = float((mpatch > 0.5).any())
                    site_vec = np.zeros(14, dtype=np.float32)
                    present_sites = (mpatch_sites.reshape(13, -1).sum(axis=1) > 0).astype(np.float32)
                    site_vec[:13] = present_sites
                    site_vec[13] = present
                    site_vec_list.append(site_vec)
                    ane_vec_list.append([present])

        images = np.stack(imgs, axis=0).astype(np.float32)     # (P, Z,H,W)
        masks  = np.stack(msks, axis=0).astype(np.float32)
        site_vec_patch = np.stack(site_vec_list, axis=0)       # (P,14)
        ane_vec_patch  = np.asarray(ane_vec_list, dtype=np.float32)  # (P,1)
        grid_norm_continuous = np.asarray(grid_norm_continuous, dtype=np.float32)  # (P,3)
        grid_norm_center     = np.asarray(grid_norm_center,     dtype=np.float32)  # (P,3)

        
        # 内部フリップ適用（画像/マスク/ラベル/位置）
        if do_flip:
            if plane in ("AXIAL","CORONAL"):
                images = images[:, :, :, ::-1].copy()
                masks  = masks[:,  :, :, ::-1].copy()
            elif plane in ("SAGITTAL","SAGITAL"):
                images = images[:, ::-1, :, :].copy()
                masks  = masks[:,  ::-1, :, :].copy()

        # 症例ターゲット
        targets14_case = self._build_targets_14(row, self.target_order)
        if do_flip:
            t = targets14_case.copy()
            for a,b in self._swap_pairs: t[a], t[b] = t[b], t[a]
            targets14_case = t
            # パッチ側の部位も左右入替
            tpatch = site_vec_patch.copy()
            for a,b in self._swap_pairs:
                tpatch[:, a], tpatch[:, b] = tpatch[:, b], tpatch[:, a]
            site_vec_patch = tpatch

        target_ane_case = np.array([targets14_case[-1]], dtype=np.float32)

        out = {
            "image": torch.from_numpy(images).float(),                 # (P, Z,H,W)
            "mask":  torch.from_numpy(masks).float(),                  # (P, Z,H,W)
            "series_id": sid,
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "plane": plane,
            "grid_counts_zyx": (int(nz), int(ny), int(nx)),
            "grid_indices_zyx": torch.tensor(np.array(grid_idx), dtype=torch.long),
            "patch_starts_zyx": torch.tensor(np.array(starts), dtype=torch.long),

            # --- パッチターゲット（マスク対応） ---
            "target_site_patch": torch.from_numpy(site_vec_patch).float(),   # (P,14)
            "target_aneurysm_patch": torch.from_numpy(ane_vec_patch).float(),# (P,1)

            # --- 症例ターゲット ---
            "target_site": torch.from_numpy(targets14_case).float(),    # (14,)
            "target_aneurysm": torch.from_numpy(target_ane_case).float(),# (1,)

            # --- 位置情報（0-1 正規化, Z,Y,X） ---
            "grid_norm_center_zyx": torch.from_numpy(grid_norm_center).float(),             # (P,3) = ((i+0.5)/n)
            "grid_norm_continuous_zyx": torch.from_numpy(grid_norm_continuous).float(),     # (P,3) = ROI 内連続座標
        }
        return out


class RSNASlidingBagDatasetV2(Dataset):
    """
    __len__ : シリーズ数
    __getitem__(i) : i番目シリーズの全パッチをまとめて返す

    返す dict:
      image: FloatTensor (P, Z_out, H_out, W_out)  # 0..1
      mask:  FloatTensor (P, Z_out, H_out, W_out)  # 0/1
      # --- ターゲット（パッチ単位：マスク対応） ---
      target_site_patch:      FloatTensor (P, 14)
      target_aneurysm_patch:  FloatTensor (P, 1)
      # --- ターゲット（症例＝シリーズ単位） ---
      target_site_case:       FloatTensor (14,)
      target_aneurysm_case:   FloatTensor (1,)
      # --- 位置情報（0-1 正規化） ---
      patch_pos_norm_zyx:       FloatTensor (P, 3)  # ROI連続座標（中心/ROIサイズ）
      patch_pos_norm_zyx_grid:  FloatTensor (P, 3)  # グリッド正規化（i/(n-1); n==1→0.5）
      # メタ
      series_id, spacing_zyx_mm, plane, grid_counts_zyx,
      grid_indices_zyx (P,3), patch_starts_zyx (P,3)
    """

    target_order = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present",
    ]
    _swap_pairs = [(0,1), (2,3), (4,5), (7,8), (9,10)]  # 左右入替

    def __init__(
        self,
        df: pd.DataFrame,
        df_loc: pd.DataFrame,
        mode: str,
        volume_size_mm: tuple[float, float, float],
        patch_size_mm: tuple[float, float, float],
        out_size_zyx: tuple[int, int, int],
        overlap_size_mm: Union[float, tuple[float, float, float]] = 0.0,
        r: float = 5.0,
        r_unit: str = "mm",
        p_flip: float = 0.0,
        transform: Optional[A.Compose] = None,         # Albumentations
        same_aug_per_bag: bool = False,                # ★ 追加：バッグ内で同一Aug
        debug: bool = False,
    ):
        self.df = df.reset_index(drop=True)
        self.df_loc = df_loc
        self.mode = mode
        self.volume_size_mm = tuple(map(float, volume_size_mm))
        self.patch_size_mm  = tuple(map(float, patch_size_mm))
        self.out_size_zyx = out_size_zyx

        if isinstance(overlap_size_mm, (int, float)):
            overlap_size_mm = (float(overlap_size_mm),) * 3
        self.overlap_size_mm = tuple(
            min(float(o), float(w) - 1e-6) for o, w in zip(overlap_size_mm, self.patch_size_mm)
        )

        self.r = float(r)
        assert r_unit in ("mm", "px")
        self.r_unit = r_unit
        self.p_flip = float(p_flip)
        self.debug = debug

        self.transform = transform
        self.same_aug_per_bag = bool(same_aug_per_bag)

        # Albumentations: same_aug_per_bag を使う場合は ReplayCompose にラップ
        self.replay_tfm: Optional[A.ReplayCompose] = None
        if self.transform is not None and self.same_aug_per_bag:
            if isinstance(self.transform, A.ReplayCompose):
                self.replay_tfm = self.transform
            elif isinstance(self.transform, A.Compose):
                self.replay_tfm = A.ReplayCompose(self.transform.transforms)
            else:
                self.replay_tfm = A.ReplayCompose([self.transform])

        self.site_to_idx = {name: i for i, name in enumerate(self.target_order)}

    def __len__(self) -> int:
        return len(self.df)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[","").replace("]","").replace("(","").replace(")","").replace(" ","")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0

    @staticmethod
    def _safe_spacing_1d(v, default=1.0):
        try:
            x = float(v)
        except Exception:
            return float(default)
        if not np.isfinite(x) or x <= 0:
            return float(default)
        return float(x)

    def _safe_pixel_spacing(self, v, default=(1.0, 1.0)):
        dy_raw, dx_raw = self._parse_pixel_spacing(v)
        dy = self._safe_spacing_1d(dy_raw, default=default[0])
        dx = self._safe_spacing_1d(dx_raw, default=default[1])
        return dy, dx

    @staticmethod
    def _pad_value_for_mod(mod: str, vmin_mr: float) -> float:
        return -100.0 if str(mod).upper() in ("CT","CTA","CCTA") else float(vmin_mr)

    @staticmethod
    def _resize3d(vol_zyx: np.ndarray, out_zyx: tuple[int,int,int], mode: str) -> np.ndarray:
        Z,H,W = vol_zyx.shape
        Zo,Ho,Wo = out_zyx
        zoom_factors = (Zo/max(1,Z), Ho/max(1,H), Wo/max(1,W))
        order = 0 if mode == "nearest" else 1
        return zoom(vol_zyx, zoom_factors, order=order)

    @staticmethod
    def _vl_normalize(arr: np.ndarray, v_low: float, v_high: float) -> np.ndarray:
        if v_high <= v_low:
            return np.zeros_like(arr, dtype=np.float32)
        arr = np.clip(arr, v_low, v_high)
        return (arr - v_low) / (v_high - v_low + 1e-6)

    @staticmethod
    def _build_targets_14(row: pd.Series, keys_14: Sequence[str]) -> np.ndarray:
        return np.array([float(row[k]) if (k in row.index and pd.notna(row[k])) else 0.0
                         for k in keys_14], dtype=np.float32)

    @staticmethod
    def _roi_center_from_bbox(xmin, xmax, ymin, ymax, zmin, zmax, W, H, Z):
        def mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if np.isnan(fa) or np.isnan(fb): raise ValueError
                return 0.5*(fa+fb)
            except Exception:
                return lim/2.0
        cx = int(round(mid(xmin, xmax, W)))
        cy = int(round(mid(ymin, ymax, H)))
        cz = int(round(mid(zmin, zmax, Z)))
        return cz, cy, cx

    @staticmethod
    def _crop_from_start(vol: np.ndarray, z0: int, y0: int, x0: int,
                         sz: int, sy: int, sx: int, pad_value: float):
        Z,H,W = vol.shape
        z1,y1,x1 = z0,y0,x0
        z2,y2,x2 = z0+sz, y0+sy, x0+sx

        z1c,z2c = max(0,z1), min(Z,z2)
        y1c,y2c = max(0,y1), min(H,y2)
        x1c,x2c = max(0,x1), min(W,x2)

        out = np.empty((sz,sy,sx), dtype=vol.dtype)
        out.fill(float(pad_value))

        dz0 = z1c - z1; dy0 = y1c - y1; dx0 = x1c - x1
        sub = vol[..., x1c:x2c]
        sub = sub[z1c:z2c, y1c:y2c]
        out[dz0:dz0+(z2c-z1c), dy0:dy0+(y2c-y1c), dx0:dx0+(x2c-x1c)] = sub
        return out

    @staticmethod
    def _nwin(L, W, S):
        if L <= W: return 1
        return int(math.floor((L - W) / S) + 1)

    def _match_site_index(self, site_name: str | None) -> int | None:
        if not site_name: return None
        s = site_name.strip().lower()
        for k, i in self.site_to_idx.items():
            if k.lower() == s: return i
        for k, i in self.site_to_idx.items():
            if s in k.lower(): return i
        return None

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = row["SeriesInstanceUID"]
        npy_path: str = row["file_name"]
        sorted_files: list[str] = row.get("sorted_files", [])
        mod = row.get("Modality", "")
        plane = row.get("OrientationLabel", "")
        vmin_mr = float(row.get("intensity_p0", 0))
        vmax_mr = float(row.get("intensity_p100", 1))

        # spacing（安全化）
        z_sp: float = self._safe_spacing_1d(row.get("z_spacing", 1.0), default=1.0)
        y_sp, x_sp = self._safe_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)), default=(1.0, 1.0))

        # ROI & ストライド（mm -> voxel）
        Vmm = self.volume_size_mm
        Wmm = self.patch_size_mm
        Omm = self.overlap_size_mm
        Smm = (max(1e-6, Wmm[0]-Omm[0]), max(1e-6, Wmm[1]-Omm[1]), max(1e-6, Wmm[2]-Omm[2]))

        half_roi_z = max(1, int(round((Vmm[0]/max(1e-6,z_sp))/2.0)))
        half_roi_y = max(1, int(round((Vmm[1]/max(1e-6,y_sp))/2.0)))
        half_roi_x = max(1, int(round((Vmm[2]/max(1e-6,x_sp))/2.0)))
        roi_sz = 2*half_roi_z; roi_sy = 2*half_roi_y; roi_sx = 2*half_roi_x

        psz = max(1, int(round(Wmm[0]/max(1e-6,z_sp))))
        psy = max(1, int(round(Wmm[1]/max(1e-6,y_sp))))
        psx = max(1, int(round(Wmm[2]/max(1e-6,x_sp))))
        stz = max(1, int(round(Smm[0]/max(1e-6,z_sp))))
        sty = max(1, int(round(Smm[1]/max(1e-6,y_sp))))
        stx = max(1, int(round(Smm[2]/max(1e-6,x_sp))))

        # ボリューム
        vol = np.load(npy_path, mmap_mode="r")
        Z,H,W = vol.shape

        # ROI 中心（bboxがあれば使用）
        x_min, x_max = row.get("x1", np.nan), row.get("x2", np.nan)
        y_min, y_max = row.get("y1", np.nan), row.get("y2", np.nan)
        z_min, z_max = row.get("z1", np.nan), row.get("z2", np.nan)
        cz, cy, cx = self._roi_center_from_bbox(x_min, x_max, y_min, y_max, z_min, z_max, W, H, Z)

        # ROI 開始座標（シリーズ座標, pad対応）
        pad_val = self._pad_value_for_mod(mod, vmin_mr)
        roi_z0 = int(cz - half_roi_z)
        roi_y0 = int(cy - half_roi_y)
        roi_x0 = int(cx - half_roi_x)

        # --- ROI 抽出 ---
        roi = self._crop_from_start(vol, roi_z0, roi_y0, roi_x0, roi_sz, roi_sy, roi_sx, pad_value=pad_val)

        # パッチ数（Z,Y,X 方向）
        nz = self._nwin(Vmm[0], Wmm[0], Smm[0])
        ny = self._nwin(Vmm[1], Wmm[1], Smm[1])
        nx = self._nwin(Vmm[2], Wmm[2], Smm[2])
        P = nz*ny*nx

        # ---- 病変マスク（ROI座標）----
        mask_roi = np.zeros_like(roi, dtype=np.uint8)
        site_mask_roi = np.zeros((13,)+roi.shape, dtype=np.uint8)
        loc_rows = self.df_loc[self.df_loc["SeriesInstanceUID"] == sid]
        for _, loc_row in loc_rows.iterrows():
            site_name = str(loc_row.get("location", "") or "")
            site_idx = self._match_site_index(site_name)
            site_ch = site_idx if (site_idx is not None and site_idx < 13) else None

            sop = str(loc_row["SOPInstanceUID"]) if "SOPInstanceUID" in loc_row.index and pd.notna(loc_row["SOPInstanceUID"]) else None
            if sop and isinstance(sorted_files, (list, tuple)) and len(sorted_files) > 0:
                names = [p.split("/")[-1][:-4] for p in sorted_files]
                try:
                    zc_l = names.index(sop)
                except ValueError:
                    coords = ast.literal_eval(loc_row["coordinates"])
                    zc_l = int(coords.get("z", 0))
            else:
                coords = ast.literal_eval(loc_row["coordinates"])
                zc_l = int(coords.get("z", 0))
            coords = ast.literal_eval(loc_row["coordinates"])
            yc_l = int(coords.get("y", 0)); xc_l = int(coords.get("x", 0))

            pz = int(np.clip(zc_l - roi_z0, 0, roi.shape[0]-1))
            py = int(np.clip(yc_l - roi_y0, 0, roi.shape[1]-1))
            px = int(np.clip(xc_l - roi_x0, 0, roi.shape[2]-1))

            if self.r_unit == "mm":
                rx0 = max(1, int(round(self.r / x_sp)))
                ry0 = max(1, int(round(self.r / y_sp)))
                rz0 = max(1, int(round(self.r / z_sp)))
            else:
                rx0 = ry0 = rz0 = max(1, int(round(self.r)))

            Zr = roi.shape[0]
            for dz in range(-rz0, rz0+1):
                zc_sl = pz + dz
                if not (0 <= zc_sl < Zr): continue
                scale = math.sqrt(max(0.0, 1.0 - (dz/float(rz0))**2))
                rx = max(1, int(round(rx0 * scale)))
                ry = max(1, int(round(ry0 * scale)))
                rr, cc = ellipse(r=py, c=px, r_radius=ry, c_radius=rx, shape=roi.shape[1:])
                mask_roi[zc_sl, rr, cc] = 1
                if site_ch is not None:
                    site_mask_roi[site_ch, zc_sl, rr, cc] = 1

        # ============================
        # ★ ここから高速化のコア部分 ★
        # ============================

        Zo,Ho,Wo = self.out_size_zyx

        # 1) グリッドを一括生成（インデックス / 開始点 / 正規化座標）
        iz = np.arange(nz, dtype=np.int32)
        iy = np.arange(ny, dtype=np.int32)
        ix = np.arange(nx, dtype=np.int32)
        IZ, IY, IX = np.meshgrid(iz, iy, ix, indexing='ij')  # (nz,ny,nx)

        z0_local = (IZ * stz).ravel().astype(np.int32)  # ROI原点(0,0,0)基準
        y0_local = (IY * sty).ravel().astype(np.int32)
        x0_local = (IX * stx).ravel().astype(np.int32)

        grid_idx = np.stack([IZ.ravel(), IY.ravel(), IX.ravel()], axis=1).astype(np.int64)
        starts_series = np.stack([
            roi_z0 + z0_local,
            roi_y0 + y0_local,
            roi_x0 + x0_local
        ], axis=1).astype(np.int64)

        # 正規化：RSNAPatchDatasetV2 と統一
        grid_norm_center = np.stack([
            (IZ.ravel().astype(np.float32) + 0.5)/max(1, nz),
            (IY.ravel().astype(np.float32) + 0.5)/max(1, ny),
            (IX.ravel().astype(np.float32) + 0.5)/max(1, nx),
        ], axis=1)

        grid_norm_continuous = np.stack([
            (z0_local.astype(np.float32) + 0.5*psz)/max(1, roi_sz),
            (y0_local.astype(np.float32) + 0.5*psy)/max(1, roi_sy),
            (x0_local.astype(np.float32) + 0.5*psx)/max(1, roi_sx),
        ], axis=1)
        grid_norm_continuous = np.clip(grid_norm_continuous, 0.0, 1.0).astype(np.float32, copy=False)

        # 2) ROI を一回だけリサイズ → パッチはスライスで切るだけ
        #    まず画像を 0..1 に正規化してから補間（線形）する
        if str(mod).upper() in ("CT","CTA","CCTA"):
            roi_norm = self._vl_normalize(roi.astype(np.float32, copy=False), v_low=-100, v_high=600)
        else:
            roi_norm = self._vl_normalize(roi.astype(np.float32, copy=False), v_low=vmin_mr, v_high=vmax_mr)

        fz = Zo / float(psz); fy = Ho / float(psy); fx = Wo / float(psx)
        Zr_s = max(1, int(round(roi.shape[0] * fz)))
        Yr_s = max(1, int(round(roi.shape[1] * fy)))
        Xr_s = max(1, int(round(roi.shape[2] * fx)))

        roi_resized  = self._resize3d(roi_norm, (Zr_s, Yr_s, Xr_s), mode="trilinear")
        mask_resized = self._resize3d(mask_roi.astype(np.float32), (Zr_s, Yr_s, Xr_s), mode="nearest")  # 0/1維持

        # サイト別マスク（13ch）も nearest で拡大
        site_mask_resized = np.empty((13, Zr_s, Yr_s, Xr_s), dtype=np.uint8)
        for ch in range(13):
            site_mask_resized[ch] = self._resize3d(site_mask_roi[ch].astype(np.float32), (Zr_s, Yr_s, Xr_s), mode="nearest").astype(np.uint8)

        # 3) スライスで全パッチを抽出（補間なし）
        images = np.empty((P, Zo,Ho,Wo), dtype=np.float32)
        masks  = np.empty((P, Zo,Ho,Wo), dtype=np.float32)
        site_vec_patch = np.zeros((P, 14), dtype=np.float32)
        ane_vec_patch  = np.zeros((P, 1),  dtype=np.float32)

        # スケール後の開始座標（四捨五入で整合）
        z0s = np.round(z0_local * fz).astype(np.int32)
        y0s = np.round(y0_local * fy).astype(np.int32)
        x0s = np.round(x0_local * fx).astype(np.int32)

        # Albumentations: バッグ内で同一乱数を共有
        do_flip = (self.mode == "train") and (np.random.rand() < self.p_flip)
        replay_params = None if self.replay_tfm is not None else None

        for p in range(P):
            z0p, y0p, x0p = int(z0s[p]), int(y0s[p]), int(x0s[p])

            patch  = roi_resized [z0p:z0p+Zo, y0p:y0p+Ho, x0p:x0p+Wo]
            mpatch = mask_resized[z0p:z0p+Zo, y0p:y0p+Ho, x0p:x0p+Wo]

            # Albumentations（必要なときだけ）
            if self.replay_tfm is not None:
                phwz = (np.clip(patch, 0, 1)*255.0).astype(np.uint8).transpose(1,2,0)
                mhwz = (mpatch > 0.5).astype(np.uint8).transpose(1,2,0)
                if replay_params is None:
                    data = self.replay_tfm(image=phwz, mask=mhwz)
                    replay_params = data["replay"]
                else:
                    data = A.ReplayCompose.replay(replay_params, image=phwz, mask=mhwz)
                patch  = (data["image"].transpose(2,0,1).astype(np.float32) / 255.0)
                mpatch =  data["mask"].transpose(2,0,1).astype(np.float32)
            elif self.transform is not None:
                phwz = (np.clip(patch, 0, 1)*255.0).astype(np.uint8).transpose(1,2,0)
                mhwz = (mpatch > 0.5).astype(np.uint8).transpose(1,2,0)
                data = self.transform(image=phwz, mask=mhwz)
                patch  = (data["image"].transpose(2,0,1).astype(np.float32) / 255.0)
                mpatch =  data["mask"].transpose(2,0,1).astype(np.float32)

            images[p] = patch.astype(np.float32, copy=False)
            masks[p]  = mpatch.astype(np.float32, copy=False)

            # パッチターゲット（マスク対応；サイト別）
            present = float((mpatch > 0.5).any())
            present_sites = []
            for ch in range(13):
                sm = site_mask_resized[ch, z0p:z0p+Zo, y0p:y0p+Ho, x0p:x0p+Wo]
                present_sites.append(float(sm.any()))
            site_vec = np.zeros(14, dtype=np.float32)
            site_vec[:13] = np.array(present_sites, dtype=np.float32)
            site_vec[13]  = present
            site_vec_patch[p] = site_vec
            ane_vec_patch[p, 0] = present

        # 内部フリップ適用（画像/マスク/ラベルのみ；位置は不変）
        if do_flip:
            if plane in ("AXIAL","CORONAL"):
                images = images[:, :, :, ::-1].copy()
                masks  = masks[:,  :, :, ::-1].copy()
            elif plane in ("SAGITTAL","SAGITAL"):
                images = images[:, ::-1, :, :].copy()
                masks  = masks[:,  ::-1, :, :].copy()

        # 症例ターゲット（左右入替）
        targets14_case = self._build_targets_14(row, self.target_order)
        if do_flip:
            t = targets14_case.copy()
            for a,b in self._swap_pairs: t[a], t[b] = t[b], t[a]
            targets14_case = t
            # パッチ側の部位も左右入替
            tpatch = site_vec_patch.copy()
            for a,b in self._swap_pairs:
                tpatch[:, a], tpatch[:, b] = tpatch[:, b], tpatch[:, a]
            site_vec_patch = tpatch
        target_ane_case = np.array([targets14_case[-1]], dtype=np.float32)

        out = {
            "image": torch.from_numpy(images).float(),                 # (P, Z,H,W)
            "mask":  torch.from_numpy(masks).float(),                  # (P, Z,H,W)
            "series_id": sid,
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "plane": plane,

            "grid_counts_zyx": (int(nz), int(ny), int(nx)),
            "grid_indices_zyx": torch.from_numpy(grid_idx),            # (P,3) long
            "patch_starts_zyx": torch.from_numpy(starts_series),       # (P,3) long（シリーズ座標）

            # --- パッチターゲット（マスク対応） ---
            "target_site_patch": torch.from_numpy(site_vec_patch).float(),   # (P,14)
            "target_aneurysm_patch": torch.from_numpy(ane_vec_patch).float(),# (P,1)

            # --- 症例ターゲット ---
            "target_site": torch.from_numpy(targets14_case.astype(np.float32)), # (14,)
            "target_aneurysm": torch.from_numpy(target_ane_case),               # (1,)

            # --- 位置情報（0-1 正規化, Z,Y,X） ---
            "grid_norm_center_zyx": torch.from_numpy(grid_norm_center.astype(np.float32)),
            "grid_norm_continuous_zyx": torch.from_numpy(grid_norm_continuous),
        }
        return out


class RSNASlidingFeatureDataset(Dataset):
    """
    事前抽出した特徴 {sid}.npy (featuresのみ; shape=(P, C_feat)) を読み込み、
    位置ベクトルや症例ラベルなどを df から再計算して返す軽量Dataset。

    返す dict:
      features:               FloatTensor (P, C_feat)
      target_site_case:       FloatTensor (14,)
      target_aneurysm_case:   FloatTensor (1,)
      # 位置・メタ
      patch_pos_norm_zyx:       FloatTensor (P, 3)  # ROI連続座標（中心/ROIサイズ）
      patch_pos_norm_zyx_grid:  FloatTensor (P, 3)  # グリッド正規化（i/(n-1); n==1→0.5）
      grid_counts_zyx:          tuple(int,int,int)  # (nz, ny, nx)
      grid_indices_zyx:         LongTensor (P,3)    # (iz,iy,ix)
      patch_starts_zyx:         LongTensor (P,3)    # ROI原点からの開始インデックス(シリーズ座標)
      spacing_zyx_mm:           tuple(float,float,float)
      series_id:                str
      plane:                    str

    * per-patch の教師 (target_site_patch / target_aneurysm_patch) は .npy が特徴のみのため含めません。
      （必要なら別ファイルに保存してこのクラスにオプション追加で読むようにしてください。）
    """

    target_order = [
        "Left Infraclinoid Internal Carotid Artery",
        "Right Infraclinoid Internal Carotid Artery",
        "Left Supraclinoid Internal Carotid Artery",
        "Right Supraclinoid Internal Carotid Artery",
        "Left Middle Cerebral Artery",
        "Right Middle Cerebral Artery",
        "Anterior Communicating Artery",
        "Left Anterior Cerebral Artery",
        "Right Anterior Cerebral Artery",
        "Left Posterior Communicating Artery",
        "Right Posterior Communicating Artery",
        "Basilar Tip",
        "Other Posterior Circulation",
        "Aneurysm Present",
    ]

    def __init__(
        self,
        df,                              # SeriesInstanceUID / spacing / bbox 等を含む DataFrame
        feature_root: str,               # {sid}.npy を置いたディレクトリ
        volume_size_mm: tuple[float,float,float],
        patch_size_mm: tuple[float,float,float],
        overlap_size_mm: Union[float, tuple[float,float,float]] = 0.0,
        assert_constant_P: bool = False,  # 全シリーズで P が一定であることをチェック
    ):
        self.df = df.reset_index(drop=True)
        self.feature_root = feature_root
        self.volume_size_mm = tuple(map(float, volume_size_mm))
        self.patch_size_mm  = tuple(map(float, patch_size_mm))
        if isinstance(overlap_size_mm, (int, float)):
            overlap_size_mm = (float(overlap_size_mm),) * 3
        # overlap は patch を超えないように（理論上）クリップ
        self.overlap_size_mm = tuple(
            min(float(o), float(w) - 1e-6) for o, w in zip(overlap_size_mm, self.patch_size_mm)
        )
        self.sid_list: list[str] = list(self.df["SeriesInstanceUID"].astype(str).values)
        self.assert_constant_P = bool(assert_constant_P)
        self._P_ref: Optional[int] = None

    def __len__(self) -> int:
        return len(self.sid_list)

    # ---------- helpers ----------
    @staticmethod
    def _parse_pixel_spacing(v) -> tuple[float, float]:
        if isinstance(v, (list, tuple, np.ndarray)) and len(v) >= 2:
            return float(v[0]), float(v[1])
        if isinstance(v, str):
            s = v.replace("[","").replace("]","").replace("(","").replace(")","").replace(" ","")
            parts = s.split(",")
            if len(parts) >= 2:
                return float(parts[0]), float(parts[1])
        return 1.0, 1.0

    @staticmethod
    def _safe_spacing_1d(v, default=1.0) -> float:
        try:
            x = float(v)
        except Exception:
            return float(default)
        if not np.isfinite(x) or x <= 0:
            return float(default)
        return float(x)

    def _safe_pixel_spacing(self, v, default=(1.0, 1.0)) -> tuple[float,float]:
        dy_raw, dx_raw = self._parse_pixel_spacing(v)
        dy = self._safe_spacing_1d(dy_raw, default=default[0])
        dx = self._safe_spacing_1d(dx_raw, default=default[1])
        return dy, dx

    @staticmethod
    def _build_targets_14(row: Any, keys_14: Sequence[str]) -> np.ndarray:
        return np.array([float(row[k]) if (k in row.index and np.isfinite(row[k])) else 0.0
                         for k in keys_14], dtype=np.float32)

    @staticmethod
    def _roi_center_from_bbox(xmin, xmax, ymin, ymax, zmin, zmax, W, H, Z) -> tuple[int,int,int]:
        def mid(a, b, lim):
            try:
                fa, fb = float(a), float(b)
                if np.isnan(fa) or np.isnan(fb): raise ValueError
                return 0.5*(fa+fb)
            except Exception:
                return lim/2.0
        cx = int(round(mid(xmin, xmax, W)))
        cy = int(round(mid(ymin, ymax, H)))
        cz = int(round(mid(zmin, zmax, Z)))
        return cz, cy, cx

    @staticmethod
    def _nwin(L, W, S) -> int:
        if L <= W: return 1
        return int(math.floor((L - W) / S) + 1)

    def _path_for_sid(self, sid: str) -> str:
        return os.path.join(self.feature_root, f"{sid}.npy")

    # ---------- main ----------
    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.df.iloc[index]
        sid: str = str(row["SeriesInstanceUID"])
        npy_path: str = row["file_name"]  # 元ボリュームへのパス（shape取得に使用）

        # ---- features をロード (P, C_feat) ----
        feat_path = self._path_for_sid(sid)
        if not os.path.exists(feat_path):
            raise FileNotFoundError(f"feature file not found: {feat_path}")
        feats = np.load(feat_path)  # 期待: (P, C_feat)
        if feats.ndim != 2:
            raise ValueError(f"{feat_path}: expected 2D (P, C_feat), got shape {feats.shape}")
        feats = feats.astype(np.float32, copy=False)
        P_feat, C_feat = feats.shape

        # ---- スペーシング・平面などメタ ----
        mod   = str(row.get("Modality", ""))
        plane = str(row.get("OrientationLabel", ""))
        z_sp: float = self._safe_spacing_1d(row.get("z_spacing", 1.0), default=1.0)
        y_sp, x_sp = self._safe_pixel_spacing(row.get("PixelSpacing", (1.0, 1.0)), default=(1.0, 1.0))

        # ---- 体積 shape だけ読む（メモリマップ; 実データは読まない）----
        vol = np.load(npy_path, mmap_mode="r")
        Z, H, W = vol.shape  # shape だけ使う

        # ---- ROI 原点（シリーズ座標） ----
        Vmm = self.volume_size_mm
        Wmm = self.patch_size_mm
        Omm = self.overlap_size_mm
        Smm = (max(1e-6, Wmm[0]-Omm[0]), max(1e-6, Wmm[1]-Omm[1]), max(1e-6, Wmm[2]-Omm[2]))

        half_roi_z = max(1, int(round((Vmm[0]/max(1e-6,z_sp))/2.0)))
        half_roi_y = max(1, int(round((Vmm[1]/max(1e-6,y_sp))/2.0)))
        half_roi_x = max(1, int(round((Vmm[2]/max(1e-6,x_sp))/2.0)))
        roi_sz = 2*half_roi_z; roi_sy = 2*half_roi_y; roi_sx = 2*half_roi_x

        # bbox → ROI 中心（シリーズ座標）
        x_min, x_max = row.get("x1", np.nan), row.get("x2", np.nan)
        y_min, y_max = row.get("y1", np.nan), row.get("y2", np.nan)
        z_min, z_max = row.get("z1", np.nan), row.get("z2", np.nan)
        cz, cy, cx = self._roi_center_from_bbox(x_min, x_max, y_min, y_max, z_min, z_max, W, H, Z)

        roi_z0 = int(cz - half_roi_z)
        roi_y0 = int(cy - half_roi_y)
        roi_x0 = int(cx - half_roi_x)

        # ---- グリッドと位置ベクトルを復元 ----
        psz = max(1, int(round(Wmm[0]/max(1e-6,z_sp))))
        psy = max(1, int(round(Wmm[1]/max(1e-6,y_sp))))
        psx = max(1, int(round(Wmm[2]/max(1e-6,x_sp))))
        stz = max(1, int(round(Smm[0]/max(1e-6,z_sp))))
        sty = max(1, int(round(Smm[1]/max(1e-6,y_sp))))
        stx = max(1, int(round(Smm[2]/max(1e-6,x_sp))))

        nz = self._nwin(Vmm[0], Wmm[0], Smm[0])
        ny = self._nwin(Vmm[1], Wmm[1], Smm[1])
        nx = self._nwin(Vmm[2], Wmm[2], Smm[2])
        P_calc = nz*ny*nx

        # P が一致するかチェック
        if P_feat != P_calc:
            raise RuntimeError(
                f"[{sid}] feature P ({P_feat}) != grid P ({P_calc}) "
                f"[nz,ny,nx]=[{nz},{ny},{nx}] / "
                f"Vmm={Vmm} Wmm={Wmm} Omm={Omm} Smm={Smm} / spacing={(z_sp,y_sp,x_sp)}"
            )

        # 位置ベクトル（ROI連続 / グリッド）
        def norm_index(i, n):
            return 0.5 if n <= 1 else float(i) / float(n - 1)

        pos_roi_norm = np.zeros((P_calc, 3), dtype=np.float32)
        pos_grid_norm = np.zeros((P_calc, 3), dtype=np.float32)
        grid_indices = np.zeros((P_calc, 3), dtype=np.int64)
        starts_zyx = np.zeros((P_calc, 3), dtype=np.int64)

        p = 0
        for iz in range(nz):
            z0 = iz*stz
            for iy in range(ny):
                y0 = iy*sty
                for ix in range(nx):
                    x0 = ix*stx

                    # ROI連続正規化（Z,Y,X）
                    cz_roi = (z0 + 0.5*psz) / max(1, roi_sz)
                    cy_roi = (y0 + 0.5*psy) / max(1, roi_sy)
                    cx_roi = (x0 + 0.5*psx) / max(1, roi_sx)
                    pos_roi_norm[p]  = [np.clip(cz_roi, 0.0, 1.0),
                                        np.clip(cy_roi, 0.0, 1.0),
                                        np.clip(cx_roi, 0.0, 1.0)]
                    pos_grid_norm[p] = [norm_index(iz, nz),
                                        norm_index(iy, ny),
                                        norm_index(ix, nx)]
                    grid_indices[p]  = [iz, iy, ix]
                    starts_zyx[p]    = [roi_z0+z0, roi_y0+y0, roi_x0+x0]
                    p += 1

        # ---- 症例ターゲット ----
        targets14_case = self._build_targets_14(row, self.target_order)
        target_ane_case = np.array([targets14_case[-1]], dtype=np.float32)
        # feats = np.concatenate([feats, pos_roi_norm], axis=1)

        out = {
            "features": torch.from_numpy(feats),                              # (P, C_feat)
            "target_site": torch.from_numpy(targets14_case.astype(np.float32)),  # (14,)
            "target_aneurysm": torch.from_numpy(target_ane_case),        # (1,)

            "patch_pos_norm_zyx": torch.from_numpy(pos_roi_norm),             # (P,3)
            "patch_pos_norm_zyx_grid": torch.from_numpy(pos_grid_norm),       # (P,3)
            "grid_counts_zyx": (int(nz), int(ny), int(nx)),
            "grid_indices_zyx": torch.from_numpy(grid_indices),               # (P,3)
            "patch_starts_zyx": torch.from_numpy(starts_zyx),                 # (P,3)
            "spacing_zyx_mm": (float(z_sp), float(y_sp), float(x_sp)),
            "series_id": sid,
            "plane": plane,
        }
        return out


def to_axial(
    vol: np.ndarray,
    plane: str,
    reverse_hint: Optional[bool] = None,
    channels: Optional[str] = None  # None / "last" / "first"
) -> np.ndarray:
    """
    入力:
      vol: (N, H, W[, C])  … N はスライス数（planeに応じて N が Z/Y/X を指す）
      plane: "axial" | "coronal" | "sagittal"
      reverse_hint: True=Z軸反転, False=反転なし, None=不明（反転しない）
      channels: カラーチャンネル次元の位置（None=なし, "last"=(..., C), "first"=(C, ...))

    出力:
      axial 配列: (Z, Y, X[, C])（C は指定に応じて末尾/先頭を維持）
    """
    assert plane in {"axial", "coronal", "sagittal"}

    if channels == "first":
        # (C, N, H, W) -> (N, H, W, C) にして処理を簡単化
        vol = np.moveaxis(vol, 0, -1)
        ch_last = True
    elif channels == "last":
        ch_last = True
    else:
        ch_last = False  # vol.ndim == 3 を想定

    # 軸入れ替え（transpose）: 目標は (Z, Y, X[, C])
    if ch_last:
        # いまは (N, H, W, C) 前提にそろえる
        if vol.ndim != 4:
            raise ValueError("channels='last' の場合、vol.ndim は 4 必要です。")
        if plane == "axial":
            axial = vol  # (Z, Y, X, C)
        elif plane == "coronal":
            # (Y, Z, X, C) に近い並びを (Z, Y, X, C) へ
            axial = np.transpose(vol, (1, 0, 2, 3))  # swap N<->H
            axial = axial[::-1]
        elif plane == "sagittal":
            # (X, Z, Y, C) に近い並びを (Z, Y, X, C) へ
            axial = np.transpose(vol, (1, 2, 0, 3))  # (N,H,W,C)->(H,W,N,C)            
            axial = axial[::-1]
    else:
        # vol.ndim == 3: (N, H, W)
        if vol.ndim != 3:
            raise ValueError("channels を指定しない場合、vol.ndim は 3 必要です。")
        if plane == "axial":
            axial = vol  # (Z, Y, X)
        elif plane == "coronal":
            axial = np.transpose(vol, (1, 0, 2))  # (Y,Z,X)へ
            axial = axial[::-1]
        elif plane == "sagittal":
            axial = np.transpose(vol, (1, 2, 0))  # (Z,Y,X)へ
            axial = axial[::-1]

    # Z 方向の反転（3D DICOM の reverse_hint を尊重）
    if reverse_hint is True:
        axial = axial[::-1, ...]  # Z 軸反転
    # reverse_hint が False/None の場合は何もしない

    # channels='first' で戻したい場合は先頭へ戻す
    if channels == "first":
        axial = np.moveaxis(axial, -1, 0)  # (C, Z, Y, X)
    axial = np.ascontiguousarray(axial)

    return axial


if __name__ == "__main__":
    root = '/mnt/project/brain/aneurysm/tamoto/RSNA2025/data/npy_float32'
    df = pd.read_csv("/home/tamoto/kaggle/RSNA2025/data/train_add_metadata_v3_with_intensity_roi.csv")

    # df["file_name"] = df.apply(asign_filename_npy, axis=1)
    series_to_remove = [
        "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306",  # 読み込みエラー
        "1.2.826.0.1.3680043.8.498.82768897201281605198635077495114055892",  # 読み込みエラー
        "1.2.826.0.1.3680043.8.498.75712554178574230484227682423862727306",  # 読み込みエラー
        "1.2.826.0.1.3680043.8.498.10063454172499468887877935052136698373",  # z_spacingエラー
        "1.2.826.0.1.3680043.8.498.22157965342587174310173115980837533982",  # 読み込みエラー
    ]
    df["sorted_files"] = df["sorted_files"].apply(lambda x: ast.literal_eval(x) if pd.notna(x) else x)
    df = df[~df["SeriesInstanceUID"].isin(series_to_remove)].reset_index(drop=True)
    df['file_name'] = df['SeriesInstanceUID'].apply(lambda x: Path(root)/x/(x+'.npy'))
    df_2d = df[~df["z_spacing"].isna()].reset_index(drop=True)
    df_loc = pd.read_csv("/home/tamoto/kaggle/RSNA2025/data/train_localizers.csv")
    transforms_valid = A.Compose([
        #A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        #A.Transpose(p=0.5),
        A.Rotate(limit=20, p=0.5, border_mode=0),
        A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.7),
        A.ShiftScaleRotate(shift_limit=0.3, scale_limit=0.3, rotate_limit=45, border_mode=4, p=0.7),
        A.OneOf([
            A.MotionBlur(blur_limit=3),
            A.MedianBlur(blur_limit=3),
            A.GaussianBlur(blur_limit=3),
            A.GaussNoise(var_limit=(3.0, 9.0)),
            ], p=0.5),
        A.OneOf([
            A.OpticalDistortion(distort_limit=1.),
            A.GridDistortion(num_steps=5, distort_limit=1.),
            ], p=0.5),
        
    ])
    
    print(df_loc.columns)
    # ds = RSNAPatchDataset(
    #     df=df_2d,
    #     df_loc=df_loc,
    #     mode='train',
    #     patch_size_mm=(30, 30, 30),   # (z_mm, y_mm, x_mm)
    #     out_size_zyx=(96, 96, 96),          # (Z_out, H_out, W_out)
    #     r=3.0,
    #     r_unit="mm",                           # "mm" or "px"
    #     p_lesion_crop=1.0,                   # 病変を含むクロップを選ぶ確率
    #     jitter_mm=None, # 病変含むクロップ時の中心ずらし許容量(mm)。Noneならパッチ半径相当まで自動設定
    #     choose_random_lesion_in_train=True,   # 複数座標があればtrainはランダム選択
    #     transform=transforms_valid,
    #     debug=False,
    # )
    # ds = RSNAPatchDatasetV3(
    #     df=df_2d,
    #     df_loc=df_loc,
    #     mode='train',
    #     volume_size_mm=(60.0, 60., 60.0),  # ボリューム全体のサイズ(mm)
    #     patch_size_mm=(60.0, 60.0, 60.0),       # 切り出すパッチのサイズ(mm)
    #     out_size_zyx=(96, 96, 96),              # 出力サイズ(voxel)
    #     r=5.0,                                  # 病変検索半径
    #     r_unit="mm",                            # 半径の単位
    #     p_lesion_crop=0.7,                      # 病変を含むパッチを選ぶ確率
    #     jitter_mm=None,                         # 中心のジッター
    #     choose_random_lesion_in_train=True,    # 複数病変時にランダム選択
    #     p_flip=0.0,                             # フリップ確率
    #     transform=transforms_valid,                   # Albumentations変換
    #     debug=False
    # )
    ds = RSNAROIDatasetV1(
        df=df_2d,
        df_loc=df_loc,
        mode='train',
        volume_size_mm=(90,90,90),   # ROI の物理サイズ (Z,Y,X) [mm]
        out_size_zyx=(128,128,128),           # ネット入力 (Z,Y,X)
        map_size_zyx=(4,4,4),  # 特徴マップ (Z,Y,X) へ縮約した mask を作る場合に指定
        r=2.0,
        r_unit="mm",
        p_flip=0.0,
        align_plane=True,
        transform=transforms_valid,
        )
    # ds = RSNASlidingBagDataset(
    #                     df=df_2d, df_loc=df_loc, mode="val",
    #                     volume_size_mm=(120,120,120),
    #                     patch_size_mm=(30,30,30),
    #                     overlap_size_mm=(0,0,0),           # 例: (10,10,10) で重なりあり
    #                     out_size_zyx=(64,96,96),
    #                     p_flip=0.0,
    #                     transform=transforms_valid,
    #                     same_aug_per_bag=True
    #                 )
    # ds = RSNASlidingFeatureDataset(
    #     df=df_2d,                              # SeriesInstanceUID / spacing / bbox 等を含む DataFrame
    #     feature_root='/home/tamoto/kaggle/RSNA2025/outputs/exp010_dsv2_multi_epoch100/features/fold0',               # {sid}.npy を置いたディレクトリ
    #     volume_size_mm=(120, 120, 120),
    #     patch_size_mm=(30, 30, 30),
    #     overlap_size_mm=(0,0,0),
    #     assert_constant_P=False
    #     )

    # サンプル取得
    dl = torch.utils.data.DataLoader(
        ds,
        batch_size=4,
        shuffle=False,
        num_workers=min(os.cpu_count(), 4),
        pin_memory=True,
        drop_last=True,
        prefetch_factor=4,
        pin_memory_device="cuda",
        persistent_workers=True,
    )
    data = ds[8]
    print(data['image'].shape)
    for k, v in data.items():
        try:
            print(k, v.shape)
        except:
            print(k, v)
    print(data['target_site'])
    print(data['mask_map'].unique(dim=0), data['mask_map'].shape)
    # for i in range(14):
    #     print(i, data['mask_map'][i].unique(), data['mask'][i].unique())

    #print(data['image'].shape)
    # for i in range(100):
    #     data = ds[i]
        #print(data['image'].shape, data['mask'].shape, data['mask'].sum()/(128**3), data['target_site_case'])
    for data in tqdm(dl, total=len(dl)):
        print(data['image'].shape, data['target_site'].shape, data["modality_encoded"])
        break
