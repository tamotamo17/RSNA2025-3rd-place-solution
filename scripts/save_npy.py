import os
from pathlib import Path
from tqdm import tqdm
import copy
import ast
from collections import defaultdict, Counter
from glob import glob

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import List, Optional

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import nibabel as nib
import cv2
from scipy.ndimage import zoom
import h5py
import pydicom


def norm_by_percentile(img, p_min, p_max):
    vmin = np.percentile(img, p_min)
    vmax = np.percentile(img, p_max)
    img = np.clip(img, vmin, vmax).astype(np.float32)
    img = (img - vmin) / (vmax - vmin + 1e-6)
    img = (img * 255).round().astype(np.uint8)
    return img


def norm_by_value(img, v_min, v_max):
    img = np.clip(img, v_min, v_max).astype(np.float32)
    img = (img - v_min) / (v_max - v_min + 1e-6)
    img = (img * 255).round().astype(np.uint8)
    return img


def save_image(img, modality, save_path):
    if modality == "CTA":
        img = norm_by_value(img, 0, 500)
    else:
        img = norm_by_percentile(img, 1, 99)
    # 保存先ディレクトリを作成
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # PNG として書き出し
    cv2.imwrite(str(save_path), img)


def _ensure_dir(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)


def _auto_chunks(shape, dtype, target_mb: float = 1.5):
    """
    約 target_mb MB 程度のchunkを目標に (Z,Y,X) で自動決定。
    ランダム3Dパッチにも順次アクセスにもバランスが良いサイズを狙う。
    """
    z, y, x = map(int, shape)
    item = np.dtype(dtype).itemsize
    # 初期案：Zは最大16、XYは最大256（大き過ぎる場合に縮める）
    cz = min(16, z) if z > 1 else 1
    cy = min(256, y)
    cx = min(256, x)

    def chunk_bytes(cz, cy, cx):
        return cz * cy * cx * item

    target_bytes = int(target_mb * 1024 * 1024)

    # 大きすぎたら、一番大きい軸から1/2に縮めていく（下限1）
    while chunk_bytes(cz, cy, cx) > target_bytes and (cz > 1 or cy > 1 or cx > 1):
        # 体積が大きい軸を優先的に縮小
        # ただし1未満にはしない
        if cy >= cx and cy >= cz and cy > 1:
            cy = max(1, cy // 2)
        elif cx >= cz and cx >= cy and cx > 1:
            cx = max(1, cx // 2)
        elif cz > 1:
            cz = max(1, cz // 2)
        else:
            break

    # 小さすぎる場合、少しだけ広げる（読み込み効率向上）※上限は原形状
    while chunk_bytes(cz, cy, cx) < target_bytes // 3:
        grown = False
        if cz < z and cz < 16:
            cz = min(z, cz * 2)
            grown = True
        if cy < y and cy < 256:
            cy = min(y, cy * 2)
            grown = True
        if cx < x and cx < 256:
            cx = min(x, cx * 2)
            grown = True
        if not grown:
            break

    return (max(1, cz), max(1, cy), max(1, cx))


def process_series_to_npy(
    root: Path,
    save_root: Path,
    sid: str,
    modality: str,
    dcm_paths: Optional[list[str]] = None,
    out_name: Optional[str] = None,
    normalize: bool = False,  # Trueなら従来PNG相当の正規化でuint8保存
):
    """
    DICOMシリーズを読み込み、(Z,H,W) の numpy 配列として .npy 保存する。

    Parameters
    ----------
    root : Path
        ルートDICOMディレクトリ
    save_root : Path
        出力の保存先ルート
    sid : str
        SeriesInstanceUID 等、シリーズID
    modality : str
        'CTA' なら値域正規化 (0–500HU)、それ以外はパーセンタイル正規化 (1–99%)
    dcm_paths : list[str] | None
        明示的に読み込む DICOM ファイルパスのリスト（Z順に並んでいる前提）
        None の場合は root/sid 配下を列挙（順序はファイル名順）
    out_name : str | None
        保存ファイル名。None の場合は f"{sid}.npy"
    normalize : bool
        True なら uint8 正規化（容量削減）。False なら元 dtype のまま保存。
    """
    # パス列挙（与えられてなければフォルダ内を拾う）
    if not dcm_paths:
        dcm_paths = [str(p) for p in (root / sid).glob("*")]
        dcm_paths.sort()

    if len(dcm_paths) == 0:
        raise FileNotFoundError(f"No DICOM found under: {root / sid}")

    # 代表ファイルで形状確認
    first = pydicom.dcmread(dcm_paths[0])
    arr = first.pixel_array

    if arr.ndim == 3:
        # マルチフレーム（Z,Y,X想定）
        vol = arr
    elif arr.ndim == 2:
        # 単フレームの集合（渡された順に積む：Zソート済前提）
        H, W = arr.shape
        Z = len(dcm_paths)
        vol = np.empty((Z, H, W), dtype=arr.dtype)
        vol[0] = arr
        for i, p in enumerate(dcm_paths[1:], start=1):
            ds = pydicom.dcmread(p)
            img = ds.pixel_array
            if img.ndim != 2:
                raise ValueError(f"Unexpected shape at {p}: {img.shape}")
            if img.shape != (H, W):
                raise ValueError(f"Size mismatch at {p}: {img.shape} vs {(H, W)}")
            vol[i] = img
    else:
        raise ValueError(f"Unexpected pixel_array shape: {arr.shape}")

    # 正規化（uint8）
    if normalize:
        if modality.upper() == "CTA":
            vol = norm_by_value(vol, 0, 500)
        else:
            vol = norm_by_percentile(vol, 1, 99)

    # 保存
    if out_name is None:
        out_name = f"{sid}.npy"
    if not out_name.lower().endswith(".npy"):
        out_name = os.path.splitext(out_name)[0] + ".npy"

    save_path = (save_root / sid / out_name).resolve()
    _ensure_dir(save_path)
    np.save(save_path, vol)  # .npy (非圧縮、メタは含まれない)

    return save_path


def process_series_to_hdf5(
    root: Path,
    save_root: Path,
    sid: str,
    modality: str,
    dcm_paths: Optional[list[str]] = None,
    out_name: Optional[str] = None,
    normalize: bool = True,  # Trueなら従来PNG相当の正規化でuint8保存
    compression: str = "lzf",  # 速度と圧縮のバランスが良い
    target_chunk_mb: float = 1.5,  # チャンク目標サイズ（1〜2MBが無難）
):
    # dcm_pathsが与えられなければ列挙順になる点に注意（既に外でソート済み前提）
    if not dcm_paths:
        dcm_paths = [str(p) for p in (root / sid).glob("*")]

    # 代表を読んで形状など把握
    first = pydicom.dcmread(dcm_paths[0])
    arr = first.pixel_array

    if arr.ndim == 3:
        # マルチフレーム（Z,Y,X想定）
        vol = arr
    elif arr.ndim == 2:
        # 複数ファイルを渡された順に積む（Zソート済み前提）
        H, W = arr.shape
        Z = len(dcm_paths)
        vol = np.empty((Z, H, W), dtype=arr.dtype)
        vol[0] = arr
        for i, p in enumerate(dcm_paths[1:], start=1):
            ds = pydicom.dcmread(p)
            img = ds.pixel_array
            if img.ndim != 2:
                raise ValueError(f"想定外shape: {p} -> {img.shape}")
            vol[i] = img
    else:
        raise ValueError(f"想定外shape: {arr.shape}")

    # （任意）正規化してuint8保存（容量削減＆圧縮効率↑）
    if normalize:
        if modality == "CTA":
            vol = norm_by_value(vol, 0, 500)
        else:
            vol = norm_by_percentile(vol, 1, 99)

    # 保存
    if out_name is None:
        out_name = f"{sid}.h5"
    save_path = save_root / sid / out_name
    _ensure_dir(save_path)

    chunks = _auto_chunks(vol.shape, vol.dtype, target_mb=target_chunk_mb)

    with h5py.File(save_path, "w") as f:
        dset = f.create_dataset(
            "vol",
            data=vol,
            chunks=chunks,
            compression=compression,  # "lzf"
            shuffle=True,  # 整数系の圧縮効率↑（LZFと併用OK）
            track_times=False,  # 余計なメタ更新を抑制
        )
        # 簡易メタ
        dset.attrs["sid"] = sid
        dset.attrs["modality"] = modality
        dset.attrs["normalized_uint8"] = bool(normalize)
        dset.attrs["shape"] = vol.shape
        dset.attrs["dtype"] = str(vol.dtype)
        dset.attrs["chunks"] = chunks


def process_series_to_hdf5_streaming(
    root: Path,
    save_root: Path,
    sid: str,
    modality: str,
    dcm_paths: Optional[list[str]] = None,
    out_name: Optional[str] = None,
    normalize: bool = True,
    compression: str = "lzf",
    chunks: Optional[tuple[int, int, int]] = None,  # 例：(16,128,128)。未指定ならHDF5に任せる
    sample_for_percentile: int = 16,  # 非CTA用の近似percentileサンプル数
):
    if not dcm_paths:
        dcm_paths = [str(p) for p in (root / sid).glob("*")]

    # 代表で形状決定（multi-frameならZ,Y,X）
    first = pydicom.dcmread(dcm_paths[0])
    arr = first.pixel_array
    if arr.ndim == 3:
        Z, H, W = arr.shape
        series_iter = [(0, None)]  # 0=use arr directly
        use_multiframe = True
    elif arr.ndim == 2:
        Z, H, W = len(dcm_paths), *arr.shape
        use_multiframe = False
    else:
        raise ValueError(f"Unexpected shape: {arr.shape}")

    # 保存先
    if out_name is None:
        out_name = f"{sid}.h5"
    save_path = save_root / sid / out_name
    _ensure_dir(save_path)

    # 出力dtype（normalize=Trueならuint8）
    out_dtype = np.uint8 if normalize else arr.dtype

    with h5py.File(save_path, "w") as f:
        dset = f.create_dataset(
            "vol",
            shape=(Z, H, W),
            dtype=out_dtype,
            chunks=chunks,
            compression=compression,
            shuffle=True,
            track_times=False,
        )
        dset.attrs["sid"] = sid
        dset.attrs["modality"] = modality
        dset.attrs["normalized_uint8"] = bool(normalize)

        # ---- percentile境界を先に近似推定（非CTAのみ）----
        if normalize and modality != "CTA":
            # ランダムに最大 sample_for_percentile 枚を抽出して分布推定
            idxs = np.linspace(0, Z - 1, num=min(Z, sample_for_percentile), dtype=int)
            samples = []
            for i in idxs:
                if use_multiframe:
                    frame = arr[i]
                else:
                    frame = pydicom.dcmread(dcm_paths[i]).pixel_array
                # サンプリングしてメモリを抑える（例：16x間引き）
                samp = frame[:: max(1, H // 256 + 1), :: max(1, W // 256 + 1)]
                samples.append(samp.ravel())
            sample_vec = np.concatenate(samples)
            vmin = np.percentile(sample_vec, 1)
            vmax = np.percentile(sample_vec, 99)
            del samples, sample_vec
        else:
            vmin, vmax = 0.0, 500.0  # CTAの既定

        # ---- 逐次読み→逐次正規化→逐次書き ----
        for z in range(Z):
            if use_multiframe:
                frame = arr[z]
            else:
                frame = pydicom.dcmread(dcm_paths[z]).pixel_array

            if normalize:
                if modality == "CTA":
                    # in-place 風に一時配列最小化
                    fr = frame.astype(np.float32, copy=False)
                    fr = np.clip(fr, 0, 500, out=fr)
                    fr -= 0
                    fr /= 500 - 0 + 1e-6
                else:
                    fr = frame.astype(np.float32, copy=False)
                    fr = np.clip(fr, vmin, vmax, out=fr)
                    fr -= vmin
                    fr /= vmax - vmin + 1e-6
                fr *= 255.0
                # 書き込み時にuint8へ
                dset[z, :, :] = fr.round().astype(np.uint8, copy=False)
            else:
                dset[z, :, :] = frame  # 原強度のまま書き込み

            # 明示的に参照を切ってGCを促す
            del frame

    # マルチフレームの元配列参照を切る
    del arr
    import gc

    gc.collect()


if __name__ == "__main__":
    df = pd.read_csv("/home/tamoto/kaggle/RSNA2025/data/train_add_metadata_v3.csv")
    root = Path("/mnt/science_data/opendata/RSNA2025v2/series")
    save_root = Path("/local_ssd/tamoto/RSNA2025/data/npy_float32")
    # s_idx      = 0
    # num_series = 10
    max_workers = 4  # CPU コア数に合わせて調整

    tasks = []
    for _, row in df.iterrows():
        sid = row["SeriesInstanceUID"]
        modality = row["Modality"]

        dcm_paths = []
        if pd.notna(row["sorted_files"]):
            dcm_paths = ast.literal_eval(row["sorted_files"])
            dcm_paths = [
                path.replace(
                    "/kaggle/input/rsna-intracranial-aneurysm-detection", "/mnt/science_data/opendata/RSNA2025v2"
                )
                for path in dcm_paths
            ]
        tasks.append((root, save_root, sid, modality, dcm_paths))

    # 並列実行
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_series_to_npy, *t) for t in tasks]
        for _ in tqdm(as_completed(futures), total=len(futures), desc="saving npy"):
            pass
    # process_series_to_npy(*tasks[0])

    print("✓ All done")
