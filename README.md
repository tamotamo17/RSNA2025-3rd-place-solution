# RSNA Intracranial Aneurysm Detection — 3rd Place

This repository contains the training code for our **3rd place** solution in the RSNA Intracranial Aneurysm Detection competition.

- Solution write-up: [here](https://www.kaggle.com/competitions/rsna-intracranial-aneurysm-detection/writeups/3rd-place-solution)  
- Inference code: [here](https://www.kaggle.com/code/tamotamo/rsna2025-3rd-place-inference)

## Hardware

1. **Kaggle Notebook** (EDA, dataset creation, and object detection)  
2. **GPU Server** (classification)  
   - GPU: 1× NVIDIA A100-SXM4-40GB (40 GB)  
   - CPU: AMD EPYC 7742 (64 cores / 256 vCPU)  
   - Memory: 1.0 TiB  
   - CUDA: 12.4  
   - OS: Ubuntu 22.04 LTS  

## Requirements

We use **Poetry** for Python environment management. After installing Poetry, set up the environment with:

```bash
poetry install

```

## Data Preparation
1. Download competition data
```
poetry run kaggle competitions download -c rsna-intracranial-aneurysm-detection
```
1. Metadata extraction & train/validation split  
Run `notebooks/prepare_dataset/rsna2025-eda-plus3d.ipynb` to create train_add_metadata_v5.csv.  
Kaggle version: [here](https://www.kaggle.com/code/tamotamo/rsna2025-eda-plus3d)

1. Dataset preparation for ROI extraction (detection training)  
Run `notebooks/prepare_dataset/rsna2025-annotate-vessel-bbox.ipynb` to create the dataset for training vessel ROI detection (labeled data only).  
Kaggle version: [here](https://www.kaggle.com/code/tamotamo/rsna2025-annotate-vessel-bbox)  
This notebook uses `train_add_metadata_v5.csv` from the previous step.  
After creating the dataset, we manualy removed axial-plane data.

1. Dataset preparation for ROI inference (full series)  
Run `notebooks/prepare_dataset/rsna2025-series-dataset-yolo-vessel-region-plus-3d.ipynb` to build the dataset for inference-time ROI detection (full series)  
This dataset contains the full series data.  
Kaggle version: [here](https://www.kaggle.com/code/tamotamo/rsna2025-series-dataset-yolo-vessel-region-plus-3d)

1. Dataset preparation for classification  
Run `scripts/save_npy.py` to convert DICOM files into 3D NumPy arrays.

## 3D ROI extraction
1. Train detection models  
Run `notebooks/roi_extraction/rsna2025-train-yolo-vessel-region-detection.ipynb` to train the ROI detection models.  
Kaggle version: [here](https://www.kaggle.com/code/tamotamo/rsna2025-train-yolo-vessel-region-detection)
1. Run detection inference  
Run `notebooks/roi_extraction/rsna2025-inference-yolo-vessel-region-detection-v3.ipynb` to generate detection results for sagittal and coronal planes for each case.  
Kaggle version [here](https://www.kaggle.com/code/tamotamo/rsna2025-inference-yolo-vessel-region-detection-v3)
1. Reconstruct 3D ROI  
Run `notebooks/roi_extraction/rsna2025-reconstruct-3d-roi.ipynb` to reconstruct 3D ROIs from sagittal/coronal detections.  
This produces `train_add_metadata_v5_roi.csv` for downstream steps.  
Kaggle version: [here](https://www.kaggle.com/code/tamotamo/rsna2025-reconstruct-3d-roi)

## 3D ROI training
1. Intensity feature extraction  
Run `scripts/add_intensity_feature.py` to extract intensity features for preprocessing.
This creates `train_add_metadata_v5_with_intensity_roi.csv` for training.
1. Model training  
Edit paths in `src/classification/config/config_roi_modelx.py`:
- train_csv
- train_loc_csv
- data_root
- output_dir

Then, run training with the following command:
```bash
sh run_train.sh
```
