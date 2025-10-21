#!/usr/bin/env python3
# -*- coding:utf-8 -*-

"""
Configuration for classification training
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
import albumentations as A


@dataclass
class AugmentationConfig:
    """Data augmentation configuration for Albumentations"""

    transforms_train = A.Compose(
        [
            # A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Transpose(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=(-0.1, 0.1), contrast_limit=(-0.1, 0.1), p=0.7),
            A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, border_mode=4, p=0.7),
            A.OneOf(
                [
                    A.MotionBlur(blur_limit=3),
                    A.MedianBlur(blur_limit=3),
                    A.GaussianBlur(blur_limit=3),
                    A.GaussNoise(var_limit=(3.0, 9.0)),
                ],
                p=0.5,
            ),
            # A.OneOf(
            #     [
            #         A.OpticalDistortion(distort_limit=1.0),
            #         A.GridDistortion(num_steps=5, distort_limit=1.0),
            #     ],
            #     p=0.5,
            # ),
        ]
    )

    transforms_valid = A.Compose([])


@dataclass
class DataConfig:
    """Data configuration"""

    train_csv: str = "/home/tamoto/kaggle/RSNA2025/data/train_add_metadata_v5_with_intensity_roi.csv"
    train_loc_csv: str = "/home/tamoto/kaggle/RSNA2025/data/train_localizers.csv"
    val_csv: Optional[str] = None
    data_root: str = "/mnt/project/brain/aneurysm/tamoto/RSNA2025/data/npy_float32"# "/local_ssd/tamoto/RSNA2025/data/npy_float32"
    plane = None#"AXIAL" # None
    modality = None # 'MRA', '
    target_spacing: List[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    # crop_size: List[int] = field(default_factory=lambda: [128, 128, 128])
    num_classes: int = 14  # 1

    # RSNAPatchDataset specific parameters
    volume_size_mm: List[float] = field(default_factory=lambda: [90.0, 90.0, 90.0])
    out_size_zyx: List[int] = field(default_factory=lambda: [192, 192, 192])  # (Z_out, H_out, W_out)
    map_size_zyx: List[int] = field(default_factory=lambda: [48, 48, 48])

    r: float = 2.0  # Search radius for lesion detection
    r_unit: str = "mm"  # "mm" or "px"
    p_flip: float = 0.5
    align_plane: bool = False
    cta_min: float = -100
    cta_max: float = 600
    # Data split settings
    seed: int = 123 # 123


@dataclass
class ModelConfig:
    """Model configuration"""

    name: str = "aneurysm_3d_v7"
    backbone: str = "resnet18" # "resnet34"
    in_chans: int = 1
    num_classes: int = 14
    drop_rate: float = 0.2
    map_size: List[int] = field(default_factory=lambda: [48, 48, 48])
    use_coords: bool = False
    pretrained: bool = True  # 3D models typically don't have ImageNet pretraining
    return_logits: bool = True


@dataclass
class LossConfig:
    """Loss configuration"""

    name: str = "aneurysm_3d_roi_v2"  # "patch_binary_focal"  # Changed default to patch-based loss
    loss_type: str = "multi"
    cls_loss_type: str = "bce"  # "focal", "bce"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    dice_smooth: float = 1.0
    pos_weight: float = 5.0
    neg_weight: float = 1.0
    channel_weight: list[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 5.0]
    )
    lambda_patch: float = 1.0
    # lambda_volume: float = 0.05
    lambda_plane: float = 0.04
    lambda_modality: float = 0.04


@dataclass
class OptimizerConfig:
    """Optimizer configuration"""

    name: str = "AdamW"
    lr: float = 1e-3
    weight_decay: float = 0.01
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    momentum: float = 0.9  # For SGD


@dataclass
class SchedulerConfig:
    """Scheduler configuration"""

    name: str = "CosineAnnealingLR"
    T_max: int = 100
    eta_min: float = 1e-6

    # For ReduceLROnPlateau
    mode: str = "min"
    factor: float = 0.5
    patience: int = 5


@dataclass
class TrainingConfig:
    """Training configuration"""

    batch_size: int = 2
    val_batch_size: int = 4
    epochs: int = 100
    num_workers: int = 8
    output_dir: str = "/home/tamoto/kaggle/RSNA2025/outputs/model9_3dv7_lossv2_mm90_imgsize192"

    # Optimizer and scheduler
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: Optional[SchedulerConfig] = field(default_factory=SchedulerConfig)

    # Training settings
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    p_mixup: float = 0.5
    alpha_mixup: float = 1.0  # Disable for initial test
    use_amp: bool = True
    save_top_k: int = 3

    # Checkpointing
    save_every_n_epochs: int = 5
    validate_every_n_epochs: int = 1

    # Early stopping
    early_stopping_patience: int = 20
    early_stopping_min_delta: float = 0.0001


@dataclass
class Config:
    """Main configuration"""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)
    # Experiment settings
    experiment_name: str = "aneurysm_3d_classification"
    debug: bool = False
    seed: int = 42

    def __post_init__(self):
        """Post initialization"""
        # Ensure consistency
        self.model.num_classes = self.data.num_classes

        # Create output directory
        output_path = Path(self.training.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""

        def dataclass_to_dict(obj):
            if hasattr(obj, "__dataclass_fields__"):
                result = {}
                for field_name in obj.__dataclass_fields__:
                    value = getattr(obj, field_name)
                    if hasattr(value, "__dataclass_fields__"):
                        result[field_name] = dataclass_to_dict(value)
                    else:
                        result[field_name] = value
                return result
            else:
                return obj

        return dataclass_to_dict(self)

    def save(self, path: str):
        """Save configuration to YAML file"""
        import yaml

        config_dict = self.to_dict()
        with open(path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

    @classmethod
    def load(cls, path: str):
        """Load configuration from YAML file"""
        import yaml

        with open(path, "r") as f:
            config_dict = yaml.load(f, Loader=yaml.SafeLoader)

        # Reconstruct dataclasses
        data_dict = config_dict.get("data", {})
        if "augmentation" in data_dict and data_dict["augmentation"]:
            data_dict["augmentation"] = AugmentationConfig(**data_dict["augmentation"])
        data_config = DataConfig(**data_dict)

        model_config = ModelConfig(**config_dict.get("model", {}))
        loss_config = LossConfig(**config_dict.get("loss", {}))

        training_dict = config_dict.get("training", {})
        if "optimizer" in training_dict:
            training_dict["optimizer"] = OptimizerConfig(**training_dict["optimizer"])
        if "scheduler" in training_dict:
            training_dict["scheduler"] = SchedulerConfig(**training_dict["scheduler"])
        training_config = TrainingConfig(**training_dict)

        return cls(
            data=data_config,
            model=model_config,
            loss=loss_config,
            training=training_config,
            experiment_name=config_dict.get("experiment_name", "aneurysm_3d_classification"),
            debug=config_dict.get("debug", False),
            seed=config_dict.get("seed", 42),
        )
