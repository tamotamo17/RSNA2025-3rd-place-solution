from __future__ import annotations
import inspect
import sys
sys.path.append('src/classification')
sys.path.append('/home/tamoto/kaggle/RSNA2025/MedicalNet')
from collections.abc import Callable
from typing import Any
from types import SimpleNamespace as SN

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

import timm
import timm_3d
import segmentation_models_pytorch as smp
import segmentation_models_pytorch_3d as smp3d

from layers import resnet_block
from MedicalNet.model import generate_model
# ================================
# モデル・レジストリ（3.10+ 型注釈）
# ================================
_MODEL_REGISTRY: dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    使い方:
        @register_model("aneurysm_v2")
        def _make(**kwargs): return AneurysmModelV2(**kwargs)
    """

    def _decorator(factory: Callable[..., Any]) -> Callable[..., Any]:
        key = name.strip().lower()
        if key in _MODEL_REGISTRY:
            raise ValueError(f"model '{name}' is already registered")
        _MODEL_REGISTRY[key] = factory
        return factory

    return _decorator


def _filter_kwargs(factory: Callable[..., Any], kwargs: dict[str, Any]) -> dict[str, Any]:
    """factoryのシグネチャにある引数だけを通す（余分は無視）"""
    sig = inspect.signature(factory)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def create_model(name: str, **kwargs: Any) -> Any:
    """
    例:
      model = create_model(
          "aneurysm_v2",
          backbone=CFG.backbone,
          in_chans=CFG.in_chans,
          out_chans=CFG.out_dim,
          drop_rate=CFG.drop_rate,
          pretrained=CFG.pretrained_weights,
      )
    """
    key = name.strip().lower()
    if key not in _MODEL_REGISTRY:
        # フォールバック: グローバル名前空間に同名のクラス/関数があればそれを使う
        g = globals()
        if name in g and callable(g[name]):
            factory = g[name]  # type: ignore[index]
        else:
            known = ", ".join(sorted(_MODEL_REGISTRY.keys()))
            raise KeyError(f"Unknown model '{name}'. Known: [{known}]")
    else:
        factory = _MODEL_REGISTRY[key]

    return factory(**_filter_kwargs(factory, kwargs))


def list_models() -> list[str]:
    """登録済みモデル名一覧"""
    return sorted(_MODEL_REGISTRY.keys())


@register_model("aneurysm_v1")
def _make_aneurysm_v1(
    backbone: str,
    in_chans: int,
    out_chans: int,
    drop_rate: float = 0.0,
    pretrained: bool = False,
) -> Any:
    return AneurysmModel(
        backbone=backbone,
        in_chans=in_chans,
        out_chans=out_chans,
        drop_rate=drop_rate,
        pretrained=pretrained,
    )


@register_model("aneurysm_v2")
def _make_aneurysm_v2(
    backbone: str,
    in_chans: int,
    out_chans: int,
    drop_rate: float = 0.0,
    pretrained: bool = False,
) -> Any:
    return AneurysmModelV2(
        backbone=backbone,
        in_chans=in_chans,
        out_chans=out_chans,
        drop_rate=drop_rate,
        pretrained=pretrained,
    )


@register_model("patch_classifier_2d")
def _make_patch_classifier_2d(
    encoder_name: str = "timm-efficientnet-b0", 
    encoder_weights: str = "imagenet",
    in_channels: int = 1,
    num_classes: int = 1,
    activation: str = "sigmoid",
    **kwargs
) -> Any:
    return PatchClassifier2D(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights, 
        in_channels=in_channels,
        num_classes=num_classes,
        activation=activation,
    )


@register_model("aneurysm_3d_v1")
def _make_aneurysm_3d_v1(
    backbone: str,
    in_chans: int,
    out_chans: int,
    num_classes: int,
    drop_rate: float = 0.0,
    pretrained: bool = False,
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    coords: bool = False,
) -> Any:
    return Aneurysm3DModel(
        backbone=backbone,
        in_chans=in_chans,
        out_chans=out_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        coords=coords
    )

@register_model("aneurysm_3d_v2")
def _make_aneurysm_3d_v2(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV2(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_v3")
def _make_aneurysm_3d_v3(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV3(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_v4")
def _make_aneurysm_3d_v4(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV4(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_v5")
def _make_aneurysm_3d_v5(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV5(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_v7")
def _make_aneurysm_3d_v7(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV7(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_v8")
def _make_aneurysm_3d_v8(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV8(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_v9")
def _make_aneurysm_3d_v9(
    backbone,
    in_chans,
    num_classes,
    drop_rate=0.0,
    map_size=(4,4,4),
    pretrained=True,
    use_coords=False,
    return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
    multilabel=False,
) -> Any:
    return Aneurysm3DModelV9(
        backbone=backbone,
        in_chans=in_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        map_size=map_size,
        pretrained=pretrained,
        use_coords=use_coords,
        return_logits=return_logits,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=multilabel,
        )

@register_model("aneurysm_3d_mil_v1")
def _make_aneurysm_3d_mil_v1(
    base_model: torch.nn.Module,
    num_classes: int,
    att_activation: str='sigmoid',
    drop_rate: float = 0.0,
    train_encoder: bool = False,
) -> Any:
    return Aneurysm3DMIL(
        base_model=base_model,         # ここから encoder を使う
        num_classes=num_classes,          # MIL の出力クラス数（多ラベルなら 14 など）
        att_activation=att_activation,
        drop_rate=drop_rate,
        multilabel=True,
        train_encoder=train_encoder
    )

@register_model("aneurysm_3d_mil_v2")
def _make_aneurysm_3d_mil_v2(
    backbone: str,
    in_chans: int,
    out_chans: int,
    num_classes: int,
    drop_rate: float = 0.0,
    pretrained: bool = False,
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
    coords: bool = False,
) -> Any:
    return Aneurysm3DMILV2(
        backbone=backbone,
        in_chans=in_chans,
        out_chans=out_chans,
        num_classes=num_classes,
        drop_rate=drop_rate,
        pretrained=pretrained,
        decoder_channels=decoder_channels,
        coords=coords
    )

@register_model("aneurysm_3d_mil_extractor")
def _make_aneurysm_3d_mil_extractor(
    base_model: torch.nn.Module,
    coords: bool
    ) -> Any:
    return Aneurysm3DMILExtractor(
        base_model=base_model,         # ここから encoder を使う
        coords=coords
        )

@register_model("aneurysm_3d_mil_feat")
def _make_aneurysm_3d_mil_feat(
    dim_features: int,
    num_classes: int,
    att_activation: str='sigmoid',
    drop_rate: float = 0.0,
) -> Any:
    return Aneurysm3DFeatureMIL(
        dim_features=dim_features,
        num_classes=num_classes,          # MIL の出力クラス数（多ラベルなら 14 など）
        att_activation=att_activation,
        drop_rate=drop_rate,
        multilabel=True,
        )


@register_model("aneurysm_3d_mil_feat_v2")
def _make_aneurysm_3d_mil_feat_v2(
    dim_features: int,
    num_classes: int,
    att_activation: str='sigmoid',
    drop_rate: float = 0.0,
) -> Any:
    return Aneurysm3DFeatureMILV2(
        dim_features=dim_features,
        num_classes=num_classes,          # MIL の出力クラス数（多ラベルなら 14 など）
        att_activation=att_activation,
        drop_rate=drop_rate,
        multilabel=True,
        )

def init_layer(layer):
    nn.init.xavier_uniform_(layer.weight)

    if hasattr(layer, "bias"):
        if layer.bias is not None:
            layer.bias.data.fill_(0.0)


def init_bn(bn):
    bn.bias.data.fill_(0.0)
    bn.weight.data.fill_(1.0)


def init_weights(model):
    classname = model.__class__.__name__
    if classname.find("Conv2d") != -1:
        nn.init.xavier_uniform_(model.weight, gain=np.sqrt(2))
        model.bias.data.fill_(0)
    elif classname.find("BatchNorm") != -1:
        model.weight.data.normal_(1.0, 0.02)
        model.bias.data.fill_(0)
    elif classname.find("GRU") != -1:
        for weight in model.parameters():
            if len(weight.size()) > 1:
                nn.init.orghogonal_(weight.data)
    elif classname.find("Linear") != -1:
        model.weight.data.normal_(0, 0.01)
        model.bias.data.zero_()


class AttBlockV2(nn.Module):
    def __init__(self, in_features: int, out_features: int, activation="linear"):
        super().__init__()

        self.activation = activation
        self.att = nn.Conv1d(
            in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.cla = nn.Conv1d(
            in_channels=in_features, out_channels=out_features, kernel_size=1, stride=1, padding=0, bias=True
        )

        self.init_weights()

    def init_weights(self):
        init_layer(self.att)
        init_layer(self.cla)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)


class AneurysmModel(nn.Module):
    def __init__(self, backbone, in_chans, out_chans, drop_rate, pretrained=False):
        super(AneurysmModel, self).__init__()
        self.out_chans = out_chans

        self.unet = smp.Unet(
            encoder_name=backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet" if pretrained else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            activation=None,
        )

        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head_mask = self.unet.segmentation_head

        if "efficient" in backbone:
            hdim = self.encoder.out_channels[-1]
        elif "convnext" in backbone:
            self.encoder.head.fc = nn.Identity()
        # self.conv_head = nn.Sequential(nn.Conv2d(hdim, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #                                nn.BatchNorm2d(1280),
        #                                nn.SiLU(inplace=True),
        #                                nn.AdaptiveAvgPool2d(1)
        #                                )
        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(CFG.drop_rate_last),
            # nn.LeakyReLU(0.1),
            nn.Linear(512, out_chans),
        )
        self.head_slice = nn.Sequential(
            nn.Linear(512, out_chans),
        )

    def forward(self, x):  # (bs, nslice, h, w)
        bs, n, h, w = x.shape
        x = x.reshape(int(bs * n), 1, h, w)
        feats = self.encoder(x)
        mask = self.decoder(*feats)
        mask = self.segmentation_head_mask(mask)
        feat = feats[-1].mean(dim=[2, 3])
        feat = feat.reshape(bs, n, -1)
        feat, _ = self.lstm(feat)
        feat_slice = feat.reshape(int(bs * n), -1)
        # feat = self.conv_head(feat)
        feat = feat.mean(1)
        out = self.head(feat)
        out_slice = self.head_slice(feat_slice)
        # print(out_slice.shape)
        out_slice = out_slice.reshape(bs, n, self.out_chans)
        mask = mask.reshape(bs, n, h, w)
        # feat_axt2 = feat_axt2.contiguous().view(bs * CFG.n_slice_sagital, -1)

        # feat = self.head(feat)
        # feat = feat.transpose(1,2)
        # feat, _, _ = self.att_block(feat)
        # feat = feat.view(bs, n_slice_per_c).contiguous()

        return {"output": out, "output_slice": out_slice, "mask": mask}


class Aneurysm3DModel(nn.Module):
    """
    3D セグメンテーション + 分類のデュアルヘッドモデル。
    - セグメンテーション: 3D U-Net
    - 分類: エンコーダ最下層 (bottleneck) のグローバルプーリング + デコーダ最終特徴のコンテキスト要約を結合
    """

    def __init__(
        self,
        backbone: str,
        in_chans: int,
        out_chans: int,                 # 分類の出力次元（クラス数）
        num_classes: int,
        drop_rate: float = 0.0,
        pretrained: bool = False,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        cls_hidden: int = 512,          # 分類 MLP の中間次元
        coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.unet = smp3d.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_chans,
            classes=out_chans,              # binary mask
            activation=None,        # 後段で必要なら sigmoid をかける
            decoder_channels=decoder_channels,
        )

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.unet.encoder.out_channels
        
        bottleneck_channels = encoder_channels[-1]
        if coords:
            bottleneck_channels += 3

        # --- Global pooling for classification ---
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        self.classification_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(bottleneck_channels, num_classes),
        )

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor, coords: torch.Tensor=None) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        encoder_features = self.unet.encoder(x)
        # bottleneck features
        bottleneck_features = encoder_features[-1]
        #print('bottleneck_features:', bottleneck_features.shape)

        # --- Decoder & mask ---
        decoder_features = self.unet.decoder(*encoder_features)         # (B, C_dec, D, H, W)
        mask = self.unet.segmentation_head(decoder_features)            # (B, 1, D, H, W)

        # --- Classification branch: bottleneck global pooling ---
        g = self.global_pool(bottleneck_features).flatten(1)            # (B, C_bottleneck)
        if coords is not None:
            g = torch.concat([g, coords], dim=1)

        logits = self.classification_head(g)                         # (B, out_chans)

        return {"mask": mask, "output": logits}

    

class Aneurysm3DModelV2(nn.Module):
    """
    3D セグメンテーション + 分類のデュアルヘッドモデル。
    - セグメンテーション: 3D U-Net
    - 分類: エンコーダ最下層 (bottleneck) のグローバルプーリング + デコーダ最終特徴のコンテキスト要約を結合
    """

    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = timm_3d.create_model(
            backbone,
            in_chans=in_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
            )
        if 'resnet' in backbone:
            self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
        elif 'convnext' in backbone:
            self.backbone.stem[0] = nn.Conv3d(1, 96, kernel_size=(4, 4, 4), stride=(2, 2, 2))
        else:
            raise ValueError(f'{backbone} is invalid.')
        # self.backbone.layer2[0].conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #self.backbone.layer2[0].downsample = nn.Identity()

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        dim_map = map_size[0] * map_size[1] * map_size[2]
        self.mask_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, num_classes),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dim_map, num_classes),
        )

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        if self.use_coords:
            self.coords = self.coords.to(features.device)
            coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
            features = torch.cat([features, coords], dim= 1)
 
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        features = features.reshape(B*N, C)
        logits = self.mask_head(features)
        logits = logits.view(B, N, -1)
        return {"output": logits}

class Aneurysm3DModelV3(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = timm_3d.create_model(
            backbone,
            in_chans=in_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
            )
        if 'resnet' in backbone:
            self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
            self.backbone.layer2[0].conv1.stride = (1,1,1)
            self.backbone.layer2[0].downsample[0].stride = (1,1,1)
        elif 'convnext' in backbone:
            self.backbone.stem[0] = nn.Conv3d(1, 96, kernel_size=(4, 4, 4), stride=(1, 1, 1))
        else:
            raise ValueError(f'{backbone} is invalid.')
        

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        dim_map = map_size[0] * map_size[1] * map_size[2]
        self.mask_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, num_classes),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dim_map, num_classes),
        )

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        if self.use_coords:
            self.coords = self.coords.to(features.device)
            coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
            features = torch.cat([features, coords], dim= 1)
 
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        features = features.reshape(B*N, C)
        logits = self.mask_head(features)
        logits = logits.view(B, N, -1)
        return {"output": logits}

class Aneurysm3DModelV4(nn.Module):
    """
    3D セグメンテーション + 分類のデュアルヘッドモデル。
    - セグメンテーション: 3D U-Net
    - 分類: エンコーダ最下層 (bottleneck) のグローバルプーリング + デコーダ最終特徴のコンテキスト要約を結合
    """

    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = timm_3d.create_model(
            'resnet18',
            in_chans=in_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
            )
        self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
        # self.backbone.layer2[0].conv1 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        #self.backbone.layer2[0].downsample = nn.Identity()

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        dim_map = map_size[0] * map_size[1] * map_size[2]
        self.mask_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, num_classes),
        )
        self.additional_resnet_block_0 = resnet_block(encoder_channels, 512)
        self.additional_resnet_block_1 = resnet_block(512, 512)
        
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.class_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(512, num_classes),
        )
        

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        if self.use_coords:
            self.coords = self.coords.to(features.device)
            coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
            features = torch.cat([features, coords], dim= 1)
        x = self.additional_resnet_block_0(features)
        x = self.additional_resnet_block_1(x)
        x = self.pool(x).flatten(1)
        logits_vol = self.class_head(x)

 
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        features = features.reshape(B*N, C)
        logits = self.mask_head(features)
        logits = logits.view(B, N, -1)
        
        return {"patch_output": logits,
                "volume_output": logits_vol
                }
    

class Aneurysm3DModelV5(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = timm_3d.create_model(
            backbone,
            in_chans=in_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
            )
        self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
        #self.backbone.layer2[0].conv1.stride = (1,1,1)
        #self.backbone.layer2[0].downsample[0].stride = (1,1,1)

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        self.pool = nn.AdaptiveAvgPool3d(1)

        self.mask_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, num_classes),
        )
        self.mod_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, 4),
        )
        self.plane_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, 3),
        )

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        if self.use_coords:
            self.coords = self.coords.to(features.device)
            coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
            features = torch.cat([features, coords], dim= 1)
        features_agg = self.pool(features).flatten(1) 
        logits_plane = self.plane_head(features_agg)
        logits_mod = self.mod_head(features_agg)
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        features = features.reshape(B*N, C)
        logits = self.mask_head(features)
        logits = logits.view(B, N, -1)
        return {"output": logits,
                "output_plane": logits_plane,
                "output_modality": logits_mod
                }
    
class Aneurysm3DModelV6(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = models.video.r3d_18(weights='KINETICS400_V1')
        if 'resnet' in backbone:
            self.backbone.stem = nn.Conv3d(1, 64, kernel_size=(3, 7, 7), stride=(1, 1, 1), padding=(1, 3, 3), bias=False)
            # self.backbone.stem.stride = (1, 1, 1)
            self.backbone.layer2[0].conv1.stride = (1,1,1)
            self.backbone.layer2[0].downsample[0].stride = (1,1,1)
        elif 'convnext' in backbone:
            self.backbone.stem[0] = nn.Conv3d(1, 96, kernel_size=(4, 4, 4), stride=(1, 1, 1))
        else:
            raise ValueError(f'{backbone} is invalid.')
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = 512# self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        dim_map = map_size[0] * map_size[1] * map_size[2]
        self.mask_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, num_classes),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dim_map, num_classes),
        )

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        if self.use_coords:
            self.coords = self.coords.to(features.device)
            coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
            features = torch.cat([features, coords], dim= 1)
 
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        features = features.reshape(B*N, C)
        logits = self.mask_head(features)
        logits = logits.view(B, N, -1)
        return {"output": logits}
    
class Aneurysm3DModelV7(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = timm_3d.create_model(
            backbone,
            in_chans=in_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
            )
        if 'resnet' in backbone:
            self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
            self.backbone.layer2[0].conv1.stride = (1,1,1)
            self.backbone.layer2[0].downsample[0].stride = (1,1,1)
            self.backbone.layer4[0].conv1.stride = (1,1,1)
            self.backbone.layer4[0].downsample[0].stride = (1,1,1)
        elif 'convnext' in backbone:
            self.backbone.stem[0] = nn.Conv3d(1, 96, kernel_size=(4, 4, 4), stride=(1, 1, 1))
        else:
            raise ValueError(f'{backbone} is invalid.')
        

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        dim_map = map_size[0] * map_size[1] * map_size[2]
        self.mask_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(encoder_channels, num_classes),
        )
        self.class_head = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(dim_map, num_classes),
        )

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        if self.use_coords:
            self.coords = self.coords.to(features.device)
            coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
            features = torch.cat([features, coords], dim= 1)
 
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        features = features.reshape(B*N, C)
        logits = self.mask_head(features)
        logits = logits.view(B, N, -1)
        return {"output": logits}
    
class Aneurysm3DModelV8(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        opt = SN(
            # モデル指定
            model='resnet',          # 'resnet', 'resnext', 'densenet' など
            model_depth=18,          # 10/18/34/50/101 ...

            # 入出力の仕様
            n_classes=1,
            n_seg_classes=num_classes,
            n_input_channels=1,
            sample_size=128,      # H=W の一辺
            sample_duration=128, # D（スライス数/フレーム数）

            input_W=128,
            input_H=128,
            input_D=128,

            no_cuda=False,
            gpu_id=[2],
            phase='train',
            

            # ResNet 系の細部（MedicalNet実装に合わせて）
            shortcut_type='B',
            no_max_pool=False,
            resnet_shortcut='B',     # 実装により不要でも害なし

            # 事前学習重み（あるならパスを指定）
            pretrain_path=str('/home/tamoto/kaggle/RSNA2025/data/pretrain/resnet_18_23dataset.pth'),
            # ↑ 無ければ "" に
        )
        self.backbone = generate_model(opt)

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        # encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3

        # dim_map = map_size[0] * map_size[1] * map_size[2]
        # self.mask_head = nn.Sequential(
        #     nn.Dropout(drop_rate),
        #     nn.Linear(encoder_channels, num_classes),
        # )
        # self.class_head = nn.Sequential(
        #     nn.Dropout(drop_rate),
        #     nn.Linear(dim_map, num_classes),
        # )

    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        # if self.use_coords:
        #     self.coords = self.coords.to(features.device)
        #     coords = self.coords.unsqueeze(0).expand(B, -1, -1, -1, -1)
        #     features = torch.cat([features, coords], dim= 1)
 
        # B, C, Dm, Hm, Wm = features.shape
        # N = Dm*Hm*Wm
        # features = features.view(B, C, N)
        # features = features.transpose(1,2)
        # features = features.reshape(B*N, C)
        # logits = self.mask_head(features)
        # logits = logits.view(B, N, -1)
        return {"output": features}

class Aneurysm3DModelV9(nn.Module):
    
    def __init__(
        self,
        backbone: str,
        in_chans: int,
        num_classes: int,
        drop_rate: float = 0.0,
        map_size: tuple = (4,4,4),
        pretrained: bool = True,
        use_coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.map_size = map_size
        self.use_coords = use_coords
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.backbone = timm_3d.create_model(
            backbone,
            in_chans=in_chans,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''
            )
        if 'resnet' in backbone:
            self.backbone.conv1 = nn.Conv3d(1, 64, kernel_size=(7, 7, 7), stride=(1, 1, 1), padding=(3, 3, 3), bias=False)
            self.backbone.layer2[0].conv1.stride = (1,1,1)
            self.backbone.layer2[0].downsample[0].stride = (1,1,1)
            self.backbone.layer4[0].conv1.stride = (1,1,1)
            self.backbone.layer4[0].downsample[0].stride = (1,1,1)
        elif 'convnext' in backbone:
            self.backbone.stem[0] = nn.Conv3d(1, 96, kernel_size=(4, 4, 4), stride=(1, 1, 1))
        else:
            raise ValueError(f'{backbone} is invalid.')
        
        self.conv_cls = nn.Conv3d(512, 14, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False)
        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.backbone.num_features
        
        if use_coords:
            self.coords = self._build_coords(map_size[0], map_size[1], map_size[2])
            encoder_channels += 3


    @staticmethod
    def _build_coords(D, H, W, center=True, dtype=torch.float32, device=None):
        if center:
            z = (torch.arange(D, device=device, dtype=dtype) + 0.5) / D
            y = (torch.arange(H, device=device, dtype=dtype) + 0.5) / H
            x = (torch.arange(W, device=device, dtype=dtype) + 0.5) / W
        else:
            z = torch.linspace(0, 1, D, device=device, dtype=dtype)
            y = torch.linspace(0, 1, H, device=device, dtype=dtype)
            x = torch.linspace(0, 1, W, device=device, dtype=dtype)
        zz, yy, xx = torch.meshgrid(z, y, x, indexing="ij")
        return torch.stack((xx, yy, zz), dim=0)  # (3, D, H, W)

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)
        B = x.shape[0]

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        features = self.backbone(x)
        features = self.conv_cls(features)
        # bottleneck features
        #print('bottleneck_features:', bottleneck_features.shape)
        # (B, 1, D, H, W)
        B, C, Dm, Hm, Wm = features.shape
        N = Dm*Hm*Wm
        features = features.view(B, C, N)
        features = features.transpose(1,2)
        return {"output": features
        }
    

class AneurysmEncoderAdapter(nn.Module):
    """Aneurysm3DModel の encoder を (B,C_feat) 埋め込みにする薄いラッパー"""
    def __init__(self, base_model, drop_rate: float = 0.0,
                 trainable: bool = True, encoder_eval_when_frozen: bool = True):
        super().__init__()
        self.base = base_model                      # base.unet.encoder を利用
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.feat_dim = int(self.base.unet.encoder.out_channels[-1])

        self.trainable = bool(trainable)
        self.encoder_eval_when_frozen = bool(encoder_eval_when_frozen)
        self._apply_trainable_state(self.trainable)

    @staticmethod
    def _set_requires_grad(module: nn.Module, flag: bool):
        for p in module.parameters():
            p.requires_grad = flag

    def _apply_trainable_state(self, flag: bool):
        self.trainable = bool(flag)
        # 勾配フラグ
        self._set_requires_grad(self.base.unet.encoder, self.trainable)
        # BN/Dropout の挙動を固定したい場合は eval に（オプション）
        if self.encoder_eval_when_frozen:
            if self.trainable:
                self.base.unet.encoder.train()
            else:
                self.base.unet.encoder.eval()

    @torch.no_grad()
    def _forward_encoder_nograd(self, x: torch.Tensor):
        feats = self.base.unet.encoder(x)          # list; 最深部 [-1]
        return feats[-1]                           # (B, C_feat, D', H', W')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # encoder 部分だけ no_grad にしてメモリ節約（pool/dropout は通常どおり）
        if self.trainable:
            feats = self.base.unet.encoder(x)
            bottleneck = feats[-1]
        else:
            bottleneck = self._forward_encoder_nograd(x)

        g = self.pool(bottleneck).flatten(1)        # (B, C_feat)
        return g

    # 外から切り替えたいとき用のAPI
    def set_trainable(self, flag: bool, *, encoder_eval_when_frozen: bool | None = None):
        if encoder_eval_when_frozen is not None:
            self.encoder_eval_when_frozen = bool(encoder_eval_when_frozen)
        self._apply_trainable_state(flag)


class Aneurysm3DMIL(nn.Module):
    """
    Aneurysm3DModel の encoder を流用して MIL 集約（AttBlockV2）。
    入力 x: (B,P,Z,H,W) or (B,P,C,Z,H,W) などを許容。
    """
    def __init__(
        self,
        base_model,                      # Aneurysm3DModel (既存)
        num_classes: int,
        att_activation: str = "linear",
        multilabel: bool = True,
        drop_rate: float = 0.0,
        train_encoder: bool = True,      # ★ 追加：エンコーダ学習のオン/オフ
        encoder_eval_when_frozen: bool = True,  # ★ 追加：凍結時に encoder を eval に
    ):
        super().__init__()
        self.multilabel = multilabel
        self.encoder = AneurysmEncoderAdapter(
            base_model, drop_rate=drop_rate,
            trainable=train_encoder,
            encoder_eval_when_frozen=encoder_eval_when_frozen
        )
        self.att_pool = AttBlockV2(in_features=self.encoder.feat_dim,
                                   out_features=num_classes,
                                   activation=att_activation)

    @staticmethod
    def _ensure_bpcdhw(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """入力を (B,P,C,D,H,W) に整形"""
        if x.dim() == 4:           # (P,D,H,W)
            x = x.unsqueeze(0).unsqueeze(2)            # (1,P,1,D,H,W)
            B, P, C = 1, x.size(1), 1
        elif x.dim() == 5:         # (B,P,D,H,W) or (B,C,D,H,W)
            B = x.size(0)
            if x.size(1) in (1, 3):                     # (B,C,D,H,W)
                x = x.unsqueeze(1)                      # (B,1,C,D,H,W)
                B, P, C = B, 1, x.size(2)
            else:                                       # (B,P,D,H,W)
                x = x.unsqueeze(2)                      # (B,P,1,D,H,W)
                B, P, C = B, x.size(1), 1
        elif x.dim() == 6:         # (B,P,C,D,H,W)
            B, P, C = x.size(0), x.size(1), x.size(2)
        else:
            raise ValueError(f"Expected 4D/5D/6D, got {tuple(x.shape)}")
        return x, B, P, C

    # 外から切り替えるためのラッパ
    def set_encoder_trainable(self, flag: bool, *, encoder_eval_when_frozen: bool | None = None):
        self.encoder.set_trainable(flag, encoder_eval_when_frozen=encoder_eval_when_frozen)

    def extract_feature(
        self,
        x: torch.Tensor,
        *,
        detach: bool = False,
        normalize: bool = False,
    ) -> torch.Tensor:
        """
        入力 x をエンコーダに通し、(B, P, C_feat) の特徴を返す。

        Args:
            x: (B,P,Z,H,W) or (B,P,C,Z,H,W) など（_ensure_bpcdhwが整形）
            detach: True のとき .detach() して返す（勾配不要の推論用）
            normalize: True のとき L2 正規化（dim=-1）を適用

        Returns:
            feats_bp: torch.Tensor, 形状 (B, P, C_feat)
        """
        # (B,P,C,D,H,W) へ整形
        x, B, P, C = self._ensure_bpcdhw(x)
        x = x.reshape(B * P, C, *x.shape[-3:])  # (B*P, C, D, H, W)

        # エンコーダで特徴抽出：（B*P, C_feat）
        feats = self.encoder(x)                 # AneurysmEncoderAdapter が trainable/no_grad を内部で制御

        # (B, P, C_feat) へ整形
        feats_bp = feats.view(B, P, -1)

        if normalize:
            feats_bp = F.normalize(feats_bp, p=2, dim=-1)
        if detach:
            feats_bp = feats_bp.detach()

        return feats_bp

    def forward(self, x: torch.Tensor, return_probs: bool = False, return_instance: bool = True):
        x, B, P, C = self._ensure_bpcdhw(x)
        x = x.reshape(B*P, C, *x.shape[-3:])                   # (B*P,C,D,H,W)

        feats = self.encoder(x)                                # (B*P, C_feat)
        C_feat = feats.size(1)
        feats = feats.view(B, P, C_feat).transpose(1, 2)       # (B, C_feat, P)

        bag_logits, att_weights, instance_scores = self.att_pool(feats)
        out = {"output": bag_logits, "att_weights": att_weights}  # att: (B,num_classes,P)

        if return_probs:
            out["bag_probs"] = torch.sigmoid(bag_logits) if self.multilabel else F.softmax(bag_logits, dim=-1)
        if return_instance:
            out["instance_logits"] = instance_scores.transpose(1, 2).contiguous()  # (B,P,num_classes)
        return out
    

class Aneurysm3DMILV2(nn.Module):
    """
    3D セグメンテーション + 分類のデュアルヘッドモデル。
    - セグメンテーション: 3D U-Net
    - 分類: エンコーダ最下層 (bottleneck) のグローバルプーリング + デコーダ最終特徴のコンテキスト要約を結合
    """

    def __init__(
        self,
        backbone: str,
        in_chans: int,
        out_chans: int,                 # 分類の出力次元（クラス数）
        num_classes: int,
        drop_rate: float = 0.0,
        pretrained: bool = False,
        decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16),
        cls_hidden: int = 512,          # 分類 MLP の中間次元
        coords: bool = False,
        return_logits: bool = True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel: bool = False,       # True のとき多ラベル想定で sigmoid, False で softmax
    ):
        super().__init__()
        self.backbone = backbone
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.return_logits = return_logits
        self.multilabel = multilabel

        # --- 3D U-Net for segmentation ---
        self.unet = smp3d.Unet(
            encoder_name=backbone,
            encoder_weights="imagenet" if pretrained else None,
            in_channels=in_chans,
            classes=out_chans,              # binary mask
            activation=None,        # 後段で必要なら sigmoid をかける
            decoder_channels=decoder_channels,
        )

        # --- Encoder/Decoder channel meta ---
        # encoder 出力チャンネル列: [stage0, stage1, ..., bottleneck]
        # 例: [3, 64, 96, 128, 160, 256] など (backbone に依存)
        encoder_channels = self.unet.encoder.out_channels
        
        bottleneck_channels = encoder_channels[-1]
        if coords:
            bottleneck_channels += 3

        # --- Global pooling for classification ---
        self.att_pool = AttBlockV2(in_features=bottleneck_channels,
                                   out_features=num_classes,
                                   activation="sigmoid"
                                   )

        # self.classification_head = nn.Sequential(
        #     nn.Dropout(drop_rate),
        #     nn.Linear(bottleneck_channels, num_classes),
        # )

    def _ensure_5d(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, D, H, W) を期待。チャネル欠落時の救済。
        if x.dim() == 4:
            # (B, D, H, W) -> (B, 1, D, H, W)
            x = x.unsqueeze(1)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input (B,C,D,H,W), got shape {tuple(x.shape)}")
        return x

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Args:
            x: (B, C, D, H, W) あるいは (B, D, H, W)
        Returns:
            {
                'mask': (B, 1, D, H, W)  # 生出力。必要なら外側で sigmoid
                'output': (B, out_chans) # 分類 (logits or probabilities)
            }
        """
        x = self._ensure_5d(x)

        # --- Encoder features ---
        # segmentation-models-pytorch 系は encoder(x) で段階ごとの特徴リストを返す実装
        encoder_features = self.unet.encoder(x)
        # bottleneck features
        bottleneck_features = encoder_features[-1]
        #print('bottleneck_features:', bottleneck_features.shape)

        # --- Decoder & mask ---
        decoder_features = self.unet.decoder(*encoder_features)         # (B, C_dec, D, H, W)
        mask = self.unet.segmentation_head(decoder_features)            # (B, 1, D, H, W)
        B, C, Dm, Hm, Wm = bottleneck_features.shape
        bottleneck_features = bottleneck_features.view(B, C, Dm*Hm*Wm)
        # --- Classification branch: bottleneck global pooling ---
        logits, _, _ = self.att_pool(bottleneck_features)            # (B, C_bottleneck)
        
        return {"mask": mask, "output": logits}
    

class Aneurysm3DMILExtractor(nn.Module):
    """
    Aneurysm3DModel の encoder を流用して MIL 集約（AttBlockV2）。
    入力 x: (B,P,Z,H,W) or (B,P,C,Z,H,W) などを許容。
    """
    def __init__(
        self,
        base_model,                      # Aneurysm3DModel (既存)
        coords
        ):
        super().__init__()
        self.encoder = base_model.unet.encoder
        self.global_pool = base_model.global_pool
        self.coords = coords

        self.classification_head = base_model.classification_head

    @staticmethod
    def _ensure_bpcdhw(x: torch.Tensor) -> tuple[torch.Tensor, int, int, int]:
        """入力を (B,P,C,D,H,W) に整形"""
        if x.dim() == 4:           # (P,D,H,W)
            x = x.unsqueeze(0).unsqueeze(2)            # (1,P,1,D,H,W)
            B, P, C = 1, x.size(1), 1
        elif x.dim() == 5:         # (B,P,D,H,W) or (B,C,D,H,W)
            B = x.size(0)
            if x.size(1) in (1, 3):                     # (B,C,D,H,W)
                x = x.unsqueeze(1)                      # (B,1,C,D,H,W)
                B, P, C = B, 1, x.size(2)
            else:                                       # (B,P,D,H,W)
                x = x.unsqueeze(2)                      # (B,P,1,D,H,W)
                B, P, C = B, x.size(1), 1
        elif x.dim() == 6:         # (B,P,C,D,H,W)
            B, P, C = x.size(0), x.size(1), x.size(2)
        else:
            raise ValueError(f"Expected 4D/5D/6D, got {tuple(x.shape)}")
        return x, B, P, C

    # 外から切り替えるためのラッパ
    def set_encoder_trainable(self, flag: bool, *, encoder_eval_when_frozen: bool | None = None):
        self.encoder.set_trainable(flag, encoder_eval_when_frozen=encoder_eval_when_frozen)

    def forward(self, x: torch.Tensor, coords: torch.Tensor=None, return_probs: bool = False):
        x, B, P, C = self._ensure_bpcdhw(x)
        x = x.reshape(B*P, C, *x.shape[-3:])                   # (B*P,C,D,H,W)

        feats = self.encoder(x)                                # (B*P, C_feat)
        bottleneck_feats = feats[-1]
        feats = self.global_pool(bottleneck_feats).flatten(1)
        if coords is not None:
            B, P, C_coords = coords.shape
            coords = coords.view(B*P, C_coords)
            feats = torch.concat([feats, coords], dim=1)
        C_feat = feats.size(1)
        out = self.classification_head(feats)
        feats = feats.view(B, P, C_feat)
        
        out = out.view(B, P, -1)
        outs = {"output": out, "feature": feats}  # att: (B,num_classes,P)
        if return_probs:
            outs["bag_probs"] = torch.sigmoid(out)
        return outs
    
class Aneurysm3DFeatureMIL(nn.Module):
    """
    Aneurysm3DModel の encoder を流用して MIL 集約（AttBlockV2）。
    入力 x: (B,P,C)
    """
    def __init__(
        self,
        dim_features: int,
        num_classes: int,
        att_activation: str = "linear",
        multilabel: bool = True,
        drop_rate: float = 0.0,
        ):
        super().__init__()
        self.multilabel = multilabel
        self.att_pool = AttBlockV2(in_features=dim_features,
                                   out_features=num_classes,
                                   activation=att_activation)
        # self.linear = nn.Linear(dim_features, num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False, return_instance: bool = True):
        feats = x.transpose(1, 2)       # (B, C_feat, P)
        # feats, _ = feats.max(2)
        # bag_logits = self.linear(feats)

        bag_logits, att_weights, instance_scores = self.att_pool(feats)
        out = {"output": bag_logits}  # att: (B,num_classes,P)

        # if return_probs:
        #     out["bag_probs"] = torch.sigmoid(bag_logits) if self.multilabel else F.softmax(bag_logits, dim=-1)
        # if return_instance:
        #     out["instance_logits"] = instance_scores.transpose(1, 2).contiguous()  # (B,P,num_classes)
        return out
    

class Aneurysm3DFeatureMILV2(nn.Module):
    """
    Aneurysm3DModel の encoder を流用して MIL 集約（AttBlockV2）。
    入力 x: (B,P,C)
    """
    def __init__(
        self,
        dim_features: int,
        num_classes: int,
        att_activation: str = "linear",
        multilabel: bool = True,
        drop_rate: float = 0.0,
        ):
        super().__init__()
        self.multilabel = multilabel
        # self.att_pool = AttBlockV2(in_features=dim_features,
        #                            out_features=num_classes,
        #                            activation=att_activation)
        self.linear = nn.Linear(dim_features, num_classes)

    def forward(self, x: torch.Tensor, return_probs: bool = False, return_instance: bool = True):
        x = x[...,13]
        x = x.unsqueeze(2)
        BS, P, D = x.shape
        feats = x.view(BS, P*D)       # (B, C_feat, P)
        # feats, _ = feats.max(2)
        bag_logits = self.linear(feats)

        # bag_logits, att_weights, instance_scores = self.att_pool(feats)
        out = {"output": bag_logits}  # att: (B,num_classes,P)

        # if return_probs:
        #     out["bag_probs"] = torch.sigmoid(bag_logits) if self.multilabel else F.softmax(bag_logits, dim=-1)
        # if return_instance:
        #     out["instance_logits"] = instance_scores.transpose(1, 2).contiguous()  # (B,P,num_classes)
        return out


@torch.no_grad()
def load_encoder_weights(
    base_model: Aneurysm3DModel,
    ckpt_path: str,
    map_location: str | torch.device = "cpu",
    verbose: bool = True,
):
    """
    ckpt から 'unet.encoder.*' のみ抽出して base_model.unet.encoder にロードする。
    - DDP などの 'module.' / 'model.' prefix にも対応
    - 分類/セグヘッド形状が違っても安全（encoder だけ読む）
    """
    ckpt = torch.load(ckpt_path, map_location=map_location)
    state = ckpt.get("state_dict", ckpt)

    def strip_prefix(k: str) -> str:
        for p in ("module.", "model."):
            if k.startswith(p):
                return k[len(p):]
        return k

    enc_only = {}
    for k, v in state.items():
        k2 = strip_prefix(k)
        if k2.startswith("unet.encoder."):
            enc_only[k2.replace("unet.encoder.", "", 1)] = v  # encoder に相対キーで渡す

    missing, unexpected = base_model.unet.encoder.load_state_dict(enc_only, strict=False)
    if verbose:
        print(f"[MIL] Loaded encoder keys: {len(enc_only)}")
        if missing:
            print(f"[MIL] Missing keys in encoder: {len(missing)} (ok if minor):\n  {missing[:10]}")
        if unexpected:
            print(f"[MIL] Unexpected keys (ignored): {len(unexpected)}:\n  {unexpected[:10]}")

class AneurysmModelV2(nn.Module):
    def __init__(self, backbone, in_chans, out_chans, drop_rate, pretrained=False):
        super(AneurysmModelV2, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.unet = smp.Unet(
            encoder_name=backbone,  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
            encoder_weights="imagenet" if pretrained else None,  # use `imagenet` pre-trained weights for encoder initialization
            in_channels=in_chans,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
            classes=1,  # model output channels (number of classes in your dataset)
            activation=None,
        )

        self.encoder = self.unet.encoder
        self.decoder = self.unet.decoder
        self.segmentation_head_mask = self.unet.segmentation_head

        if "efficient" in backbone:
            hdim = self.encoder.out_channels[-1]
        elif "convnext" in backbone:
            self.encoder.head.fc = nn.Identity()
        # self.conv_head = nn.Sequential(nn.Conv2d(hdim, 1280, kernel_size=(1, 1), stride=(1, 1), bias=False),
        #                                nn.BatchNorm2d(1280),
        #                                nn.SiLU(inplace=True),
        #                                nn.AdaptiveAvgPool2d(1)
        #                                )
        self.lstm = nn.LSTM(hdim, 256, num_layers=2, dropout=drop_rate, bidirectional=True, batch_first=True)
        self.head = nn.Sequential(
            # nn.Linear(512, 256),
            # nn.BatchNorm1d(256),
            # nn.Dropout(CFG.drop_rate_last),
            # nn.LeakyReLU(0.1),
            nn.Linear(512, out_chans),
        )
        self.head_slice = nn.Sequential(
            nn.Linear(512, out_chans),
        )

    def __reshape(self, x, reverse=False):
        """
        reshape helper:
        - reverse=False: (bs, n, h, w) -> (bs*(n//n_group), n_group, h, w)
        - reverse=True: (bs*(n//n_group), ...) -> (bs, n//n_group, ...)
        """
        n_group = self.in_chans
        if not reverse:
            bs, n, h, w = x.shape
            if n % n_group != 0:
                raise ValueError("n must be divisible by n_group")
            x = x.reshape(bs, n // n_group, n_group, h, w)
            x = x.reshape(bs * (n // n_group), n_group, h, w)
        else:
            # 後処理側
            bs_ng, *rest = x.shape
            # 推定 bs
            bs = bs_ng // (self.n_slices_per_sample)
            x = x.reshape(bs, self.n_slices_per_sample, *rest)
        return x

    def forward(self, x):  # (bs, nslice, h, w)
        bs, n, h, w = x.shape
        self.n_slices_per_sample = n // self.in_chans
        x = self.__reshape(x, reverse=False)
        feats = self.encoder(x)
        mask = self.decoder(feats)
        mask = self.segmentation_head_mask(mask)
        feat = feats[-1].mean(dim=[2, 3])
        feat = feat.reshape(bs, self.n_slices_per_sample, -1)
        feat, _ = self.lstm(feat)
        feat_slice = feat.reshape(int(bs * self.n_slices_per_sample), -1)
        # feat = self.conv_head(feat)
        feat = feat.mean(1)
        out = self.head(feat)
        out_slice = self.head_slice(feat_slice)
        # print(out_slice.shape)
        out_slice = out_slice.reshape(bs, self.n_slices_per_sample, self.out_chans)
        mask = mask.reshape(bs, self.n_slices_per_sample, h, w)
        # feat_axt2 = feat_axt2.contiguous().view(bs * CFG.n_slice_sagital, -1)

        # feat = self.head(feat)
        # feat = feat.transpose(1,2)
        # feat, _, _ = self.att_block(feat)
        # feat = feat.view(bs, n_slice_per_c).contiguous()

        return {"output": out, "output_slice": out_slice, "mask": mask}


class PatchClassifier2D(nn.Module):
    """
    Simple 2D patch classifier for binary aneurysm classification
    Uses timm models for efficient and clean implementation
    """
    
    def __init__(
        self,
        encoder_name: str = "timm-efficientnet-b0",
        encoder_weights: str = "imagenet",
        in_channels: int = 1,
        num_classes: int = 1,
        activation: str = "sigmoid",
        drop_rate: float = 0.2,
    ):
        super().__init__()
        
        self.num_classes = num_classes
        self.activation = activation
        
        # Create timm model
        self.encoder = timm.create_model(
            encoder_name,
            pretrained=(encoder_weights == "imagenet"),
            in_chans=in_channels,
            num_classes=0,  # Remove classification head, use features only
            global_pool='',  # Remove global pooling, we'll add our own
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy_input = torch.randn(1, in_channels, 224, 224)
            features = self.encoder(dummy_input)
            if isinstance(features, (tuple, list)):
                features = features[-1]  # Take last feature map
            feature_dim = features.shape[1]  # Channel dimension
        
        # Add classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(feature_dim, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor (B, C, H, W) for 2D patches or (B, C, D, H, W) for 3D patches
        Returns:
            Dictionary with 'output' key containing classification logits
        """
        
        # Handle 3D input by taking middle slice or processing all slices
        if x.dim() == 5:  # (B, C, D, H, W)
            B, C, D, H, W = x.shape
            # Take middle slice for now - could be enhanced to process all slices
            middle_slice = D // 2
            x = x[:, :, middle_slice, :, :]  # (B, C, H, W)
        
        # Extract features
        features = self.encoder(x)  # (B, feature_dim, H', W')
        
        # Global pooling and classification
        pooled = self.global_pool(features)  # (B, feature_dim, 1, 1)
        pooled = pooled.flatten(1)  # (B, feature_dim)
        
        logits = self.classifier(pooled)  # (B, num_classes)
        
        # Apply activation if specified
        if self.activation == "sigmoid":
            output = torch.sigmoid(logits)
        elif self.activation == "softmax":
            output = torch.softmax(logits, dim=1)
        else:
            output = logits  # Return raw logits
            
        return {"output": logits}  # Always return logits for loss computation


if __name__ == "__main__":
    # Test 2D model
    backbone = "timm-efficientnet-b0"  #'tf_efficientnetv2_s_in21ft1k' #"tf_efficientnet_b0.ns_jft_in1k" #
    in_chans = 1
    out_chans = 14
    drop_rate = 0.2
    pretrained = "noisy-student"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input = torch.randn(2, 30, 224, 224, device=device)

    print("Testing 2D model (aneurysm_v1):")
    # m = create_model(
    #     "aneurysm_v1",
    #     backbone=backbone,
    #     in_chans=in_chans,
    #     out_chans=out_chans,
    #     drop_rate=drop_rate,
    #     pretrained=pretrained,
    # ).to(device)
    # out = m(dummy_input)
    # for k, v in out.items():
    #     print(f"  {k}: {v.shape}")
    opt = SN(
        # モデル指定
        model='resnet',          # 'resnet', 'resnext', 'densenet' など
        model_depth=18,          # 10/18/34/50/101 ...

        # 入出力の仕様
        n_classes=1,
        n_seg_classes=1,
        n_input_channels=1,
        sample_size=128,      # H=W の一辺
        sample_duration=128, # D（スライス数/フレーム数）

        input_W=128,
        input_H=128,
        input_D=128,

        no_cuda=False,
        gpu_id=[2],
        phase='train',
        

        # ResNet 系の細部（MedicalNet実装に合わせて）
        shortcut_type='B',
        no_max_pool=False,
        resnet_shortcut='B',     # 実装により不要でも害なし

        # 事前学習重み（あるならパスを指定）
        pretrain_path=str('/home/tamoto/kaggle/RSNA2025/data/pretrain/resnet_18_23dataset.pth'),
        # ↑ 無ければ "" に
    )
    # model = generate_model(opt)
    # print(model)
    # Test 3D model
    print("\nTesting 3D model (aneurysm_3d_v1):")
    backbone_3d = "resnet18"  # 3D backbone
    dummy_input_3d = torch.randn(2, 1, 128, 128, 128, device=device)  # (batch, channel, depth, height, width)
    # model.conv_seg = nn.Identity()
    # out = model(dummy_input_3d)
    # print(out.shape)
    # m_3d = create_model(
    #     "aneurysm_3d_v1",
    #     backbone=backbone_3d,
    #     in_chans=1,
    #     out_chans=1,
    #     num_classes=14,
    #     drop_rate=0.2,
    #     pretrained=False,  # 3D models typically don't have ImageNet pretraining
    #     decoder_channels=(256, 128, 64, 32, 16),
    #     return_logits=True
    # ).to(device)
    m_3d = Aneurysm3DModelV9(
        backbone=backbone_3d,
        in_chans=1,
        num_classes=14,
        drop_rate=0.2,
        map_size=(8,8,8),
        pretrained=False,
        use_coords=False,
        return_logits=True,     # True: raw logits / False: softmax (多クラス) or sigmoid (多ラベル) を返す
        multilabel=False,
    ).to(device)
    print(m_3d)
    out_3d = m_3d(dummy_input_3d)
    for k, v in out_3d.items():
        print(f"  {k}: {v.shape}")

    B, P = 2, 16
    Z, H, W = 96, 96, 96
    dummy_bag = torch.randn(B, P, Z, H, W, device=device)  # (B,P,Z,H,W)  ← in_chans=1想定
    
    # m = models.video.r3d_18(weights='KINETICS400_V1')
    # print(m)
    # MIL モデル（encoder は m_3d のものを流用）
    # pretrained_weight_path = '/home/tamoto/kaggle/RSNA2025/outputs/exp010_dsv2_multi_epoch100/models/best_fold0.pth'

    # m_3d.load_state_dict(torch.load(pretrained_weight_path, map_location=('cpu')), strict=False)
    # mil = Aneurysm3DMIL(
    #     base_model=m_3d,         # ここから encoder を使う
    #     num_classes=14,          # MIL の出力クラス数（多ラベルなら 14 など）
    #     att_activation="sigmoid",
    #     multilabel=True,
    #     drop_rate=0.1,
    #     train_encoder=False,
    # ).to(device).train()         # 学習モードで勾配も確認

    # out_mil = mil(dummy_bag, return_probs=True, return_instance=True)

    # print("\n[MILFromAneurysm3D] outputs:")
    # for k, v in out_mil.items():
    #     print(f"  {k:>16}: {tuple(v.shape)}")

    # mil_extractor = Aneurysm3DMILExtractor(
    #     base_model=m_3d
    # )
    # out_mil_exr = mil_extractor(dummy_bag, return_probs=True)
    
    # print("\n[MILExtractorFromAneurysm3D] outputs:")
    # for k, v in out_mil_exr.items():
    #     print(f"  {k:>16}: {tuple(v.shape)}, {v.min()}, {v.max()}")
