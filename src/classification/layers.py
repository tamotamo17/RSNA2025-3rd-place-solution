import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


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
    def __init__(self, in_features: int, out_features: int, activation: str = "linear"):
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
        """
        x: (batch, n, dim)
        """
        norm_att = torch.softmax(torch.tanh(self.att(x)), dim=-1)
        cla = self.nonlinear_transform(self.cla(x))
        x = torch.sum(norm_att * cla, dim=2)
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == "linear":
            return x
        elif self.activation == "sigmoid":
            return torch.sigmoid(x)
        
def conv3x3x3(in_ch, out_ch, stride=1, padding=1, bias=False):
    return nn.Conv3d(in_ch, out_ch, kernel_size=3, stride=stride, padding=padding, bias=bias)

def conv1x1x1(in_ch, out_ch, stride=1, bias=False):
    return nn.Conv3d(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=bias)

class BasicBlock3d(nn.Module):
    """Matches the printed structure:
    conv1 -> bn1 -> drop_block(Identity) -> act1(ReLU) -> aa(Identity) -> conv2 -> bn2 -> (+down) -> act2(ReLU)
    """
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, zero_init_last=True):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm3d(planes)
        self.drop_block = nn.Identity()   # placeholder for DropBlock
        self.act1  = nn.ReLU(inplace=True)
        self.aa    = nn.Identity()        # placeholder for anti-alias (e.g., BlurPool)

        self.conv2 = conv3x3x3(planes, planes, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm3d(planes)
        self.act2  = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                conv1x1x1(inplanes, planes * self.expansion, stride=stride, bias=False),
                nn.BatchNorm3d(planes * self.expansion),
            )

        if zero_init_last:
            nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.drop_block(out)
        out = self.act1(out)
        out = self.aa(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.act2(out)
        return out

# ---- layer4 (Sequential) as shown in your print ----
def resnet_block(inplanes, planes):
    layer = nn.Sequential(
        # (0): BasicBlock with stride 2 and downsample 1x1x1, in:256 out:512
        BasicBlock3d(inplanes=inplanes, planes=planes, stride=2),
        # (1): BasicBlock with stride 1, in:512 out:512
        BasicBlock3d(inplanes=planes, planes=planes, stride=1),
    )
    return layer