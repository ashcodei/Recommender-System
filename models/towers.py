import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

def _make_backbone(name: str, pretrained=True):
    name = name.lower()
    if name in ("resnet18","resnet34","resnet50"):
        ctor = getattr(models, name)
        weights = models.get_model_weights(name).DEFAULT if pretrained else None
        m = ctor(weights=weights)
        feat_dim = m.fc.in_features
        m.fc = nn.Identity()
        return m, feat_dim
    if name in ("vit_b_16","vit_b_32","vit_l_16"):
        ctor = getattr(models, name)
        weights = models.get_model_weights(name).DEFAULT if pretrained else None
        m = ctor(weights=weights)
        feat_dim = m.heads.head.in_features
        m.heads = nn.Identity()
        return m, feat_dim
    raise ValueError(f"Unsupported backbone: {name}")

class Tower(nn.Module):
    def __init__(self, backbone: str, embedding_dim: int, freeze_backbone: bool = False):
        super().__init__()
        self.backbone, feat_dim = _make_backbone(backbone, pretrained=True)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor):
        h = self.backbone(x)
        z = self.proj(h)
        z = F.normalize(z, dim=-1)
        return z

class TwoTower(nn.Module):
    def __init__(self, backbone: str, embedding_dim: int, freeze_backbone: bool = False, shared: bool = False):
        super().__init__()
        if shared:
            self.query = Tower(backbone, embedding_dim, freeze_backbone)
            self.cand = self.query
        else:
            self.query = Tower(backbone, embedding_dim, freeze_backbone)
            self.cand = Tower(backbone, embedding_dim, freeze_backbone)

    def forward(self, q, c):
        qz = self.query(q)
        cz = self.cand(c)
        return qz, cz
