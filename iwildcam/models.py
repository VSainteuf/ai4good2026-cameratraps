from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tvm

BACKBONES = {
    "resnet18": (tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (tvm.resnet34, tvm.ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V1),
}


class ResNetClassifier(nn.Module):
    """A torchvision ResNet with the ImageNet head replaced by a species head."""

    def __init__(self, n_classes: int, backbone: str = "resnet50",
                 pretrained: bool = True, dropout: float = 0.0, freeze_bn: bool = False):
        """Build the backbone and attach a fresh head.

        Args:
            n_classes: number of species, the output dimension.
            backbone: one of `BACKBONES` -- resnet18, resnet34 or resnet50.
            pretrained: start from ImageNet weights rather than from scratch.
            dropout: dropout probability before the head. 0 disables it.
            freeze_bn: keep BatchNorm running statistics fixed during training.

        Raises:
            ValueError: if `backbone` is not in `BACKBONES`.
        """
        super().__init__()
        if backbone not in BACKBONES:
            raise ValueError(f"unknown backbone {backbone!r}, have {list(BACKBONES)}")
        ctor, weights = BACKBONES[backbone]
        net = ctor(weights=weights if pretrained else None)
        self.d_feat = net.fc.in_features
        net.fc = nn.Identity()
        self.backbone = net
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.head = nn.Linear(self.d_feat, n_classes)
        self.freeze_bn = freeze_bn

    def train(self, mode: bool = True):
        """Set training mode, keeping BatchNorm in eval mode if `freeze_bn`.

        Args:
            mode: True for training, False for evaluation.

        Returns:
            self, as `nn.Module.train` does.
        """
        super().train(mode)
        if mode and self.freeze_bn:
            for m in self.backbone.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The representation an alignment or adversarial loss should read.

        Args:
            x: image batch, `(batch, 3, size, size)`.

        Returns:
            Pooled pre-head features, `(batch, d_feat)`.
        """
        return self.backbone(x)

    def logits(self, feats: torch.Tensor) -> torch.Tensor:
        """Class scores from features you have already computed.

        Use this when you need both the representation and the prediction, so the
        backbone runs once:

            feats  = model.features(x)
            logits = model.logits(feats)

        Calling `model(x)` and `model.features(x)` on the same batch runs the backbone
        twice and updates BatchNorm statistics twice per step. It still trains, which is
        what makes the mistake easy to miss.

        Args:
            feats: output of `features()`, `(batch, d_feat)`.

        Returns:
            Class scores, `(batch, n_classes)`.
        """
        return self.head(self.dropout(feats))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Classify a batch of images.

        Args:
            x: image batch, `(batch, 3, size, size)`.

        Returns:
            Class scores, `(batch, n_classes)`.
        """
        return self.logits(self.features(x))


def build_model(name: str, n_classes: int, **kw) -> nn.Module:
    """Build the model named in a config.

    Args:
        name: backbone name, e.g. `"resnet18"`.
        n_classes: number of species.
        **kw: passed to `ResNetClassifier` -- `pretrained`, `dropout`, `freeze_bn`.

    Returns:
        A `ResNetClassifier`.
    """
    return ResNetClassifier(n_classes, backbone=name, **kw)
