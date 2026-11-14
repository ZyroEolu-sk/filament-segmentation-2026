"""U-Net con encoder preentrenado (transfer learning desde ImageNet).

smp adapta la primera convolución a 1 canal sumando los pesos RGB, así que el
encoder conserva lo aprendido en ImageNet. Salidas (logits): 0 = filamento,
1 = borde de instancia.
"""

from __future__ import annotations

import segmentation_models_pytorch as smp
import torch
from torch import nn

MEAN, STD = 0.45, 0.225  # escala ImageNet en gris


class FilamentNet(nn.Module):
    def __init__(self, arch: str = "Unet", encoder: str = "resnet34", weights: str | None = "imagenet",
                 decoder_attention_type: str | None = None, no_maxpool: bool = False):
        super().__init__()
        kwargs = {"decoder_attention_type": decoder_attention_type} if decoder_attention_type is not None else {}
        self.net = getattr(smp, arch)(encoder_name=encoder, encoder_weights=weights, in_channels=1, classes=2,
                                      **kwargs)
        if no_maxpool:
            if not encoder.startswith("resnet") or not isinstance(self.net.encoder.maxpool, nn.MaxPool2d):
                raise ValueError("no_maxpool solo está disponible para encoders ResNet")
            self.net.encoder.maxpool = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - MEAN) / STD)


def build(cfg: dict, pretrained: bool = True) -> FilamentNet:
    model = FilamentNet(cfg["arch"], cfg["encoder"], "imagenet" if pretrained else None,
                       cfg.get("decoder_attention_type"), cfg.get("no_maxpool", False))
    # Opt-in for attention encoders: native-resolution tiles avoid quadratic
    # attention memory at 2048px. Old configurations retain full-image inference.
    model.inference_tile = cfg.get("inference_tile", 0)
    model.inference_overlap = cfg.get("inference_overlap", 128)
    return model


class SkeletonRecallLoss(nn.Module):
    """1 - recall blando sobre el esqueleto tubular del GT (Kirchhoff et al., ECCV'24).

    Penaliza sobre todo los huecos que parten un filamento: un trozo de esqueleto sin predicción cuesta
    igual aunque sea fino. Es la versión barata de clDice (no hay que esqueletizar la predicción).
    """

    def forward(self, logits: torch.Tensor, skel: torch.Tensor) -> torch.Tensor:
        p = torch.sigmoid(logits)
        has = skel.flatten(1).sum(1) > 0
        if not has.any():
            return logits.sum() * 0
        recall = (p * skel).flatten(1).sum(1) / (skel.flatten(1).sum(1) + 1e-6)
        return 1 - recall[has].mean()


class SegLoss(nn.Module):
    """Filamento: (BCE | focal) + Dice [+ Skeleton Recall]; borde: BCE ponderada + Dice (clase muy rara).

    target tiene 3 canales: filamento, borde, esqueleto (este último solo se usa en la loss); un 4.º canal opcional
    marca los píxeles válidos del filamento (objetivo consensus-ignore): en los inválidos el logit se fija a un valor
    muy negativo con objetivo 0, así BCE/focal y Dice no reciben gradiente ni contribución de esos píxeles.
    """

    def __init__(self, boundary_weight: float = 0.5, fg_loss: str = "bce", skel_weight: float = 0.0):
        super().__init__()
        self.dice = smp.losses.DiceLoss("binary", from_logits=True)
        # focal (alpha 0.66, gamma 2, como MORDEN): baja el peso de los píxeles fáciles de fondo y otras manchas oscuras
        self.fg = (smp.losses.FocalLoss("binary", alpha=0.66, gamma=2.0) if fg_loss == "focal" else nn.BCEWithLogitsLoss())
        self.bce_b = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(5.0))
        self.skel = SkeletonRecallLoss()
        self.bw, self.sw = boundary_weight, skel_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        lf, tf = logits[:, :1].contiguous(), target[:, :1].contiguous()  # FocalLoss de smp usa .view
        if target.shape[1] > 3:
            ignore = target[:, 3:4] < 0.5
            lf, tf = lf.masked_fill(ignore, -1e4), tf.masked_fill(ignore, 0.0)
        fg = self.fg(lf, tf) + self.dice(lf, tf)
        b = self.bce_b(logits[:, 1:2], target[:, 1:2]) + self.dice(logits[:, 1:2], target[:, 1:2])
        loss = fg + self.bw * b
        if self.sw:
            loss = loss + self.sw * self.skel(logits[:, :1], target[:, 2:3])
        return loss
