import torch
from torch import nn
from detectron2.utils.registry import Registry

__all__ = ["build_depth_head", "PlaneRCNNDepthHead", "DEPTH_HEAD_REGISTRY"]

DEPTH_HEAD_REGISTRY = Registry("DEPTH_HEAD")
DEPTH_HEAD_REGISTRY.__doc__ = """
Registry for depth head in a generalized R-CNN model.
ROIHeads take feature maps and predict depth.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`nn.module`.
"""


def l1LossMask(pred, gt, mask):
    """L1 loss with a mask"""
    return torch.sum(torch.abs(pred - gt) * mask) / torch.clamp(mask.sum(), min=1)


def build_depth_head(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.DEPTH_HEAD.NAME
    return DEPTH_HEAD_REGISTRY.get(name)(cfg)


def conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=None):
    return nn.Sequential(
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.LeakyReLU(inplace=True),
    )


def deconv2d(
    scale_factor=2,
    mode="nearest",
    in_channels=256,
    out_channels=128,
    kernel_size=3,
    stride=1,
    padding=1,
):
    return nn.Sequential(
        torch.nn.Upsample(scale_factor=scale_factor, mode=mode),
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        ),
        nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.01),
        nn.ReLU(inplace=True),
    )


@DEPTH_HEAD_REGISTRY.register()
class PlaneRCNNDepthHead(nn.Module):
    """
    The depth head for Plane RCNN
    """

    def __init__(self, cfg):
        super(PlaneRCNNDepthHead, self).__init__()
        num_output_channels = 1
        self.conv1 = conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv2 = conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv3 = conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv4 = conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.conv5 = conv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv1 = deconv2d(
            in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv2 = deconv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv3 = deconv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv4 = deconv2d(
            in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1
        )
        self.deconv5 = deconv2d(
            in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1
        )
        self.depth_pred = nn.Conv2d(
            64, num_output_channels, kernel_size=3, stride=1, padding=1
        )
        self._loss_weight = cfg.MODEL.DEPTH_HEAD.LOSS_WEIGHT

    def forward(self, features, gt_depth=None):
        """
        p2 256*120*160  -> 256*120*120
        p3 256*60*80    -> 256*60*60
        p4 256*30*40    -> 256*30*30
        p5 256*15*20    -> 256*15*20
        p6 256*8*10     -> 256*16*20
        """
        id2name = {0: "p6", 1: "p5", 2: "p4", 3: "p3", 4: "p2"}
        x = self.deconv1(self.conv1(features[id2name[0]]))
        x = torch.nn.functional.interpolate(
            x, size=(15, 20), mode="bilinear", align_corners=False
        )
        x = self.deconv2(torch.cat([self.conv2(features[id2name[1]]), x], dim=1))
        x = self.deconv3(torch.cat([self.conv3(features[id2name[2]]), x], dim=1))
        x = self.deconv4(torch.cat([self.conv4(features[id2name[3]]), x], dim=1))
        x = self.deconv5(torch.cat([self.conv5(features[id2name[4]]), x], dim=1))
        x = self.depth_pred(x)
        x = torch.nn.functional.interpolate(
            x, size=(480, 640), mode="bilinear", align_corners=False
        )
        pred_depth = x.view(-1, 480, 640)
        losses = {}
        if self.training:
            loss = l1LossMask(pred_depth, gt_depth, (gt_depth > 1e-4).float())
            losses.update({"depth_loss": self._loss_weight * loss})
            if torch.isnan(loss):
                import pdb; pdb.set_trace()
                pass
            return losses
        else:
            return pred_depth
