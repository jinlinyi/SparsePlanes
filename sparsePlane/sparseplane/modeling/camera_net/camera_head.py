import torch
import fvcore.nn.weight_init as weight_init
from torch import nn
from torch.nn import functional as F
from detectron2.utils.registry import Registry

__all__ = ["build_camera_head", "PlaneRCNNCameraHead", "CAMERA_HEAD_REGISTRY"]

CAMERA_HEAD_REGISTRY = Registry("CAMERA_HEAD")
CAMERA_HEAD_REGISTRY.__doc__ = """
Registry for camera head in a generalized R-CNN model.
It takes feature maps from two images and predict relative camera transformation.
The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`nn.module`.
"""


def build_camera_head(cfg):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.CAMERA_HEAD.NAME
    return CAMERA_HEAD_REGISTRY.get(name)(cfg)


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


@CAMERA_HEAD_REGISTRY.register()
class PlaneRCNNCameraHead(nn.Module):
    """
    The camera head for Plane RCNN
    """

    def __init__(self, cfg):
        super(PlaneRCNNCameraHead, self).__init__()
        self.backbone_feature = cfg.MODEL.CAMERA_HEAD.BACKBONE_FEATURE
        if self.backbone_feature == "p5":
            self.convs_backbone = nn.Sequential(
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            )
        elif self.backbone_feature == "p4":
            self.convs_backbone = nn.Sequential(
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                ),
                conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            )
        elif self.backbone_feature == "p3":
            self.convs_backbone = nn.Sequential(
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                ),
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
                nn.MaxPool2d(
                    kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False
                ),
                conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                conv2d(in_channels=512, out_channels=512, kernel_size=3, padding=1),
            )
            for block in self.convs_backbone:
                if isinstance(block, nn.modules.container.Sequential):
                    for layer in block:
                        if isinstance(layer, nn.Conv2d):
                            weight_init.c2_msra_fill(layer)
        else:
            raise NotImplementedError
        self.convs = nn.Sequential(
            conv2d(in_channels=300, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        self.fc = nn.Linear(768, cfg.MODEL.CAMERA_HEAD.FEATURE_SIZE)
        self.trans_branch = nn.Linear(
            cfg.MODEL.CAMERA_HEAD.FEATURE_SIZE, cfg.MODEL.CAMERA_HEAD.TRANS_CLASS_NUM
        )
        self.rot_branch = nn.Linear(
            cfg.MODEL.CAMERA_HEAD.FEATURE_SIZE, cfg.MODEL.CAMERA_HEAD.ROTS_CLASS_NUM
        )
        self._loss_weight = cfg.MODEL.CAMERA_HEAD.LOSS_WEIGHT
        self.celoss = nn.CrossEntropyLoss()

    def forward(self, features1, features2, gt_cls=None):
        """
        p2 256*120*160
        p3 256*60*80
        p4 256*30*40
        p5 256*15*20
        p6 256*8*10
        """
        x1 = self.backbone(features1)
        x2 = self.backbone(features2)
        aff = self.compute_corr_softmax(x1, x2)
        x = self.convs(aff)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc(x))
        trans = self.trans_branch(x)
        rot = self.rot_branch(x)
        if self.training:
            tran_loss = self.celoss(trans, gt_cls["tran_cls"].squeeze(1))
            rot_loss = self.celoss(rot, gt_cls["rot_cls"].squeeze(1))
            losses = {"camera_loss": tran_loss + rot_loss}
            if torch.isnan(tran_loss) or torch.isnan(rot_loss):
                import pdb; pdb.set_trace()
            return losses
        else:
            return {"tran": trans, "rot": rot}

    def backbone(self, features):
        x = self.convs_backbone(features[self.backbone_feature])
        return x

    def compute_corr_softmax(self, im_feature1, im_feature2):
        _, _, h1, w1 = im_feature1.size()
        _, _, h2, w2 = im_feature2.size()
        im_feature2 = im_feature2.transpose(2, 3)
        im_feature2_vec = im_feature2.contiguous().view(
            im_feature2.size(0), im_feature2.size(1), -1
        )
        im_feature2_vec = im_feature2_vec.transpose(1, 2)
        im_feature1_vec = im_feature1.contiguous().view(
            im_feature1.size(0), im_feature1.size(1), -1
        )
        corrfeat = torch.matmul(im_feature2_vec, im_feature1_vec)
        corrfeat = corrfeat.view(corrfeat.size(0), h2 * w2, h1, w1)
        corrfeat = F.softmax(corrfeat, dim=1)
        return corrfeat
