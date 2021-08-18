import torch
import numpy as np


@torch.no_grad()
def compare_planes(
    pred_planes,
    gt_planes,
):
    """
    naively calculate 3d vector l2 distance
    """
    pred_planes = torch.tensor(np.array(pred_planes), dtype=torch.float32)
    pred_offsets = torch.norm(pred_planes, p=2, dim=1) + 1e-5
    pred_norms = pred_planes.div(pred_offsets.view(-1, 1).expand_as(pred_planes))
    gt_planes = torch.tensor(np.array(gt_planes), dtype=torch.float32)
    gt_offsets = torch.norm(gt_planes, p=2, dim=1) + 1e-5
    gt_norms = gt_planes.div(gt_offsets.view(-1, 1).expand_as(gt_planes))
    norm_distance_matrix = torch.clamp(torch.cdist(pred_norms, gt_norms, p=2), 0, 2)
    norm_angle_matrix = 2 * torch.asin(norm_distance_matrix / 2) / np.pi * 180
    offset_distance_matrix = torch.cdist(
        pred_offsets.view(-1, 1), gt_offsets.view(-1, 1), p=1
    )
    return {"norm": norm_angle_matrix, "offset": offset_distance_matrix}


def compare_planes_one_to_one(
    pred_planes,
    gt_planes,
):
    pred_planes = torch.tensor(np.array(pred_planes), dtype=torch.float32)
    pred_offsets = torch.clamp(torch.norm(pred_planes, p=2, dim=1), min=1e-5)
    pred_norms = pred_planes.div(pred_offsets.view(-1, 1).expand_as(pred_planes))
    gt_planes = torch.tensor(np.array(gt_planes), dtype=torch.float32)
    gt_offsets = torch.clamp(torch.norm(gt_planes, p=2, dim=1), min=1e-5)
    gt_norms = gt_planes.div(gt_offsets.view(-1, 1).expand_as(gt_planes))

    l2 = torch.norm(pred_planes - gt_planes, dim=1).numpy().mean()
    norm = (
        torch.acos(torch.clamp(torch.sum(pred_norms * gt_norms, dim=1), max=1, min=-1))
        .numpy()
        .mean()
    )
    offset = torch.abs(pred_offsets - gt_offsets).numpy().mean()
    return {"l2": l2, "norm": norm, "offset": offset}
