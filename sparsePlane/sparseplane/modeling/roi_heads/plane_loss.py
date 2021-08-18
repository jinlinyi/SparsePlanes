import torch
import torch.nn as nn
import quaternion
from torch.nn.utils.rnn import pad_sequence


class GeoConsistencyLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, device, max_num_planes=20):
        super(GeoConsistencyLoss, self).__init__()
        self.device = torch.device(device)
        self.max_num_planes = max_num_planes

    def pack_data(
        self, pred_instances1, pred_instances2, cam_poses, batched_inputs=None
    ):
        if isinstance(pred_instances1[0], dict):
            has_instances_key = True
        else:
            has_instances_key = False

        numPlanes1 = []
        numPlanes2 = []
        planeparam1 = []
        planeparam2 = []
        if False:
            for x, cam_pose in zip(pred_instances1, cam_poses):
                if has_instances_key:
                    planeparam_tmp = x["instances"].pred_plane
                else:
                    planeparam_tmp = x.pred_plane
                planeparam_tmp = self.get_plane_params_in_global(
                    planeparam_tmp, cam_pose
                )
                planeparam1.append(planeparam_tmp)
                numPlanes1.append([len(planeparam_tmp)])
            planeparam1 = pad_sequence(planeparam1, batch_first=True)
            for x in pred_instances2:
                if has_instances_key:
                    planeparam_tmp = x["instances"].pred_plane
                else:
                    planeparam_tmp = x.pred_plane
                planeparam_tmp = planeparam_tmp * torch.FloatTensor([1, -1, -1]).to(
                    self.device
                )
                planeparam2.append(planeparam_tmp)
                numPlanes2.append([len(planeparam_tmp)])
            planeparam2 = pad_sequence(planeparam2, batch_first=True)
        else:
            for x, cam_pose in zip(pred_instances1, cam_poses):
                if has_instances_key:
                    if "pred_plane" in x.keys():
                        planeparam_tmp = x["pred_plane"]
                    else:
                        planeparam_tmp = x["instances"].pred_plane
                else:
                    planeparam_tmp = x.gt_planes
                planeparam_tmp = self.get_plane_params_in_global(
                    planeparam_tmp, cam_pose
                )
                planeparam1.append(planeparam_tmp)
                numPlanes1.append([len(planeparam_tmp)])
            planeparam1 = pad_sequence(planeparam1, batch_first=True)
            for x in pred_instances2:
                if has_instances_key:
                    if "pred_plane" in x.keys():
                        planeparam_tmp = x["pred_plane"]
                    else:
                        planeparam_tmp = x["instances"].pred_plane
                else:
                    planeparam_tmp = x.gt_planes
                planeparam_tmp = planeparam_tmp * torch.FloatTensor([1, -1, -1]).to(
                    self.device
                )
                planeparam2.append(planeparam_tmp)
                numPlanes2.append([len(planeparam_tmp)])
            planeparam2 = pad_sequence(planeparam2, batch_first=True)

        gt_corr_ms = []
        if batched_inputs is not None:
            for x1, x2, x in zip(pred_instances1, pred_instances2, batched_inputs):
                gt_plane_idx1 = x1.gt_plane_idx
                gt_plane_idx2 = x2.gt_plane_idx
                gt_corr = x["gt_corrs"]
                gt_corr_m = torch.zeros(
                    [len(planeparam1[0]), len(planeparam2[0])], dtype=torch.bool
                )
                for corr in gt_corr:
                    x1_idx = (gt_plane_idx1 == corr[0]).nonzero()[:, 0]
                    x2_idx = (gt_plane_idx2 == corr[1]).nonzero()[:, 0]
                    for i in x1_idx:
                        for j in x2_idx:
                            gt_corr_m[i, j] = True
                gt_corr_ms.append(gt_corr_m)
            gt_corr_ms = torch.stack(gt_corr_ms)
        return (
            planeparam1,
            planeparam2,
            gt_corr_ms,
            numPlanes1,
            numPlanes2,
        )

    def forward(
        self, batched_inputs, pred_instances1, pred_instances2, cam_poses, loss_weight
    ):
        (
            planeparam1,
            planeparam2,
            gt_corr_ms,
            numPlanes1,
            numPlanes2,
        ) = self.pack_data(pred_instances1, pred_instances2, cam_poses, batched_inputs)
        # torch.cdist is similar to scipy.spatial.distance.cdist
        # input: planeparam1 B*N1*D, planeparam2 B*N2*D,
        geo_distance_matrix = torch.cdist(planeparam1, planeparam2, p=2)
        losses = geo_distance_matrix[gt_corr_ms].mean()
        if torch.isnan(losses):
            losses = 0
        return {"geo_consistency_loss": loss_weight * losses}

    def inference(self, pred_instances1, pred_instances2, cam_poses, distance="l2"):
        planeparam1, planeparam2, _, numPlanes1, numPlanes2 = self.pack_data(
            pred_instances1, pred_instances2, cam_poses
        )
        # torch.cdist is similar to scipy.spatial.distance.cdist
        # input: embedding1 B*N1*D, embedding2 B*N2*D,
        # output: B*N1*N2. Each entry is ||e1-e2||
        geo_distance_matrix = {}
        if distance == "l2":
            geo_distance_matrix["l2"] = torch.cdist(planeparam1, planeparam2, p=2)
        elif distance == "normal":
            offset1 = torch.norm(planeparam1, dim=2)[:, :, None]
            normal1 = planeparam1 / offset1
            offset2 = torch.norm(planeparam2, dim=2)[:, :, None]
            normal2 = planeparam2 / offset2
            nTn = torch.bmm(normal1, normal2.transpose(1, 2))
            geo_distance_matrix["offset"] = torch.abs(offset1 - offset2.transpose(1, 2))
            geo_distance_matrix["offset"][nTn < 0] = torch.abs(
                offset1 + offset2.transpose(1, 2)
            )[nTn < 0]
            geo_distance_matrix["normal"] = torch.acos(
                torch.clamp(torch.abs(nTn), -1, 1)
            )  # in rad
        else:
            raise NotImplementedError

        return geo_distance_matrix, numPlanes1, numPlanes2

    def get_plane_params_in_global(self, planes, camera_info):
        """
        input:
        @planes: plane params
        @camera_info: plane params from camera info, type = dict, must contain 'position' and 'rotation' as keys
        output:
        plane parameters in global frame.
        """
        tran = torch.FloatTensor(camera_info["position"]).to(self.device)
        rot = quaternion.from_float_array(camera_info["rotation"])
        start = torch.ones((len(planes), 3)).to(self.device) * tran
        end = planes * torch.tensor([1, -1, -1]).to(self.device)  # suncg2habitat
        end = (
            torch.mm(
                torch.FloatTensor(quaternion.as_rotation_matrix(rot)).to(self.device),
                (end).T,
            ).T
            + tran
        )  # cam2world
        a = end
        b = end - start
        planes_world = ((a * b).sum(dim=1) / (torch.norm(b, dim=1) + 1e-5) ** 2).view(-1, 1) * b
        return planes_world
