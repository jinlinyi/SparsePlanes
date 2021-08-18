import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(
        np.logical_and(loss_values < margin, loss_values > 0)
    )[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector:
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, distance_matrix, gt_corr_ms, numPlanes1, numPlanes2):
        batch_size, row_len, column_len = distance_matrix.size()
        # create mask
        masks = np.ones(distance_matrix.size())
        for batch_idx, (num1, num2) in enumerate(zip(numPlanes1, numPlanes2)):
            masks[batch_idx][: num1[0], : num2[0]] = 0
        distance_matrix_masked = distance_matrix.cpu().data.numpy()
        distance_matrix_masked[masks > 0.5] = 100
        # Reshape distance_matrix
        distance_matrix_flatten = distance_matrix_masked.reshape(-1, column_len)

        gt_corr_ms_flatten = gt_corr_ms.reshape(-1, column_len)
        # add offset to gt_corr and remove redundent corrs
        anchor_positives = gt_corr_ms_flatten.nonzero()
        selected_rows = distance_matrix_flatten[anchor_positives[:, 0], :]
        ap_distances = distance_matrix_flatten[
            anchor_positives[:, 0], anchor_positives[:, 1]
        ].reshape(-1, 1)
        positive_mask = gt_corr_ms_flatten[anchor_positives[:, 0], :]
        loss_total = ap_distances - selected_rows + self.margin
        assert (np.array(positive_mask.size()) == loss_total.shape).all()
        try:
            loss_total[positive_mask] = 0
        except:
            print(loss_total)
            print(positive_mask)
            return torch.LongTensor([[0, 0, 0]])
        # select semihard_negatives
        triplets = []
        for loss, anchor_positive in zip(loss_total, anchor_positives):
            negative_idx = self.negative_selection_fn(loss)
            if negative_idx is not None:
                triplets.append([anchor_positive[0], anchor_positive[1], negative_idx])
        if len(triplets) == 0:
            # print("triplet length is 0")
            triplets.append([0, 0, 0])
        triplets = np.array(triplets)
        return torch.LongTensor(triplets)


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, device, selector_type, max_num_planes=20):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        if selector_type == "semihard":
            self.triplet_selector = FunctionNegativeTripletSelector(
                margin=margin,
                negative_selection_fn=lambda x: semihard_negative(x, margin),
            )
        elif selector_type == "hardest":
            self.triplet_selector = FunctionNegativeTripletSelector(
                margin=margin, negative_selection_fn=hardest_negative
            )
        elif selector_type == "random":
            self.triplet_selector = FunctionNegativeTripletSelector(
                margin=margin, negative_selection_fn=random_hard_negative
            )
        else:
            raise NotImplementedError
        self.device = torch.device(device)
        self.max_num_planes = max_num_planes

    def pack_data(self, pred_instances1, pred_instances2, batched_inputs=None):
        if isinstance(pred_instances1[0], dict):
            has_instances_key = True
        else:
            has_instances_key = False

        embeddings1 = []
        embeddings2 = []
        numPlanes1 = []
        numPlanes2 = []
        for x in pred_instances1:
            if has_instances_key:
                embedding_tmp = x["instances"].embedding
            else:
                embedding_tmp = x.embedding
            embeddings1.append(embedding_tmp)
            numPlanes1.append([len(embedding_tmp)])
        embeddings1 = pad_sequence(embeddings1, batch_first=True)
        for x in pred_instances2:
            if has_instances_key:
                embedding_tmp = x["instances"].embedding
            else:
                embedding_tmp = x.embedding
            embeddings2.append(embedding_tmp)
            numPlanes2.append([len(embedding_tmp)])
        embeddings2 = pad_sequence(embeddings2, batch_first=True)

        gt_corr_ms = []
        if batched_inputs is not None:
            for x1, x2, x in zip(pred_instances1, pred_instances2, batched_inputs):
                gt_plane_idx1 = x1.gt_plane_idx
                gt_plane_idx2 = x2.gt_plane_idx
                gt_corr = x["gt_corrs"]
                gt_corr_m = torch.zeros(
                    [len(embeddings1[0]), len(embeddings2[0])], dtype=torch.bool
                )
                for corr in gt_corr:
                    x1_idx = (gt_plane_idx1 == corr[0]).nonzero()[:, 0]
                    x2_idx = (gt_plane_idx2 == corr[1]).nonzero()[:, 0]
                    for i in x1_idx:
                        for j in x2_idx:
                            gt_corr_m[i, j] = True
                gt_corr_ms.append(gt_corr_m)
            gt_corr_ms = torch.stack(gt_corr_ms)
        return embeddings1, embeddings2, gt_corr_ms, numPlanes1, numPlanes2

    def forward(self, batched_inputs, pred_instances1, pred_instances2, loss_weight):
        embeddings1, embeddings2, gt_corr_ms, numPlanes1, numPlanes2 = self.pack_data(
            pred_instances1, pred_instances2, batched_inputs
        )
        # torch.cdist is similar to scipy.spatial.distance.cdist
        # input: embedding1 B*N1*D, embedding2 B*N2*D,
        # output: B*N1*N2. Each entry is ||e1-e2||
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)
        # get triplets
        triplets = self.triplet_selector.get_triplets(
            distance_matrix, gt_corr_ms, numPlanes1, numPlanes2
        )
        distance_matrix_flatten = distance_matrix.view(-1, distance_matrix.size(2))
        # calculate loss
        ap_distances = distance_matrix_flatten[triplets[:, 0], triplets[:, 1]]
        an_distances = distance_matrix_flatten[triplets[:, 0], triplets[:, 2]]
        losses = F.relu(ap_distances - an_distances + self.margin)
        return {"embedding_loss": loss_weight * losses.mean()}

    def inference(self, pred_instances1, pred_instances2):
        embeddings1, embeddings2, _, numPlanes1, numPlanes2 = self.pack_data(
            pred_instances1, pred_instances2
        )
        # torch.cdist is similar to scipy.spatial.distance.cdist
        # input: embedding1 B*N1*D, embedding2 B*N2*D,
        # output: B*N1*N2. Each entry is ||e1-e2||
        distance_matrix = torch.cdist(embeddings1, embeddings2, p=2)
        return distance_matrix, numPlanes1, numPlanes2


class OnlineRelaxMatchLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, device, max_num_planes=20):
        super(OnlineRelaxMatchLoss, self).__init__()
        self.margin = margin
        self.device = torch.device(device)
        self.max_num_planes = max_num_planes

    def pack_data(self, pred_instances1, pred_instances2, batched_inputs=None):
        numCorr = []
        gt_corr = []
        if batched_inputs is not None:
            assert "gt_corrs" in batched_inputs[0]
            for x in batched_inputs:
                numCorr.append(torch.tensor([len(x["gt_corrs"])]))
                gt_corr.append(torch.tensor(x["gt_corrs"]))
            gt_corr = pad_sequence(gt_corr, batch_first=True).numpy()
        if isinstance(pred_instances1[0], dict):
            has_instances_key = True
        else:
            has_instances_key = False

        embeddings1 = []
        embeddings2 = []
        numPlanes1 = []
        numPlanes2 = []
        for x in pred_instances1:
            if has_instances_key:
                embedding_tmp = x["instances"].embedding
            else:
                embedding_tmp = x.embedding
            embeddings1.append(embedding_tmp)
            numPlanes1.append([len(embedding_tmp)])
        embeddings1 = pad_sequence(embeddings1, batch_first=True)
        for x in pred_instances2:
            if has_instances_key:
                embedding_tmp = x["instances"].embedding
            else:
                embedding_tmp = x.embedding
            embeddings2.append(embedding_tmp)
            numPlanes2.append([len(embedding_tmp)])
        embeddings2 = pad_sequence(embeddings2, batch_first=True)
        return embeddings1, embeddings2, gt_corr, numPlanes1, numPlanes2, numCorr

    def get_gt_corr_matrix(self, gt_corr, num_corr, pred):
        corr_ms = []
        zeros = torch.zeros_like(pred[0])
        for corr, num in zip(gt_corr, num_corr):
            corr_m = zeros.clone()
            for i, j in corr[: num[0]]:
                corr_m[i, j] = 1.0
            corr_ms.append(corr_m)
        return torch.stack(corr_ms).cuda()

    def forward(self, batched_inputs, pred_instances1, pred_instances2, loss_weight):
        (
            embeddings1,
            embeddings2,
            gt_corr,
            numPlanes1,
            numPlanes2,
            numCorr,
        ) = self.pack_data(pred_instances1, pred_instances2, batched_inputs)
        affinity_matrix = torch.sigmoid(
            torch.bmm(embeddings1, torch.transpose(embeddings2, 1, 2)) * 5.0
        )
        gt_corr_m = self.get_gt_corr_matrix(gt_corr, numCorr, affinity_matrix)
        zero_grad_mask = (affinity_matrix < 1 - self.margin) * (1 - gt_corr_m)
        losses = F.mse_loss(
            affinity_matrix * (1 - zero_grad_mask),
            gt_corr_m * (1 - zero_grad_mask),
            reduction="sum",
        )
        losses /= len(batched_inputs)
        return {"embedding_loss": loss_weight * losses}

    def inference(self, pred_instances1, pred_instances2):
        embeddings1, embeddings2, _, numPlanes1, numPlanes2, _ = self.pack_data(
            pred_instances1, pred_instances2
        )
        affinity_matrix = torch.sigmoid(
            torch.bmm(embeddings1, torch.transpose(embeddings2, 1, 2)) * 5.0
        )
        affinity_matrix[affinity_matrix < 1 - self.margin] = 0.0
        pred_corr_m = affinity_matrix
        return pred_corr_m, numPlanes1, numPlanes2


class CooperativeTripletLoss(nn.Module):
    """
    Loss for ASNet
    Cosine similarity as weight
    """

    def __init__(self, margin, device, selector_type, max_num_planes=20):
        super(CooperativeTripletLoss, self).__init__()
        self.margin = margin
        if selector_type == "semihard":
            self.triplet_selector = FunctionNegativeTripletSelector(
                margin=margin,
                negative_selection_fn=lambda x: semihard_negative(x, margin),
            )
        elif selector_type == "hardest":
            self.triplet_selector = FunctionNegativeTripletSelector(
                margin=margin, negative_selection_fn=hardest_negative
            )
        elif selector_type == "random":
            self.triplet_selector = FunctionNegativeTripletSelector(
                margin=margin, negative_selection_fn=random_hard_negative
            )
        else:
            raise NotImplementedError
        self.device = torch.device(device)
        self.max_num_planes = max_num_planes

    def pack_data(self, pred_instances1, pred_instances2, batched_inputs=None):
        if isinstance(pred_instances1[0], dict):
            has_instances_key = True
        else:
            has_instances_key = False

        embeddings1_c = []
        embeddings1_s = []
        embeddings2_c = []
        embeddings2_s = []
        numPlanes1 = []
        numPlanes2 = []
        for x in pred_instances1:
            if has_instances_key:
                embedding_tmp_c = x["instances"].embedding_c
                embedding_tmp_s = x["instances"].embedding_s
            else:
                embedding_tmp_c = x.embedding_c
                embedding_tmp_s = x.embedding_s
            embeddings1_c.append(embedding_tmp_c)
            embeddings1_s.append(embedding_tmp_s)
            numPlanes1.append([len(embedding_tmp_c)])
        embeddings1_c = pad_sequence(embeddings1_c, batch_first=True)
        embeddings1_s = pad_sequence(embeddings1_s, batch_first=True)
        for x in pred_instances2:
            if has_instances_key:
                embedding_tmp_c = x["instances"].embedding_c
                embedding_tmp_s = x["instances"].embedding_s
            else:
                embedding_tmp_c = x.embedding_c
                embedding_tmp_s = x.embedding_s
            embeddings2_c.append(embedding_tmp_c)
            embeddings2_s.append(embedding_tmp_s)
            numPlanes2.append([len(embedding_tmp_c)])
        embeddings2_c = pad_sequence(embeddings2_c, batch_first=True)
        embeddings2_s = pad_sequence(embeddings2_s, batch_first=True)

        gt_corr_ms = []
        if batched_inputs is not None:
            for x1, x2, x in zip(pred_instances1, pred_instances2, batched_inputs):
                gt_plane_idx1 = x1.gt_plane_idx
                gt_plane_idx2 = x2.gt_plane_idx
                gt_corr = x["gt_corrs"]
                gt_corr_m = torch.zeros(
                    [len(embeddings1_c[0]), len(embeddings2_c[0])], dtype=torch.bool
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
            embeddings1_c,
            embeddings1_s,
            embeddings2_c,
            embeddings2_s,
            gt_corr_ms,
            numPlanes1,
            numPlanes2,
        )

    def forward(self, batched_inputs, pred_instances1, pred_instances2, loss_weight):
        (
            embeddings1_c,
            embeddings1_s,
            embeddings2_c,
            embeddings2_s,
            gt_corr_ms,
            numPlanes1,
            numPlanes2,
        ) = self.pack_data(pred_instances1, pred_instances2, batched_inputs)
        center_dist = torch.cdist(embeddings1_c, embeddings2_c, p=2)
        surrnd_dist = torch.cdist(embeddings1_s, embeddings2_s, p=2)
        surrnd_weigt = torch.cos(torch.asin(torch.clamp(surrnd_dist / 2, -1, 1)) * 2)

        distance_matrix = (1 - surrnd_weigt) * center_dist + surrnd_weigt * surrnd_dist

        triplets = self.triplet_selector.get_triplets(
            distance_matrix, gt_corr_ms, numPlanes1, numPlanes2
        )
        distance_matrix_flatten = distance_matrix.view(-1, distance_matrix.size(2))
        # calculate loss
        ap_distances = distance_matrix_flatten[triplets[:, 0], triplets[:, 1]]
        an_distances = distance_matrix_flatten[triplets[:, 0], triplets[:, 2]]
        losses = F.relu(ap_distances - an_distances + self.margin)
        return {"embedding_loss": loss_weight * losses.mean()}

    def inference(self, pred_instances1, pred_instances2):
        (
            embeddings1_c,
            embeddings1_s,
            embeddings2_c,
            embeddings2_s,
            _,
            numPlanes1,
            numPlanes2,
        ) = self.pack_data(pred_instances1, pred_instances2)

        center_dist = torch.cdist(embeddings1_c, embeddings2_c, p=2)
        surrnd_dist = torch.cdist(embeddings1_s, embeddings2_s, p=2)
        surrnd_weigt = torch.cos(torch.asin(torch.clamp(surrnd_dist / 2, -1, 1)) * 2)

        distance_matrix = (1 - surrnd_weigt) * center_dist + surrnd_weigt * surrnd_dist
        return distance_matrix, numPlanes1, numPlanes2
