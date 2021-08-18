# Definition of PoseNet
# author: ynie
# date: March, 2020

import torch
import torch.nn as nn
import math
import numpy as np


class Relation_Config(object):
    def __init__(self):
        self.d_g = 64
        self.d_k = 64
        self.Nr = 16

rel_cfg = Relation_Config()


def get_g_features(bbox2d_poses, d_model=int(rel_cfg.d_g/4)):
    pes = []
    split = []
    rel_pair_counts = [0]
    obj_count = 0
    for bbox2d_pos in bbox2d_poses:
        bbox2d_pos = bbox2d_pos.to("cpu").tensor.numpy()
        n_objects = len(bbox2d_pos)
        g_feature = [[((loc2[0] + loc2[2]) / 2. - (loc1[0] + loc1[2]) / 2.) / (loc1[2] - loc1[0]),
                    ((loc2[1] + loc2[3]) / 2. - (loc1[1] + loc1[3]) / 2.) / (loc1[3] - loc1[1]),
                    math.log((loc2[2] - loc2[0]) / (loc1[2] - loc1[0])),
                    math.log((loc2[3] - loc2[1]) / (loc1[3] - loc1[1]))] \
                    for id1, loc1 in enumerate(bbox2d_pos)
                    for id2, loc2 in enumerate(bbox2d_pos)]

        locs = np.array(g_feature).flatten()

        pe = torch.zeros(len(locs), d_model)
        position = torch.from_numpy(np.array(locs)).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pes.append(pe.view(n_objects * n_objects, rel_cfg.d_g).to("cuda"))
        split.append([obj_count, n_objects + obj_count])
        obj_count += n_objects
        rel_pair_counts.append(rel_pair_counts[-1] + n_objects ** 2)
    pes = torch.cat(pes)
    split = torch.tensor(split)
    rel_pair_counts = torch.tensor(rel_pair_counts)

    return pes, split, rel_pair_counts


class RelationNet(nn.Module):
    def __init__(self):
        super(RelationNet, self).__init__()

        # branch to estimate geometric weights
        self.fc_g = nn.Linear(rel_cfg.d_g, rel_cfg.Nr)
        self.threshold = nn.Threshold(1e-6, 1e-6)
        self.softmax = nn.Softmax(dim=1)

        # branch to estimate appearance weights
        self.fc_K = nn.Linear(1024, rel_cfg.d_k * rel_cfg.Nr)
        self.fc_Q = nn.Linear(1024, rel_cfg.d_k * rel_cfg.Nr)

        # # to ensemble appearance and geometric feature
        # self.fc_V = nn.Linear(2048, 2048)

        # control scale
        self.conv_s = nn.Conv1d(1,1,1)


    def forward(self, a_features, g_features, split, rel_pair_counts):
        '''
        Extract relational features from appearance feature and geometric feature (see Hu et al. [2]).
        :param a_features: Patch_size x 2048 (Appearance_feature_size)
        a_features records the ResNet-34 feature for each object in Patch.
        :param g_features: SUM(N_i^2) x 64 (i.e. Number_of_object_pairs x Geometric_feature_size)
        g_features records the geometric features (64-D) between each pair of objects (see Hu et al. [2]). So the dimension
        is Number_of_pairs_in_images x 64 (or SUM(N_i^2) x 64). N_i is the number of objects in the i-th image.
        :param split: Batch_size x 2
        split records which batch a object belongs to.
        e.g. split = torch.tensor([[0, 5], [5, 8]]) when batch size is 2, and there are 5 objects in the first batch and
        3 objects in the second batch.
        Then the first 5 objects in the whole patch belongs to the first batch, and the rest belongs to the second batch.
        :param rel_pair_counts: (Batch_size + 1)
        rel_pair_counts records which batch a geometric feature belongs to, and gives the start and end index.
        e.g. rel_pair_counts = torch.tensor([0, 49, 113]).
        The batch size is two. The first 49 geometric features are from the first batch.
        The index begins from 0 and ends at 49. The second 64 geometric features are from the second batch.
        The index begins from 49 and ends at 113.
        :return: Relational features for each object.
        '''
        # branch to estimate geometric weights
        g_weights = self.fc_g(g_features)
        g_weights = self.threshold(g_weights)
        # Nr x num_pairs_in_batch x dim
        g_weights = g_weights.transpose(0, 1)

        # branch to estimate appearance weights
        k_features = self.fc_K(a_features)
        q_features = self.fc_Q(a_features)

        # divided by batch and relational group
        # Nr x num_objects_in_batch x dim
        k_features = k_features.view(-1, rel_cfg.Nr, rel_cfg.d_k).transpose(0, 1)
        q_features = q_features.view(-1, rel_cfg.Nr, rel_cfg.d_k).transpose(0, 1)

        # relational features for final weighting
        # v_features = self.fc_V(a_features).view(a_features.size(0), rel_cfg.Nr, -1).transpose(0, 1)
        v_features = a_features.view(a_features.size(0), rel_cfg.Nr, -1).transpose(0, 1)

        # to estimate appearance weight
        r_features = []

        for interval_idx, interval in enumerate(split):
            sample_k_features = k_features[:, interval[0]:interval[1], :]
            sample_q_features = q_features[:, interval[0]:interval[1], :]

            sample_a_weights = torch.div(torch.bmm(sample_k_features, sample_q_features.transpose(1, 2)), math.sqrt(rel_cfg.d_k))

            sample_g_weights = g_weights[:, rel_pair_counts[interval_idx]:rel_pair_counts[interval_idx + 1]]
            sample_g_weights = sample_g_weights.view(sample_g_weights.size(0), interval[1]-interval[0], interval[1]-interval[0])

            fin_weight = self.softmax(torch.log(sample_g_weights) + sample_a_weights)

            # # mask the weight from objects themselves.
            # fin_weight-=torch.diag_embed(torch.diagonal(fin_weight, dim1=-2, dim2=-1))

            sample_v_features = v_features[:, interval[0]:interval[1], :]

            sample_r_feature = torch.bmm(sample_v_features.transpose(1, 2), fin_weight)

            sample_r_feature = sample_r_feature.view(sample_r_feature.size(0) * sample_r_feature.size(1),
                                                     sample_r_feature.size(2)).transpose(0, 1)

            r_features.append(sample_r_feature)

        r_features = torch.cat(r_features, 0)
        r_features = self.conv_s(r_features.unsqueeze(1)).squeeze(1)

        return r_features