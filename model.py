import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
from torch.nn import Linear

class pointnet_2(torch.nn.Module):
    def __init__(self, ratio, rad, nn):
        super(pointnet_2, self).__init__()
        self.ratio = ratio
        self.rad = rad
        self.conv = PointConv(nn)
    def forward(self, xyz, point, batch, num_samples):
        idx = fps(point, batch, ratio=self.ratio)
        r, c = radius(point, point[idx], self.rad, batch, batch[idx], max_num_neighbors=num_samples)
        edge_idx = torch.stack([c, r], dim=0)
        xyz = self.conv(xyz, (point, point[idx]), edge_idx)
        point = point[idx]
        batch = batch[idx]
        return xyz, point, batch


class pointnet_2_group(torch.nn.Module):
    def __init__(self, nn):
        super(pointnet_2_group, self).__init__()
        self.nn = nn

    def forward(self, xyz, point, batch):
        xyz = self.nn(torch.cat([xyz, point], dim=1))
        xyz = global_max_pool(xyz, batch)
        point = point.new_zeros((xyz.size(0), 3))
        batch = torch.arange(xyz.size(0), device=batch.device)
        return xyz, point, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        #               ,BN(channels[i]))
        for i in range(1, len(channels))
    ])

class SA_net(torch.nn.Module):
    def __init__(self, num_input, num_output):
        super(SA_net, self).__init__()
        self.fc = Linear(num_input, num_output)
        self.point_net1 = pointnet_2(0.25, 0.2, MLP([3 + 3, 64, 64, 128]))
        self.point_net2 = pointnet_2(0.5, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.point_net3 = pointnet_2_group(MLP([256 + 3, 256, 512, 512]))

    def encoder(self, x, batch):
        xyz, point, batch = self.point_net1(x, x, batch, 32)
        xyz, point, batch = self.point_net2(xyz, point, batch, 32)
        xyz, point, batch = self.point_net3(xyz, point, batch)

        return xyz, point, batch

    def forward(self, x, batch):
        out = self.encoder(x, batch)

        return out
