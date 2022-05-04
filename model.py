import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU, GroupNorm
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np


class Self_Attn(torch.nn.Module):
    def __init__(self, in_dim):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = torch.nn.Parameter(torch.zeros(1))

        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):

        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x

        return out, attention

def MLP(channels, enable_group_norm=True):
    if enable_group_norm:
        num_groups = [0]
        for i in range(1, len(channels)):
            if channels[i] >= 32:
                num_groups.append(channels[i]//32)
            else:
                num_groups.append(1)
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
                     for i in range(1, len(channels))])
    else:
        return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
                     for i in range(1, len(channels))])

class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()

class FoldingBlock(torch.nn.Module):
    def __init__(self, num_input_features):
        super(FoldingBlock, self).__init__()
        self.input = num_input_features
        self.mlp1 = MLP([2*num_input_features+2,2*num_input_features])
        self.mlp2 = MLP([num_input_features,2*num_input_features])
        self.atten = Self_Attn(2*num_input_features)
    def Up_module(self,points,grid):
        ratio = 2
        dup = points.repeat(int(ratio),1).contiguous()
        point_1 = torch.cat((dup,grid),-1)
        point_2 = self.mlp1(point_1)
        point_3 = torch.cat((dup,point_2),-1)
        point_4 , =self.atten(point_3)

        return point_4

    def Down_module(self, points,up_points):
        x,y = up_points.shape
        point_1 = torch.reshape(up_points, (int(x/2), y*2)).contiguous()
        point_2 = point_1-points
        point_3 = self.mlp2(point_2)
        return point_3

    def forward(self, points, grid):
        up = self.Up_module(points,grid= grid)
        down = self.Down_module(points,up)
        up1 = self.Up_module(down,grid= grid)
        # print("1")
        return up1+up

