import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN, LeakyReLU, GroupNorm
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
import numpy as np
from torch.nn import Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def MLP(channels):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU())
        for i in range(1, len(channels))
    ])



#
# def MLP(channels, enable_group_norm=True):
#     if enable_group_norm:
#         num_groups = [0]
#         for i in range(1, len(channels)):
#             if channels[i] >= 32:
#                 num_groups.append(channels[i]//32)
#             else:
#                 num_groups.append(1)
#         return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2), GroupNorm(num_groups[i], channels[i]))
#                      for i in range(1, len(channels))])
#     else:
#         return Seq(*[Seq(Lin(channels[i - 1], channels[i]), LeakyReLU(negative_slope=0.2))
#                      for i in range(1, len(channels))])


class PointNet(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(PointNet, self).__init__()


class SelfAttention(torch.nn.Module):
    def __init__(self, mlp_h, mlp_g, mlp_l, mlp_map):
        super(SelfAttention, self).__init__()
        self.mlp_h = MLP(mlp_h)
        self.mlp_g = MLP(mlp_g)
        self.mlp_l = MLP(mlp_l)
        self.mlp_map = MLP(mlp_map)


    def forward(self, x):
        h = self.mlp_h(x).transpose(-1, -2)
        l = self.mlp_l(x)
        mm = torch.matmul(l, h)
        attn_weights = F.softmax(mm, dim=-1)
        g = self.mlp_g(x)
        atten_appllied = torch.bmm(attn_weights, g)
        return self.mlp_map(x + atten_appllied)


class FoldingBlock(torch.nn.Module):
    def __init__(self, num_input_features, num_output_features, mlp_up, mlp_down, mlp_h, mlp_g, mlp_l, mlp_map):
        super(FoldingBlock, self).__init__()
        self.input = num_input_features
        self.output = num_output_features
        self.ratio = self.output/self.input
        self.mlp_up = MLP(mlp_up)
        self.mlp_down = MLP(mlp_down)
        self.self_attention = SelfAttention(mlp_h, mlp_g, mlp_l, mlp_map)

    def Up_module(self, points, grid):
        dup = points.repeat(1, int(self.ratio), 1).contiguous()
        grid = grid.squeeze()
        if len(grid.size()) == 2:
            grid = grid.unsqueeze(0)
        point_1 = torch.cat((dup, grid), -1)
        point_2 = self.mlp_up(point_1)
        point_3 = torch.cat((dup, point_2), -1)
        point_4 = self.self_attention(point_3)
        return point_4

    def Down_module(self, points, up_points):
        # point_1 = torch.reshape(up_points, (up_points.shape[0], self.input, int(self.ratio)*up_points.shape[2])).contiguous()
        # point_2 = point_1-points
        # point_3 = self.mlp_down(point_2)
        # return point_3
        up_reshape = up_points.view(-1, self.input, int(self.ratio) * up_points.size(2))
        mlp_up_points = self.mlp_down(up_reshape)
        return points - mlp_up_points


    def forward(self, points, grid):
        up = self.Up_module(points, grid=grid)
        down = self.Down_module(points, up)
        up1 = self.Up_module(down, grid=grid)
        return up1+up


# class Self_Attn(torch.nn.Module):
#     def __init__(self, in_dim, activation):
#         super(Self_Attn, self).__init__()
#         self.chanel_in = in_dim
#         self.activation = activation
#
#         self.query_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.key_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
#         self.value_conv = torch.nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
#         self.gamma = torch.nn.Parameter(torch.zeros(1))
#
#         self.softmax = torch.nn.Softmax(dim=-1)
#
#     def forward(self, x):
#         m_batchsize, C, width, height = x.size()
#         proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
#         proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
#         energy = torch.bmm(proj_query, proj_key)  # transpose check
#         attention = self.softmax(energy)  # BX (N) X (N)
#         proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N
#
#         out = torch.bmm(proj_value, attention.permute(0, 2, 1))
#         out = out.view(m_batchsize, C, width, height)
#
#         out = self.gamma * out + x
#
#         return out, attention


class SkipAttention(torch.nn.Module):

    def __init__(self, mlp_h, mlp_l, mlp_g, mlp_f=None):
        super(SkipAttention, self).__init__()
        self.mlp_h = mlp_h
        self.mlp_l = mlp_l
        self.mlp_g = mlp_g
        self.mlp_f = mlp_f

    def forward(self, p, r, batch, method='learnable'):
        r = torch.reshape(r, (batch[-1] + 1, int(r.shape[0]/(batch[-1]+1)), r.shape[1])).unsqueeze(-3)

        h = self.mlp_h(p)
        l = self.mlp_l(r)

        if method == 'learnable':
            h = h.expand(-1, -1, r.shape[2], -1).unsqueeze(-2)
            l = l.expand(-1, h.shape[1], -1, -1).unsqueeze(-1)
            mm = torch.matmul(h, l).squeeze()
            attn_weights = F.softmax(mm, dim=-1)
        else:
            h = h.expand(-1, -1, r.shape[2], -1)
            l = l.expand(-1, h.shape[1], -1, -1)
            cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
            attn_weights = cos(h, l)
        g = self.mlp_g(r)
        g = g.squeeze()
        if len(attn_weights.size()) == 2:
            attn_weights = attn_weights.unsqueeze(0)
        if len(g.size()) == 2:
            g = g.unsqueeze(0)
        atten_appllied = torch.bmm(attn_weights, g)
        if self.mlp_f is not None:
            return self.mlp_f(p.squeeze() + atten_appllied)
        else:
            return p.squeeze() + atten_appllied

class SA_net(torch.nn.Module):
    meshgrid = [[-0.3, 0.3, 46], [-0.3, 0.3, 46]]
    x = np.linspace(*meshgrid[0])
    y = np.linspace(*meshgrid[1])
    points = torch.tensor(np.meshgrid(x, y), dtype=torch.float32)
    def __init__(self):
        super(SA_net, self).__init__()
        self.point_net0 = pointnet_2(0.5, 0.2, MLP([3 + 3, 32, 32, 64]))
        self.point_net1 = pointnet_2(0.5, 0.2, MLP([64 + 3, 64, 64, 128]))
        self.point_net2 = pointnet_2(0.5, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.point_net3 = pointnet_2_group(MLP([256 + 3, 256, 512, 512]))

        self.skip_attention1 = SkipAttention(MLP([514, 128]), MLP([256, 128]), MLP([256, 512 + 2]), MLP([514, 512]))
        self.skip_attention2 = SkipAttention(MLP([256, 64]), MLP([128, 64]), MLP([128, 256]), MLP([256, 256]))
        self.skip_attention3 = SkipAttention(MLP([128, 32]), MLP([64, 32]), MLP([64, 128]), MLP([128, 128]))

        self.folding_block1 = FoldingBlock(64, 256, [514, 512], [1024, 512], [1024, 256], [1024, 1024], [1024, 256], [1024, 512, 256])
        self.folding_block2 = FoldingBlock(256, 512, [258, 256], [256, 256], [512, 64], [512, 512], [512, 64], [512, 256, 128])
        self.folding_block3 = FoldingBlock(512, 2048, [130, 128], [512, 256, 128], [256, 64], [256, 256], [256, 64], [256, 128])

        self.ln1 = Linear(128, 64)
        self.relu = torch.nn.ReLU()
        self.ln2 = Linear(64, 3)


    def get_2dplane(self, m, n, batch):
        indeces_x = np.round(np.linspace(0, 45, m)).astype(int)
        indeces_y = np.round(np.linspace(0, 45, n)).astype(int)
        x, y = np.meshgrid(indeces_x, indeces_y)
        p = SA_net.points[:, x.ravel(), y.ravel()].T.contiguous()
        p = p[None, :, None, :].repeat(batch[-1] + 1, 1, 1, 1)
        return p.to(device)
        # self.plane2D = torch.tensor(np.random.normal(0, 1, (n, 2))[np.newaxis, :, np.newaxis, :], dtype=torch.float32)\
        #                     .repeat(16, 1, 1, 1).to(device)
        # return self.plane2D

    def encoder(self, x, batch):
        level0 = self.point_net0(x.pos, x.pos, batch, batch[-1] + 1)
        level1 = self.point_net1(level0[0], level0[1], level0[2], batch[-1] + 1)
        level2 = self.point_net2(level1[0], level1[1], level1[2], batch[-1] + 1)
        level3 = self.point_net3(level2[0], level2[1], level2[2])
        return level0, level1, level2, level3

    def decoder(self, level0, level1, level2, level3, batch):
        x = level3[0][:, None, None, :].repeat(1, 64, 1, 1)
        grid = self.get_2dplane(8, 8, batch)
        #x = torch.cat((x, p.view(1, p.size(0), 1, p.size(-1)).repeat(x.size(0), 1, 1, 1)), -1)
        x = torch.cat((x, grid), -1)

        x = self.skip_attention1(x, level2[0], batch)
        grid = self.get_2dplane(16, 16, batch)
        x = self.folding_block1(x, grid)
        x = x.unsqueeze(-2)

        x = self.skip_attention2(x, level1[0], batch, 'cosine')
        grid = self.get_2dplane(32, 16, batch)
        x = self.folding_block2(x, grid)
        x = x.unsqueeze(-2)

        x = self.skip_attention3(x, level0[0], batch, 'cosine')
        grid = self.get_2dplane(64, 32, batch)
        x = self.folding_block3(x, grid)
        x = self.ln1(x)
        x = self.relu(x)
        x = self.ln2(x)
        return x

    def forward(self, x, batch):
        level0, level1, level2, level3 = self.encoder(x, batch)
        out = self.decoder(level0, level1, level2, level3, batch)
        return out
