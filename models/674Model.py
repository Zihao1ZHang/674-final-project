from utils import *
from modelFromGithub import *
import torch
import torch.nn as nn
import torch_cluster
from models.pointnet2_cls_Github import pointnet2_cls_msg


class 674model(nn.Module):
    def __init__(self, N, N1, M1, N2, M2, M3):
        self.pointnet2_1 = pointnet2_cls_msg()
        self.pointnet2_2 = pointnet2_cls_msg()
        self.pointnet2_3 = pointnet2_cls_msg()
