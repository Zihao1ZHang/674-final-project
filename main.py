from utils import *
from modelFromGithub import *
import torch
import torch_cluster
from models.pointnet2_cls_Github import pointnet2_cls_msg

if __name__ == '__main__':
    # train_dataset = load_data("./data/Completion3D/raw/", "train", "Airplane")
    # val_dataset = load_data("./data/Completion3D/raw/", "val", "Airplane")
    device = torch.device('cuda')

    xyz = torch.randn(16, 2048, 3)
    points = torch.randn(16, 2048, 3)
    label = torch.randint(0, 40, size=(16, ))
    ssg_model = pointnet2_cls_msg(6, 40)
    print(ssg_model)
