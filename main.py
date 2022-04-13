from utils import *
import open3d as o3d
import h5py
import numpy as np
import torch

Config = {
    "filedir": "./data/Completion3D/raw/train/partial/02691156/",
    "train": "./tensor/train/02691156/",
    "val": "./tensor/train/02691156/",
    "test": "./tensor/train/02691156/",

}

if __name__ == '__main__':
    filename = "./data/Completion3D/raw/train/partial/02691156/1a04e3eab45ca15dd86060f189eb133.h5"
    filename = "./data/Completion3D/raw/train/partial/02691156/1a9b552befd6306cc8f2d5fe7449af61.h5"

    example_dt = h5py.File(filename, 'r')
    key = list(example_dt.keys())[0]
    data = torch.tensor(example_dt[key], dtype=torch.float32)
    data_np = np.array(example_dt[key])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data)

    o3d.visualization.draw_geometries([pcd])
    print("sada")

