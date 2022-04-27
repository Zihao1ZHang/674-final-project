import os
import numpy as np
from skimage import measure
import open3d as o3d
import h5py
import torch
from torch_geometric.data import Data
from torch.utils.data import DataLoader
import multiprocessing


category_ids = {
    'Airplane': '02691156',
    'Bag': '02773838',
    'Cap': '02954340',
    'Car': '02958343',
    'Chair': '03001627',
    'Earphone': '03261776',
    'Guitar': '03467517',
    'Knife': '03624134',
    'Lamp': '03636649',
    'Laptop': '03642806',
    'Motorbike': '03790512',
    'Mug': '03797390',
    'Pistol': '03948459',
    'Rocket': '04099429',
    'Skateboard': '04225987',
    'Table': '04379243',
}


def read_h5py(file):
    data_list = []

    with h5py.File(file, 'r') as f:
        data_key = list(f.keys())[0]
        data_np = np.array(f[data_key])
        data = torch.tensor(data_np, dtype=torch.float32)
        #print(data.shape)
        data_list.append(data)
    return data_list


def loop_dir(dir):
    data_list = []
    files = os.listdir(dir)
    filenames = [dir + "/" + x for x in files]
    core_num = 6
    pool = multiprocessing.Pool(processes=core_num)
    #result = pool.apply_async(read_h5py, filenames)
    for data in enumerate(pool.imap(read_h5py, filenames)):
        data_list.append(data)
    pool.close()
    pool.join()
    return data_list


def load_data(dir, phase, category):
    if phase != "test":
        assert (category in category_ids)
        category = category_ids[category]
        partial_dir = dir + phase + "/partial/" + category + "/"
        gt_dir = dir + phase + "/gt/" + category + "/"
        partial_list = loop_dir(partial_dir)
        gt_list = loop_dir(gt_dir)

        # partial_tensor = torch.stack(partial_list)
        data_list = []
        for partial, gt in zip(partial_list, gt_list):
            data_list.append([partial[1][0], gt[1][0]])
        out = DataLoader(data_list, batch_size=32, shuffle=True)
        return out
    else:
        assert (category in category_ids)
        category = category_ids[category]
        partial_dir = dir + phase + "/partial/all/"
        partial_list = loop_dir(partial_dir)
        return


def load_h5(filename):
    data = np.loadtxt(filename)
    cloud = o3d.io.read_point_cloud(filename)  # Read the point cloud
    o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud
    return data


def visualize_result(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    return
