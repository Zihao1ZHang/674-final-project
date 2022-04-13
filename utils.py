import os
import numpy as np
from skimage import measure
import open3d as o3d
import h5py


def loop_dir(config):

    return


def load_data(filename):
    data = np.loadtxt(filename)
    cloud = o3d.io.read_point_cloud(filename)  # Read the point cloud
    o3d.visualization.draw_geometries([cloud])  # Visualize the point cloud
    return data


def visualize_result(points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.visualization.draw_geometries([pcd])
    return
