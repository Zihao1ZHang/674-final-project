import os, sys
import argparse
import numpy as np
from utils.data_process import DataProcess, get_while_running, kill_data_processes
from utils.data_utils import load_h5, load_csv, augment_cloud, pad_cloudN
from utils.vis import plot_pcds
from main import DataPreprocess
from torch_geometric.data import DataLoader
# from model import SA_net
# from model_modified import SA_net
from model_cos import SA_net
import torch
import kaolin
from utils.emd_func import *


def cal_loss(data_loader, test_model, dataset):
    total_loss = 0
    for data_test in data_loader:
        with torch.no_grad():
            data_test = data_test.to(device)
            decoded = test_model(data_test, data_test.batch).type(torch.float64)
            loss = torch.sum(emd(decoded.reshape(-1, 2048, 3), data_test.y.reshape(-1, 2048, 3), 0.002, 10000)[0]) + 10 * torch.sum(kaolin.metrics.pointcloud.chamfer_distance(decoded.reshape(-1, 2048, 3), data_test.y.reshape(-1, 2048, 3)))
            total_loss += loss.item() * data_test.num_graphs
    return total_loss / len(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.mode = 'val'
    args.category = 'plane'
    args.dataset = 'shapenet'
    args.DATA_PATH = "./data/Completion3D/raw"
    args.nworkers = 1
    args.batch_size = 1
    args.pc_augm_scale = 0
    args.pc_augm_rot = 0
    args.pc_augm_mirror_prob = 0
    args.pc_augm_jitter = 0
    args.inpts = 2048
    test_dataset = DataPreprocess('./data/Completion3D', args_input=args)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SA_net().to(device)
    model.load_state_dict(torch.load('./trained/SA_net_Ch_Airplane_674_cos_100.pt', map_location=device))
    model.eval()
    gen_data = {
        'pred': np.empty([len(test_dataset), 2048, 3]),
        'true': np.empty([len(test_dataset), 2048, 3]),
        'original': np.empty([len(test_dataset), 2048, 3]),
    }
    batch_size = 1
    for i, data in enumerate(test_loader):
        # if data.batch[-1] != 15:
        #     continue
        data = data.to(device)
        with torch.no_grad():
            begin = i * batch_size
            end = begin + len(data.category)
            gen_data['pred'][begin:end] = model(data, data.batch)[0].cpu().numpy().reshape((-1, 2048, 3))
            gen_data['true'][begin:end] = data.y.cpu().numpy().reshape((-1, 2048, 3))
            gen_data['original'][begin:end] = data.pos.cpu().numpy().reshape((-1, 2048, 3))
    emd = emdModule()
    print(cal_loss(data_loader=test_loader, test_model=model, dataset=test_dataset))
    for i in range(100):
        path = "./result/model_cos/model_cos_" + str(i) + ".png"
        plot_pcds(path, [gen_data['pred'][i].squeeze(), gen_data['original'][i].squeeze(), gen_data['true'][i].squeeze()], ['pred', 'partial', 'gt'], use_color=[0, 0, 0], color=[None, None, None])
