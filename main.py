import torch
from multiprocessing import Queue
from torch_geometric.data import Data, InMemoryDataset, download_url, extract_zip
from utils.shapenet import ShapenetDataProcess
import argparse
from utils.data_process import get_while_running, kill_data_processes
from torch_geometric.data import DataLoader
from model import SA_net
import kaolin
from utils.emd_func import *

class DataPreprocess(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None, args_input=None):
        self.category = args_input.category
        self.args = args_input
        self.url = 'https://download.cs.stanford.edu/downloads/completion3d/dataset2019.zip'
        super(DataPreprocess, self).__init__(root, transform, pre_transform, pre_filter)
        print(1)
        if args_input.mode == 'train':
            path = self.processed_paths[0]
        elif args_input.mode == 'val':
            path = self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return ['train', 'val']

    # @property
    # def processed_file_names(self):
    #     return ['train.pt']

    @property
    def processed_paths(self):
        return ['./data/Completion3D/processed/train.pt', './data/Completion3D/processed/val.pt']

    def download(self):
        path = download_url(self.url, self.root)
        extract_zip(path, self.root)

    def process(self):
        count = 0
        for mode in ['train', 'val']:
            data_processes = []
            data_queue = Queue(1)
            data_list = []
            for i in range(self.args.nworkers):
                data_processes.append(ShapenetDataProcess(data_queue, self.args, split=mode, repeat=False))
                data_processes[-1].start()

            for targets, clouds_data in get_while_running(data_processes, data_queue, 0.1):
                inp = clouds_data[1][0].squeeze().T
                targets = targets[0]
                data_list += [Data(pos=torch.tensor(inp), y=torch.tensor(targets), category=self.category)]
            data_save = self.collate(data_list)
            torch.save(data_save, self.processed_paths[count])
            count += 1
        kill_data_processes(data_queue, data_processes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.mode = 'train'
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
    train_dataset = DataPreprocess('./data/Completion3D', args_input=args)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    print(1)

