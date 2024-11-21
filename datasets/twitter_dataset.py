import os.path as osp
import pickle as pkl
import os
import torch
import random
import numpy as np
import shutil
from typing import Callable, List, Optional
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip, DataLoader
from torch_geometric.io import read_tu_data

cwd = os.getcwd()

class TwitterDataset(InMemoryDataset):
    def __init__(self, root, fold_x, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        # transform: do before accessing, usually for data augmentation.
        # pre_transform: do before saving the data to the disk, usually for heavy precomputation.
        # pre_filter: filter out data samples before saving.
        self.fold_x = fold_x
        self.data_path = root
        self.tddroprate=0.1
        self.budroprate=0.1

    @property
    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data_1.pt', 'data_2.pt', ...]

    @property
    def length(self):
        return len(self.fold_x)

    #def download(self):
    #    print('Please download the dataset to local path: data/TWITTER/raw mannually. ')

    def loadTree(self):
        treePath = os.path.join(cwd, 'data/TWITTER/raw/data.TD_RVNN.vol_5000.txt')
        print('reading Twitter dataset')
        treeDict = {}
        for line in open(treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not treeDict.__contains__(eid):
                treeDict[eid] = {}
            treeDict[eid][indexC] = {'parent': indexP, 'max_degree': max_degree, 'maxL': maxL, 'vec': Vec}
        return treeDict


    #def process(self):
        # Data is already processed during 'getTwitterGraph.py'
    #    pass

    def get(self, idx):
        id =self.fold_x[idx]
        data=np.load(os.path.join(self.data_path, id + ".npz"), allow_pickle=True)
        edgeindex = data['edgeindex']
        if self.tddroprate > 0:
            row = list(edgeindex[0])
            col = list(edgeindex[1])
            length = len(row)
            poslist = random.sample(range(length), int(length * (1 - self.tddroprate)))
            poslist = sorted(poslist)
            row = list(np.array(row)[poslist])
            col = list(np.array(col)[poslist])
            new_edgeindex = [row, col]
        else:
            new_edgeindex = edgeindex

        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        if self.budroprate > 0:
            length = len(burow)
            poslist = random.sample(range(length), int(length * (1 - self.budroprate)))
            poslist = sorted(poslist)
            row = list(np.array(burow)[poslist])
            col = list(np.array(bucol)[poslist])
            bunew_edgeindex = [row, col]
        else:
            bunew_edgeindex = [burow,bucol]
        return Data(x=torch.tensor(data['x'],dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),BU_edge_index=torch.LongTensor(bunew_edgeindex),
             y=torch.LongTensor([int(data['y'])]), root=torch.LongTensor(data['root']),
             rootindex=torch.LongTensor([int(data['rootindex'])]))

if __name__ == '__main__':
    fold_x = np.load('../data/TWITTER/Twitter15/fold_x.npy')

    test_dataset = TwitterDataset(root='../data/TWITTER/Twitter15/bigcn_processed/', fold_x=fold_x)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False)

    for graph in test_loader:
        print(graph.num_graphs, graph.x, graph.y)