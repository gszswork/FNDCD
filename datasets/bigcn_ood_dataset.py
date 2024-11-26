from typing import Callable, Optional
import torch
from torch_geometric.data import InMemoryDataset, download_url
import os
import os.path as osp
import random
import torch
import numpy as np
from torch_geometric.data import Dataset, download_url, Data

class BiGCNDataset(Dataset):
    def __init__(self, root, fold_x, droprate=0, transform=None, pre_transform=None, pre_filter=None, td_droprate=0.2, bu_droprate=0.2):
        # TODO: Fit the root into the path
        self.fold_x = fold_x
        self.tddroprate = droprate
        self.budroprate = droprate
        self.raw_path = root
        self.tddroprate = td_droprate
        self.budroprate = bu_droprate
        #self.processed_dir = os.path.abspath(os.path.join(root, '..', 'data', 'TWITTER', 'Twitter15', 'processed'))
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self):
        return [x + '.npz' for x in self.fold_x]

    @property
    def processed_file_names(self):
        return ['data_' + x + '.pt' for x in self.fold_x]


    def bigcn_process(self, data, idx, id):
        edgeindex = data['edgeindex']
        seqlen = torch.LongTensor([(data['seqlen'])]).squeeze()

        x_features=torch.tensor([item.detach().numpy() for item in list(data['x'])],dtype=torch.float32)
        #x_features = torch.stack([torch.from_numpy(item.detach().cpu().numpy()) for item in list(data['x'])]).type(torch.float32)

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
        return Data(x=x_features,
                    edge_index=torch.LongTensor(new_edgeindex),
                    BU_edge_index=torch.LongTensor(bunew_edgeindex),
                    y=torch.LongTensor([int(data['y'])]), 
                    root=torch.from_numpy(data['root']),
                    rootindex=torch.LongTensor([int(data['rootindex'])]), 
                    seqlen=seqlen, 
                    idx=idx,
                    id=id)

    def process(self):
        for idx, id in enumerate(self.fold_x):
            # Read data from `raw_path`.
            data = np.load(osp.join(self.raw_path, id+'.npz'), allow_pickle=True)
            data = self.bigcn_process(data, idx, id)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, osp.join(self.processed_dir, f'data_{id}.pt'))

    def len(self):
        return len(self.processed_file_names)

    def get(self, id):
        data = torch.load(osp.join(self.processed_dir, f'data_{self.fold_x[id]}.pt'))
        return data
