from typing import Callable, Optional
import torch
from torch_geometric.data import InMemoryDataset, download_url

url = None
import os
import os.path as osp
import random
import torch
import numpy as np
from torch_geometric.data import Dataset, download_url, Data
from torch_geometric.loader import DataLoader


def random_pick(list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


class NoisyGACLDataset(Dataset):
    def __init__(self, root, fold_x,  droprate=0, label_flip=False, edge_drop=False, edge_disorder=False, 
                 transform=None, pre_transform=None, pre_filter=None, large_droprate=0.6, datasetname='Twitter'):
        # The droprate is for the GACL data augmentation and contrastive learning. 
        self.fold_x = fold_x
        self.droprate = droprate
        self.raw_path = os.path.join(root, datasetname)

        # params about noise synthesis: 
        self.label_flip = label_flip
        self.edge_drop = edge_drop
        self.large_droprate = large_droprate
        self.edge_disorder = edge_disorder

        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [x + '.npz' for x in self.fold_x]
    
    @property
    def processed_file_names(self):
        return ['noise_gacl_'+x+ '.pt' for x in self.fold_x]
    
    def gacl_process(self, data, idx, id):
        edgeindex = data['edgeindex']
        # Here x0 has grad. Need to remove the grad
        # print('What is the type of data[x]? ', '-', type(data['x']))
        x0 = np.array([i.detach().numpy() for i in data['x']])
        label = int(data['y'])

        init_row = list(edgeindex[0])
        init_col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row = init_row + burow
        col = init_col + bucol

        new_edgeindex = [row, col]

        # Not sure whether the edgeindex starts from 0 or 1.  from 0, and the adj_matrix is not utilised in TGN code.
        # ==================================- dropping + adding + misplacing -===================================#

        choose_list = [1, 2, 3]  # 1-drop 2-add 3-misplace
        probabilities = [0.7, 0.2, 0.1]  # T15: probabilities = [0.5,0.3,0.2]
        choose_num = random_pick(choose_list, probabilities)

        if self.droprate > 0:
            if choose_num == 1:

                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                poslist = sorted(poslist)
                row2 = list(np.array(row)[poslist])
                col2 = list(np.array(col)[poslist])
                new_edgeindex2 = [row2, col2]
                # new_edgeindex = [row2, col2]

            elif choose_num == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                new_edgeindex2 = [row2, col2]


            elif choose_num == 3:
                length = len(init_row)
                mis_index_list = random.sample(range(length), int(length * self.droprate))
                # print('mis_index_list:', mis_index_list)
                Sort_len = len(list(set(sorted(row))))
                if Sort_len > int(length * self.droprate):
                    mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                    # print('mis_valu_list:', mis_value_list)
                    # val_i = 0
                    for i, item in enumerate(init_row):
                        for mis_i, mis_item in enumerate(mis_index_list):
                            if i == mis_item and mis_value_list[mis_i] != item:
                                init_row[i] = mis_value_list[mis_i]
                    row2 = init_row + init_col
                    col2 = init_col + init_row
                    new_edgeindex2 = [row2, col2]


                else:
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
        else:
            new_edgeindex = [row, col]
            new_edgeindex2 = [row, col]

        x = x0
        x_list = list(x)
        if self.droprate > 0:
            if choose_num == 1:
                zero_list = [0] * 768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list
                #print(x_list)
                x2 = np.array(x_list)
                x = x2

        return Data(x0=torch.tensor(x, dtype=torch.float32),
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2),
                    y1=torch.LongTensor([label]),
                    y2=torch.LongTensor([label]))


    # The strategy to add noise into the data:
    # After the process of GACL data augmentation, we add noise based on the edge-dropped and augmented data. 
    def flip_label(self, data):
        # data.y1, data.y2
        data.y1 = torch.tensor([1], dtype=torch.int64) - data.y1
        data.y2 = torch.tensor([1], dtype=torch.int64) - data.y2
        return data
    
    def drop_edge(self, data):
        # data.edge_index, data.edge_index2
        edgeindex = data.edge_index
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        length = len(row)
        poslist = random.sample(range(length), int(length * (1 - self.large_droprate)))
        poslist = sorted(poslist)
        row = list(np.array(row)[poslist])
        col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]
        data.edge_index = torch.LongTensor(new_edgeindex)

        edgeindex = data.edge_index2
        row = list(edgeindex[0])
        col = list(edgeindex[1])
        length = len(row)
        poslist = random.sample(range(length), int(length * (1 - self.large_droprate)))
        poslist = sorted(poslist)
        row = list(np.array(row)[poslist])
        col = list(np.array(col)[poslist])
        new_edgeindex = [row, col]
        data.edge_index2 = torch.LongTensor(new_edgeindex)

        return data
    
    def disorder_edge(self, data):
        # data.edge_index, data.edge_index2
        edgeindex = data.edge_index
        len_edgeindex = len(edgeindex[0])
        disorder_index = np.random.permutation(range(len_edgeindex))
        disorder_index = torch.LongTensor(disorder_index)
        new_edge_index = np.array([edgeindex[0], edgeindex[1][disorder_index]])
        data.edge_index = torch.LongTensor(new_edge_index)

        edgeindex = data.edge_index2
        len_edgeindex = len(edgeindex[0])
        disorder_index = np.random.permutation(range(len_edgeindex))
        disorder_index = torch.LongTensor(disorder_index)
        new_edge_index = np.array([edgeindex[0], edgeindex[1][disorder_index]])
        data.edge_index2 = torch.LongTensor(new_edge_index)
        
        return data
    
    def process(self):
        for idx, id in enumerate(self.fold_x):
            data = np.load(osp.join(self.raw_path, id+'.npz'), allow_pickle=True)
            data = self.gacl_process(data, idx, id)
            # Add noise to the data.
            if self.label_flip:
                data = self.flip_label(data)
            if self.edge_drop:
                data = self.drop_edge(data)
            if self.edge_disorder:
                data = self.disorder_edge(data)
            
            if self.pre_filter is not None and not self.pre_filter(data):
                continue   
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, osp.join(self.processed_dir, f'noise_gacl_{id}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'noise_gacl_{self.fold_x[idx]}.pt'))
        return data


class GACLDataset(Dataset):
    def __init__(self, root, fold_x, droprate=0, transform=None, pre_transform=None, 
                 pre_filter=None, datasetname='Twitter'):
        self.fold_x = fold_x

        self.droprate = droprate
        self.raw_path = os.path.join(root, datasetname)
        super().__init__(root, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self):
        return [x + '.npz' for x in self.fold_x]
    
    @property
    def processed_file_names(self):
        # add 'gacl' prefix to differentiate it from the bigcn one. 
        return ['gacl_data_' +x + '.pt' for x in self.fold_x]
    
    def gacl_process(self, data, idx, id):
        edgeindex = data['edgeindex']
        # Here x0 has grad. Need to remove the grad
        # print('What is the type of data[x]? ', '-', type(data['x']))
        x0 = np.array([i.detach().numpy() for i in data['x']])
        label = int(data['y'])

        init_row = list(edgeindex[0])
        init_col = list(edgeindex[1])
        burow = list(edgeindex[1])
        bucol = list(edgeindex[0])
        row = init_row + burow
        col = init_col + bucol

        new_edgeindex = [row, col]

        # Not sure whether the edgeindex starts from 0 or 1.  from 0, and the adj_matrix is not utilised in TGN code.
        # ==================================- dropping + adding + misplacing -===================================#

        choose_list = [1, 2, 3]  # 1-drop 2-add 3-misplace
        probabilities = [0.7, 0.2, 0.1]  # T15: probabilities = [0.5,0.3,0.2]
        choose_num = random_pick(choose_list, probabilities)

        if self.droprate > 0:
            if choose_num == 1:

                length = len(row)
                poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                poslist = sorted(poslist)
                row2 = list(np.array(row)[poslist])
                col2 = list(np.array(col)[poslist])
                new_edgeindex2 = [row2, col2]
                # new_edgeindex = [row2, col2]

            elif choose_num == 2:
                length = len(list(set(sorted(row))))
                add_row = random.sample(range(length), int(length * self.droprate))
                add_col = random.sample(range(length), int(length * self.droprate))
                row2 = row + add_row + add_col
                col2 = col + add_col + add_row

                new_edgeindex2 = [row2, col2]


            elif choose_num == 3:
                length = len(init_row)
                mis_index_list = random.sample(range(length), int(length * self.droprate))
                # print('mis_index_list:', mis_index_list)
                Sort_len = len(list(set(sorted(row))))
                if Sort_len > int(length * self.droprate):
                    mis_value_list = random.sample(range(Sort_len), int(length * self.droprate))
                    # print('mis_valu_list:', mis_value_list)
                    # val_i = 0
                    for i, item in enumerate(init_row):
                        for mis_i, mis_item in enumerate(mis_index_list):
                            if i == mis_item and mis_value_list[mis_i] != item:
                                init_row[i] = mis_value_list[mis_i]
                    row2 = init_row + init_col
                    col2 = init_col + init_row
                    new_edgeindex2 = [row2, col2]


                else:
                    length = len(row)
                    poslist = random.sample(range(length), int(length * (1 - self.droprate)))
                    poslist = sorted(poslist)
                    row2 = list(np.array(row)[poslist])
                    col2 = list(np.array(col)[poslist])
                    new_edgeindex2 = [row2, col2]
        else:
            new_edgeindex = [row, col]
            new_edgeindex2 = [row, col]

        x = x0
        x_list = list(x)
        if self.droprate > 0:
            if choose_num == 1:
                zero_list = [0] * 768
                x_length = len(x_list)
                r_list = random.sample(range(x_length), int(x_length * self.droprate))
                r_list = sorted(r_list)
                for idex, line in enumerate(x_list):
                    for r in r_list:
                        if idex == r:
                            x_list[idex] = zero_list
                #print(x_list)
                x2 = np.array(x_list)
                x = x2

        return Data(x0=torch.tensor(x, dtype=torch.float32),
                    x=torch.tensor(x, dtype=torch.float32),
                    edge_index=torch.LongTensor(new_edgeindex),
                    edge_index2=torch.LongTensor(new_edgeindex2),
                    y1=torch.LongTensor([label]),
                    y2=torch.LongTensor([label]))


    def process(self):
        for idx, id in enumerate(self.fold_x):
            data = np.load(osp.join(self.raw_path, id + '.npz'), allow_pickle=True)
            data = self.gacl_process(data, idx, id)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            torch.save(data, osp.join(self.processed_dir, f'gacl_data_{id}.pt'))

    def len(self):
        return len(self.processed_file_names)
    
    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, f'gacl_data_{self.fold_x[idx]}.pt'))
        return data

if __name__ == '__main__':
    twitter_id_ids = np.load('./data/in-domain/twi_id_ids.npy')
    twitter_ood_ids = np.load('./data/out-of-domain/twi_ood_ids.npy')

    id_dataset = GACLDataset(root='./data/in-domain', fold_x=twitter_id_ids, droprate=0.4)
    id_dataloader = DataLoader(id_dataset, batch_size=120, shuffle=True)
    
    for Batch in id_dataloader:
        print(Batch)
        print(Batch.batch)