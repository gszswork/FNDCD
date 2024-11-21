# Train test split
# Eg. with Twitter data: 
# Training: 70% proportion of Twitter dataset, 
# Validation: 10% proportion of Twitter dataset,
# Test: 20% proportion of Twitter dataset + same amount of Twitter-COVID19 data. 
import os
from random import shuffle
import numpy as np

cwd = os.getcwd()

def my_loadTree(dataname):
    if dataname == 'Twitter':
        id_treePath = os.path.join(cwd,'data/in-domain/Twitter/Twitter_data_all.txt')
        ood_treePath = os.path.join(cwd,'data/out-of-domain/Twitter/Twitter_data_all.txt')
        id_treeDic = {}
        for line in open(id_treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            time_span, Vec = float(line.split('\t')[3]), line.split('\t')[4]
            #max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            if not id_treeDic.__contains__(eid):
                id_treeDic[eid] = {}
            id_treeDic[eid][indexC] = {'parent': indexP, 'time': time_span, 'vec': Vec}

        ood_treeDic = {}
        for line in open(ood_treePath):
            line = line.rstrip()
            eid, indexP, indexC = line.split('\t')[0], line.split('\t')[1], int(line.split('\t')[2])
            #max_degree, maxL, Vec = int(line.split('\t')[3]), int(line.split('\t')[4]), line.split('\t')[5]
            time_span, Vec = float(line.split('\t')[3]), line.split('\t')[4]
            if not ood_treeDic.__contains__(eid):
                ood_treeDic[eid] = {}
            ood_treeDic[eid][indexC] = {'parent': indexP, 'time': time_span, 'vec': Vec}
    
    return id_treeDic, ood_treeDic


def get_id_ood_twitter_ids():
    #print(os.getcwd())
    id_twitter_label_path =  os.getcwd() + '/data/in-domain/Twitter/' + 'Twitter_label_all.txt'

    id_twitter_ids = []
    for line in open(id_twitter_label_path):
        line = line.rstrip()
        eid,label = line.split('\t')[0], line.split('\t')[1]
        id_twitter_ids.append(eid)
        #print(eid,label)

    ood_twitter_label_path = os.getcwd() + '/data/out-of-domain/Twitter/' + 'Twitter_label_all.txt'

    ood_twitter_ids = []
    for line in open(ood_twitter_label_path):
        line = line.rstrip()
        eid,label = line.split('\t')[0], line.split('\t')[1]
        ood_twitter_ids.append(eid)
        #print(eid,label)
    
    #print(len(id_twitter_ids), len(ood_twitter_ids))  # 1154 400
    return id_twitter_ids, ood_twitter_ids

def train_valid_test_split(datasetname):
    # This function can be extended with train, valid, test ratio and ood ratio. 
    id_twitter_ids, ood_twitter_ids = get_id_ood_twitter_ids()
    shuffle(id_twitter_ids)
    shuffle(ood_twitter_ids)
    train_ids = id_twitter_ids[:int(len(id_twitter_ids)*0.7)]
    valid_ids = id_twitter_ids[int(len(id_twitter_ids)*0.7):int(len(id_twitter_ids)*0.8)]
    test_ids = id_twitter_ids[int(len(id_twitter_ids)*0.8):]
    ood_test_ids = ood_twitter_ids[:len(test_ids)]

    print(len(train_ids), len(valid_ids), len(test_ids))
    return train_ids, valid_ids, test_ids, ood_test_ids




if __name__ == '__main__':
    train_ids, valid_ids, test_ids, ood_test_ids = train_valid_test_split('Twitter')
    print(len(train_ids), len(valid_ids), len(test_ids), len(ood_test_ids))