# Do OOD detection before classification. Not implemented yet. 
# v4: Activate 5 fold cross-validation? # Author: DK
# v5: Flipped-label, nosiy-propagation data from in-domain. 
# Test the performance on verified in-distribution samples, threshold=0.5.
import torch
import argparse
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import os
import sys
import os.path as osp
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed, args_print
from utils.measure import calc_acc_bigcn, my_ood_detection, cal_evaluation
import torch.nn.functional as F
from copy import deepcopy
import torch_geometric.transforms as T
#from datasets import TUDataset, twitter_dataset, bigcn_dataset
from datasets.bigcn_ood_dataset import BiGCNDataset, NoisyBiGCNDataset
from gnn.bigcn_gnn import *
from utils.train_test_split import *
from sklearn.model_selection import KFold
import warnings
import wandb
wandb.login()

# Globally suppress UserWarning messages
warnings.filterwarnings("ignore")

in_channels = 768
num_classes = 2
# total id_data: 1137, total ood_data: 400. 
# Previous division has overlaps between n_train_data and n_val_data. 
# n_train_data, n_val_data, n_in_test_data, n_out_test_data = 654, 163, 320, 320
# 5 fold cross-validation is activated, so we don't need to define the dataset size. 
dataset_dir = 'data/'
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

seed = 42
set_seed(seed)

print('Current device: ', device)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training arguments for MNISTSP')
    parser.add_argument('--epoch', default=400, type=int, help='training epoch')
    parser.add_argument('--seed', nargs='?', default='[1,2, 3, 4,5]', help='random seed')
    parser.add_argument('--channels', default=64, type=int, help='width of network')
    # hyper
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--biased_ratio', default=0.05, type=float, help='prior outlier mixed ratio')
    parser.add_argument('--backbone', default='BiGCN', type=str, help='select backbone model')
    parser.add_argument('--prior_ratio', default=0.125, type=float, help='prior outlier ratio')
    parser.add_argument('--neg_ratio', default=1.0, type=float, help='edge negative sampling ratio')
    # basic
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--num_unit', default=2, type=int, help='gnn layers number')
    parser.add_argument('--net_lr', default=1e-4, type=float, help='learning rate for the predictor')
    parser.add_argument('--e_lr', default=1e-3, type=float, help='learning rate for the learnable e')
    parser.add_argument('--dropout', default=0.0, type=float, help='dropout rate for training')
    parser.add_argument('--dropedge', default=0.0, type=float, help='dropedge for regularization')
    parser.add_argument('--bn', action='store_true', help='if using batchnorm')
    parser.add_argument('--graphde_v', action='store_true', help='if enable GraphDE-v')
    parser.add_argument('--graphde_a', action='store_true', help='if enable GraphDE-a')
    # grand
    parser.add_argument('--dropnode', default=0.2, type=float, help='dropnode rate for grand')
    parser.add_argument('--lam', default=1., type=float, help='consistency loss weight for grand')
    parser.add_argument('--tem', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--sample', default=4, type=int, help='sampling time of dropnode')
    parser.add_argument('--order', default=2, type=int, help='propagation step')
    parser.add_argument('--grand', action='store_true', help='if enable grand training')

    args = parser.parse_args()
    args.seed = eval(args.seed)
    if not args.prior_ratio:
        args.prior_ratio = args.biased_ratio
    if args.graphde_a:
        args.pretrain = max(100, args.pretrain)

    return args



# Define the sweep search space: 
sweep_config = {
    "method": "random",
    "name": "graphde_v0_sweep",
    "metric": {"goal": "maximize", "name": "out_test_acc"},
    "parameters":{
        "prior_ratio": {"values": [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]},
        "neg_ratio": {"values": [0.5, 1.0, 1.5]},
        "net_lr": {"values": [1e-2, 1e-3, 1e-4, 1e-5]}
    }

}

# Define the Training Function:
def train_bigcnde(args, id_train_ids, id_test_ids, ood_train_ids, ood_test_ids, fold):
    run = wandb.init()
    sweep_config = wandb.config
    print('In fold ', fold, ': Size of id/ood, train/test are: ', len(id_train_ids), len(id_test_ids), 
          len(ood_train_ids), len(ood_test_ids))
    id_train_dataset = BiGCNDataset(os.getcwd()+'/data/in-domain/', fold_x=id_train_ids)
    id_test_dataset = BiGCNDataset(os.getcwd()+'/data/in-domain/', fold_x=id_test_ids)
    ood_train_dataset = BiGCNDataset(os.getcwd()+'/data/out-of-domain/', fold_x=ood_train_ids)
    # Only flip the labels of ood_train_dataset. 
    ood_test_dataset = BiGCNDataset(os.getcwd()+'/data/out-of-domain/', fold_x=ood_test_ids)
    # When perform the 5 fold cross-validation, we take the best parameter when id_test get best performance. 
    # id_flipped_label_data = BiGCNDataset(os.getcwd()+'/data/in-domain/', fold_x=id_train_ids)

    flip_label_id_train_ids = id_train_ids[:int(len(id_train_ids)*args.biased_ratio/3)]
    drop_edge_id_train_ids = id_train_ids[int(len(id_train_ids)*args.biased_ratio/3):int(len(id_train_ids)*args.biased_ratio*2/3)]
    disorder_edge_id_train_ids = id_train_ids[int(len(id_train_ids)*args.biased_ratio*2/3):int(len(id_train_ids)*args.biased_ratio)]
   
    print('length of three noisy dataset:', len(flip_label_id_train_ids), len(drop_edge_id_train_ids), len(disorder_edge_id_train_ids))
    mixed_flip_label_dataset = NoisyBiGCNDataset(os.getcwd()+'/data/in-domain/', fold_x=flip_label_id_train_ids, label_flip=True)
    mixed_drop_edge_dataset = NoisyBiGCNDataset(os.getcwd()+'/data/in-domain/', fold_x=drop_edge_id_train_ids, edge_drop=True)
    mixed_disorder_edge_dataset = NoisyBiGCNDataset(os.getcwd()+'/data/in-domain/', fold_x=disorder_edge_id_train_ids, edge_disorder=True)
    #train_dataset = ConcatDataset([id_train_dataset, mixed_flip_label_dataset, mixed_drop_edge_dataset, mixed_disorder_edge_dataset, ood_train_dataset])
    train_dataset = ConcatDataset([id_train_dataset, mixed_flip_label_dataset, mixed_drop_edge_dataset])
    train_dataset = id_train_dataset
    # print(mixed_drop_edge_dataset)
    # debug %%%%
    #my_test_loader = DataLoader(mixed_drop_edge_dataset, batch_size=args.batch_size, shuffle=True)
    #for item in my_test_loader:
        #print(item.edge_index)    # %%%%%

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=True)
    ood_test_loader = DataLoader(ood_test_dataset, batch_size=args.batch_size, shuffle=True)

    # log 
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    experiment_name = f'collab.graphde-a_{args.graphde_a}.graphde-v_{args.graphde_v}.backbone_{args.backbone}.' \
                    f'ood-ratio_{args.biased_ratio}.prior-ratio{sweep_config.prior_ratio}.' \
                    f'neg_ratio_{sweep_config.neg_ratio}.' \
                    f'netlr_{args.net_lr}.dropout_{args.dropout}.batch_{args.batch_size}.channels_{args.channels}.' \
                    f'pretrain_{args.pretrain}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    # args_print(args, logger)

    g = BiGCNNet(in_channels=in_channels, num_classes=num_classes, n_train_data=len(train_dataset), args=args, sweep_config=sweep_config).to(device)


    model_optimizer = torch.optim.Adam([
        {'params': g.gnn_model.parameters()},
        {'params': g.structure_model.parameters()},
        {'params': g.out_mlp.parameters()},
        {'params': g.e_logits}
    ], lr=sweep_config.net_lr)

    cnt, last_val_acc, last_train_acc, last_in_test_acc, last_out_test_acc, last_state_dict = 0, 0, 0, 0, 0, None
    best_in_test_acc, best_out_test_acc = 0, 0
    for epoch in tqdm(range(args.epoch)):
        all_loss, n_bw = 0, 0
        g.train()
        for graph in train_loader:
            n_bw += 1
            graph.to(device)
            N = graph.num_graphs
            loss = g.get_graphde_a_loss1(graph) # always graphde_a_loss

            all_loss += loss
        all_loss /= n_bw
        model_optimizer.zero_grad()
        all_loss.backward()
        model_optimizer.step()

        g.eval()
        with torch.no_grad():
            train_acc = calc_acc_bigcn(train_loader, g)
            in_test_acc = calc_acc_bigcn(id_test_loader, g)
            out_test_acc = calc_acc_bigcn(ood_test_loader, g)  
            
            #in_test_eval = cal_evaluation(id_test_loader, g)
            #out_test_eval = cal_evaluation(ood_test_loader, g)

            if in_test_acc > best_in_test_acc:
                best_in_test_acc = in_test_acc
                best_out_test_acc = out_test_acc

            logger.info("Fold {:d} Epoch [{:3d}/{:d}]  all_loss:{:.3f}  "  
                        "Train_ACC:{:.3f}  In-dis_ACC:{:.3f}  OOD_ACC:{:.3f}".format(fold, epoch, args.epoch, all_loss, train_acc, in_test_acc, out_test_acc))
            

            wandb.log({'train_acc': train_acc, 'in_test_acc': in_test_acc, 'out_test_acc': out_test_acc, 'all_loss': all_loss})


        
            # activate early stopping
            if epoch >= args.pretrain:
                if in_test_acc < last_val_acc:
                    cnt += 1
                else:
                    cnt = 0
                    last_val_acc = in_test_acc
                    last_train_acc = train_acc
                    last_in_test_acc = in_test_acc
                    last_out_test_acc = out_test_acc
                    last_state_dict = g.state_dict()
            if cnt >= 400:
                logger.info("Early Stopping...")
                break

        # Save the parameters locally. 
        torch.save(last_state_dict, osp.join(exp_dir, 'predictor-'+str(datetime_now)+'.pt'))
    return best_in_test_acc, best_out_test_acc
        

# Start the sweep: 
sweep_id = wandb.sweep(sweep=sweep_config, project="bigcn_de_sweep_twitter2twitter_covid19")


def main():
    # wandb.init(project="bigcn_de_sweep")
    args = parse_arguments()
    id_ids, ood_ids = np.load('data/in-domain/twi_id_ids.npy'), np.load('data/out-of-domain/twi_ood_ids.npy')
    kfold = KFold(n_splits=5, shuffle=True)
    for fold, ((id_train, id_test), (ood_test, ood_train)) in enumerate(zip(kfold.split(id_ids), kfold.split(ood_ids))):
        best_in_test_acc, best_out_test_acc = train_bigcnde(args, id_ids[id_train], id_ids[id_test], ood_ids[ood_train], ood_ids, fold)
        # sys.exit(0)
        break

if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=40)

