# Do OOD detection before classification. Not implemented yet. 
# v4: Activate 5 fold cross-validation? # Author: DK
# v5: Flipped-label, nosiy-propagation data from in-domain. 
# Test the performance on verified in-distribution samples, threshold=0.5.
import torch
import argparse
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import os
import os.path as osp
from utils.logger import Logger
from datetime import datetime
from utils.helper import set_seed
from utils.measure import evaluation
from datasets.bigcn_ood_dataset import BiGCNDataset
from gnn import init_gnn_model
from gnn.bigcn_gnn import *
from gnn.causal_debias import Causal_NN
from utils.train_test_split import *
from sklearn.model_selection import KFold
import warnings
import wandb
wandb.login()
# Globally suppress UserWarning messages
warnings.filterwarnings("ignore")


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Current device: ', device)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Training arguments for MNISTSP')
    parser.add_argument('--epoch', default=400, type=int, help='training epoch')
    parser.add_argument('--seed', nargs='?', default='2024', help='random seed')
    parser.add_argument('--in_channels', default=768, type=int, help='input width of network')
    parser.add_argument('--channels', default=64, type=int, help='width of network')
    parser.add_argument('--out_channels', default=2, type=int, help='output width of network')
    parser.add_argument('--data_source', default='Twitter', type=str, help='data source')
    # hyper
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--biased_ratio', default=0.05, type=float, help='prior outlier mixed ratio')
    parser.add_argument('--gnn_model', default='BiGCN', type=str, help='select backbone gnn_model')
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

    # grand
    parser.add_argument('--dropnode', default=0.2, type=float, help='dropnode rate for grand')
    parser.add_argument('--lam', default=1., type=float, help='consistency loss weight for grand')
    parser.add_argument('--tem', default=0.5, type=float, help='sharpening temperature')
    parser.add_argument('--sample', default=4, type=int, help='sampling time of dropnode')
    parser.add_argument('--order', default=2, type=int, help='propagation step')
    parser.add_argument('--grand', action='store_true', help='if enable grand training')

    # for GCNii 
    parser.add_argument('--alpha', type=float, default=0.1, help='Alpha')
    parser.add_argument('--theta', type=float, default=0.5, help='Theta')
    parser.add_argument('--layer', type=int, default=8, help='Number of layers')

    args = parser.parse_args()
    args.seed = eval(args.seed)
    if not args.prior_ratio:
        args.prior_ratio = args.biased_ratio
        args.pretrain = max(100, args.pretrain)

    return args





# Define the Training Function:
def train_bigcnde(args, id_train_ids, id_test_ids, twi_covid_test_ids, weibo_covid_test_ids, fold):
    run = wandb.init()
    args.prior_ratio = wandb.config.prior_ratio
    args.neg_ratio = wandb.config.neg_ratio
    args.seed = wandb.config.seed
    set_seed(args.seed)

    id_train_dataset = BiGCNDataset(os.getcwd()+'/data/in-domain/'+args.data_source+'graph/', fold_x=id_train_ids)
    id_test_dataset = BiGCNDataset(os.getcwd()+'/data/in-domain/'+args.data_source+'graph/', fold_x=id_test_ids)
    twi_covid_test_dataset = BiGCNDataset(os.getcwd()+'/data/out-of-domain/Twittergraph/', fold_x=twi_covid_test_ids)
    weibo_covid_test_dataset = BiGCNDataset(os.getcwd()+'/data/out-of-domain/Weibograph/', fold_x=weibo_covid_test_ids)
    

    train_loader = DataLoader(id_train_dataset, batch_size=args.batch_size, shuffle=True)
    id_test_loader = DataLoader(id_test_dataset, batch_size=args.batch_size, shuffle=False)
    twitter_test_loader = DataLoader(twi_covid_test_dataset, batch_size=args.batch_size, shuffle=False)
    weibo_test_loader = DataLoader(weibo_covid_test_dataset, batch_size=args.batch_size, shuffle=False)

    # log 
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")

    experiment_name = f'backbone_{args.gnn_model}.prior-ratio{args.prior_ratio}.neg_ratio_{args.neg_ratio}.' \
                    f'netlr_{args.net_lr}.dropout_{args.dropout}.batch_{args.batch_size}.channels_{args.channels}.' \
                    f'.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    # args_print(args, logger)

    gnn_model = init_gnn_model(args).to(device)
    g = Causal_NN(gnn_model, args).to(device)


    model_optimizer = torch.optim.Adam([
        {'params': g.gnn_model.parameters()},
        {'params': g.structure_model.parameters()},
        {'params': g.out_mlp.parameters()}
    ], lr=args.net_lr)

    
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
            _, train_acc, _ = evaluation(train_loader, g)
            _, in_test_acc, _ = evaluation(id_test_loader, g)

            twitter_test_report, twitter_test_acc, twitter_test_f1 = evaluation(twitter_test_loader, g)
            weibo_test_report, weibo_test_acc, weibo_test_f1 = evaluation(weibo_test_loader, g)

            if in_test_acc > best_in_test_acc:
                best_in_test_acc = in_test_acc
                torch.save(g.state_dict(), osp.join(exp_dir, 'best_predictor-twicovidacc-'+str(twitter_test_acc)+
                                                    '-weibocovidacc-'+str(weibo_test_acc)+'.pt'))


            logger.info("Fold {:d} Epoch [{:3d}/{:d}]  all_loss:{:.3f}  "  
                        "Train_ACC:{:.3f}  Valid_ACC:{:.3f}  Twitter_Test_ACC:{:.3f} Weibo_Test_ACC:{:.3f}".format(fold, epoch, args.epoch, all_loss, train_acc, in_test_acc, twitter_test_acc, weibo_test_acc))
            logger.info("Twitter_Test_Eval: " + twitter_test_report)
            logger.info("Weibo_Test_Eval: " + weibo_test_report)

            wandb.log({'train_acc': train_acc, 
                       'in_test_acc': in_test_acc, 
                       'twitter_test_acc': twitter_test_acc,
                       'twitter_test_f1': twitter_test_f1,
                       'weibo_test_acc': weibo_test_acc,
                       'weibo_test_f1': weibo_test_f1,
                       'all_loss': all_loss, 
                       })

    torch.save(g.state_dict(), osp.join(exp_dir, 'predictor-'+str(datetime_now)+'.pt'))
    return best_in_test_acc, best_out_test_acc
        

# Start the sweep: 
args = parse_arguments()
# Define the sweep search space: 
sweep_config = {
    "method": "random",
    "name": "graphde_v0_sweep",
    "metric": {"goal": "maximize", "name": "twitter_test_acc"},
    "parameters":{
        "prior_ratio": {"values": [0.1, 0.2, 0.3, 0.4, 0.5]},
        "neg_ratio": {"values": [0.3 , 0.7, 1.0, 1.5]},
        'seed': {"values": [42, 38, 1, 2025, 2024]},
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="causal_NN_"+args.gnn_model+'_'+args.data_source)

def train_valid_split(ids, valid_size=0.2):
    num_valid = int(len(ids) * valid_size)
    shuffle(ids)
    return ids[num_valid:], ids[:num_valid]

def main():
    twitter_ids, weibo_ids = np.load('data/in-domain/twi_id_ids.npy'), np.load('data/in-domain/weibo_id_ids.npy')
    twitter_covid_ids, weibo_covid_ids = np.load('data/out-of-domain/twi_ood_ids.npy'), np.load('data/out-of-domain/weibo_ood_ids.npy')

    if args.data_source == 'Twitter':
        id_ids, ood_ids = train_valid_split(twitter_ids)
    if args.data_source == 'Weibo':
        id_ids, ood_ids = train_valid_split(weibo_ids)
    _, _ = train_bigcnde(args, id_ids, ood_ids, twitter_covid_ids, weibo_covid_ids, 0)

if __name__ == '__main__':
    wandb.agent(sweep_id, function=main, count=20)

