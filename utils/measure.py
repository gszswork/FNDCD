import numpy as np
import sklearn.metrics as sk
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from copy import deepcopy
import torch.nn.functional as F
from ogb.graphproppred import Evaluator
import subprocess
import pickle
import os

recall_level_default = 0.95
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def evaluation(loader, predictor):
    # return the evaluation report, acc and f1 score.  
    true, pred = [], []
    for graph in loader:
        graph.to(device)
        pred_out = predictor(graph)
        pred.append(pred_out.argmax(-1).view(-1).detach().cpu().numpy())
        true.append(graph.y.view(-1).detach().cpu().numpy())
    evaluation_res = sk.classification_report(np.concatenate(true), np.concatenate(pred), digits=4)
    acc = sk.accuracy_score(np.concatenate(true), np.concatenate(pred))
    f1 = sk.f1_score(np.concatenate(true), np.concatenate(pred), average='macro')
    return evaluation_res, acc, f1


def calc_evaluation_gacl(loader, predictor):
    true, pred = [], []
    for graph in loader:
        graph.to(device)
        pred1, pred2, y1, y2, _ = predictor(graph)
        pred.append(pred1.argmax(-1).view(-1).detach().cpu().numpy())
        pred.append(pred2.argmax(-1).view(-1).detach().cpu().numpy())
        true.append(y1.view(-1).detach().cpu().numpy())
        true.append(y2.view(-1).detach().cpu().numpy())
    evaluation_res = sk.classification_report(np.concatenate(true), np.concatenate(pred), digits=3)
    return evaluation_res



def ood_detection(iid_loader, ood_loader, predictor, exp_dir, seed):
    pos_softmax_scores, pos_prob_scores, neg_softmax_scores, neg_prob_scores = [], [], [], []
    pos_labels, pos_ids, neg_labels, neg_ids = [], [], [], []
    for graph in iid_loader:
        graph.to(device)
        pos_pred_out = predictor(x=graph.x, edge_index=graph.edge_index,
                            edge_attr=graph.edge_attr, batch=graph.batch)
        pos_softmax_score, _ = torch.max(F.softmax(pos_pred_out, dim=1), dim=1)
        pos_prob_score = predictor.infer_e_gx(x=graph.x, edge_index=graph.edge_index,
                                            batch=graph.batch)

        pos_softmax_scores += deepcopy(pos_softmax_score.detach().cpu().numpy().tolist())
        pos_prob_scores += deepcopy(pos_prob_score.detach().cpu().numpy().tolist())
        pos_labels += deepcopy(graph.y.detach().cpu().numpy().tolist())
        pos_ids += deepcopy(graph.idx.detach().cpu().numpy().tolist())
    for graph in ood_loader:
        graph.to(device)
        neg_pred_out = predictor(x=graph.x, edge_index=graph.edge_index,
                                edge_attr=graph.edge_attr, batch=graph.batch)
        neg_softmax_score, _ = torch.max(F.softmax(neg_pred_out, dim=1), dim=1)
        neg_prob_score = predictor.infer_e_gx(x=graph.x, edge_index=graph.edge_index,
                                            batch=graph.batch)

        neg_softmax_scores += deepcopy(neg_softmax_score.detach().cpu().numpy().tolist())
        neg_prob_scores += deepcopy(neg_prob_score.detach().cpu().numpy().tolist())
        neg_labels += deepcopy(graph.y.detach().cpu().numpy().tolist())
        neg_ids += deepcopy(graph.idx.detach().cpu().numpy().tolist())

    pkl_file = open(os.path.join(exp_dir, f'detection_res-seed{seed}.pkl'), 'wb')
    pickle.dump(pos_ids, pkl_file); pickle.dump(pos_labels, pkl_file); pickle.dump(pos_prob_scores, pkl_file)
    pickle.dump(neg_ids, pkl_file); pickle.dump(neg_labels, pkl_file); pickle.dump(neg_prob_scores, pkl_file)
    pkl_file.close()

    base_auroc, base_aupr, base_fpr = get_measures(pos_softmax_scores, neg_softmax_scores)
    vi_auroc, vi_aupr, vi_fpr = get_measures(pos_prob_scores, neg_prob_scores)
    # plot_distribution_comparison(pos_softmax_scores, neg_softmax_scores, pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    plot_distribution(pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    return base_auroc, base_aupr, base_fpr, vi_auroc, vi_aupr, vi_fpr

def get_pred_distribution(iid_loader, ood_loader, predictor, threshold=0.5):
    # The implementation to get in-distribution and out-of-distribution sample ids of the two domains data. 

    pos_softmax_scores, pos_prob_scores, neg_softmax_scores, neg_prob_scores = [], [], [], []
    pos_labels, pos_ids, neg_labels, neg_ids = [], [], [], []
    for graph in iid_loader:
        graph.to(device)
        pos_pred_out = predictor(graph)
        #print(pos_pred_out)
        pos_softmax_score, _ = torch.max(F.softmax(pos_pred_out, dim=1), dim=1)
        pos_prob_score = predictor.infer_e_gx(x=graph.x, edge_index=graph.edge_index,
                                            batch=graph.batch)

        pos_softmax_scores += deepcopy(pos_softmax_score.detach().cpu().numpy().tolist())
        pos_prob_scores += deepcopy(pos_prob_score.detach().cpu().numpy().tolist())
        pos_labels += deepcopy(graph.y.detach().cpu().numpy().tolist())
        # In my dataset, pos_ids are originally list.
        pos_ids += deepcopy(graph.id)
        
    for graph in ood_loader:
        graph.to(device)
        neg_pred_out = predictor(graph)
        neg_softmax_score, _ = torch.max(F.softmax(neg_pred_out, dim=1), dim=1)
        neg_prob_score = predictor.infer_e_gx(x=graph.x, edge_index=graph.edge_index,
                                            batch=graph.batch)

        neg_softmax_scores += deepcopy(neg_softmax_score.detach().cpu().numpy().tolist())
        neg_prob_scores += deepcopy(neg_prob_score.detach().cpu().numpy().tolist())
        neg_labels += deepcopy(graph.y.detach().cpu().numpy().tolist())
        neg_ids += deepcopy(graph.id)
    pred_id_in_domain_ids = [pos_ids[i] for i in range(len(pos_ids)) if pos_prob_scores[i] >= threshold]
    pred_ood_in_domain_ids = [pos_ids[i] for i in range(len(pos_ids)) if pos_prob_scores[i] < threshold]
    pred_id_out_domain_ids = [neg_ids[i] for i in range(len(neg_ids)) if neg_prob_scores[i] >= threshold]
    pred_ood_out_domain_ids = [neg_ids[i] for i in range(len(neg_ids)) if neg_prob_scores[i] < threshold]
    return pred_id_in_domain_ids, pred_ood_in_domain_ids, pred_id_out_domain_ids, pred_ood_out_domain_ids


def my_ood_detection(iid_loader, ood_loader, predictor, exp_dir, seed):
    # My fix of ood_detection function, because the predictor()'s parameters are different. 
    # predict.infer_e_gx（)'s params may have issues, because the graph.edge_index is the top-down edge_index.
    pos_softmax_scores, pos_prob_scores, neg_softmax_scores, neg_prob_scores = [], [], [], []
    pos_labels, pos_ids, neg_labels, neg_ids = [], [], [], []
    for graph in iid_loader:
        graph.to(device)
        pos_pred_out = predictor(graph)
        #print(pos_pred_out)
        pos_softmax_score, _ = torch.max(F.softmax(pos_pred_out, dim=1), dim=1)
        pos_prob_score = predictor.infer_e_gx(x=graph.x, edge_index=graph.edge_index,
                                            batch=graph.batch)

        pos_softmax_scores += deepcopy(pos_softmax_score.detach().cpu().numpy().tolist())
        pos_prob_scores += deepcopy(pos_prob_score.detach().cpu().numpy().tolist())
        pos_labels += deepcopy(graph.y.detach().cpu().numpy().tolist())
        # In my dataset, pos_ids are originally list.
        pos_ids += deepcopy(graph.idx)
        
    for graph in ood_loader:
        graph.to(device)
        neg_pred_out = predictor(graph)
        neg_softmax_score, _ = torch.max(F.softmax(neg_pred_out, dim=1), dim=1)
        neg_prob_score = predictor.infer_e_gx(x=graph.x, edge_index=graph.edge_index,
                                            batch=graph.batch)

        neg_softmax_scores += deepcopy(neg_softmax_score.detach().cpu().numpy().tolist())
        neg_prob_scores += deepcopy(neg_prob_score.detach().cpu().numpy().tolist())
        neg_labels += deepcopy(graph.y.detach().cpu().numpy().tolist())
        neg_ids += deepcopy(graph.idx)

    pkl_file = open(os.path.join(exp_dir, f'detection_res-seed{seed}.pkl'), 'wb')
    pickle.dump(pos_ids, pkl_file); pickle.dump(pos_labels, pkl_file); pickle.dump(pos_prob_scores, pkl_file)
    pickle.dump(neg_ids, pkl_file); pickle.dump(neg_labels, pkl_file); pickle.dump(neg_prob_scores, pkl_file)
    pkl_file.close()

    base_auroc, base_aupr, base_fpr = get_measures(pos_softmax_scores, neg_softmax_scores)

    # Generate the OOD prediction label. 
    pos_id_label = (pos_prob_score >= 0.5).int()
    pos_id_proportion = torch.sum(pos_id_label).float() / len(pos_id_label)
    neg_id_label = (neg_prob_score >= 0.5).int()
    neg_id_proportion = torch.sum(neg_id_label).float() / len(neg_id_label)

    vi_auroc, vi_aupr, vi_fpr = get_measures(pos_prob_scores, neg_prob_scores)
    # plot_distribution_comparison(pos_softmax_scores, neg_softmax_scores, pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    plot_distribution(pos_prob_scores, neg_prob_scores, os.path.join(exp_dir,f"distribution-seed{seed}.pdf"))
    return base_auroc, base_aupr, base_fpr, vi_auroc, vi_aupr, vi_fpr, pos_id_proportion, neg_id_proportion


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=recall_level_default, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=recall_level_default):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = sk.roc_auc_score(labels, examples)
    aupr = sk.average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr

def plot_distribution_comparison(base_pos, base_neg, vi_pos, vi_neg, filename):
    """plot vi/base score distribution comparison figure"""
    sns.set(style='white')
    fig, ax = plt.subplots(1, 2, figsize=(9, 3))

    # plot vi score distribution
    sns.distplot(vi_pos, hist=False,ax=ax[0], kde_kws={'fill': True}, color='blue', label='in-distribution')
    sns.distplot(vi_neg, hist=False, ax=ax[0], kde_kws={'fill': True}, color='red', label='out-of-distribution')
    ax[0].spines['bottom'].set_linewidth(0.5)
    ax[0].spines['left'].set_linewidth(0.5)
    ax[0].spines['top'].set_linewidth(0.5)
    ax[0].spines['right'].set_linewidth(0.5)
    ax[0].set_xlabel('(a) VI Probability Score', size=12)
    ax[0].set_ylabel('Frequency', size=12)

    # plot base score distribution
    sns.distplot(base_pos, hist=False, ax=ax[1], kde_kws={'fill': True}, color='blue', label='in-distribution')
    sns.distplot(base_neg, hist=False, ax=ax[1], kde_kws={'fill': True}, color='red', label='out-of-distribution')
    ax[1].spines['bottom'].set_linewidth(0.5)
    ax[1].spines['left'].set_linewidth(0.5)
    ax[1].spines['top'].set_linewidth(0.5)
    ax[1].spines['right'].set_linewidth(0.5)
    ax[1].set_xlabel('(b) Max Softmax Score', size=12)
    ax[1].set_ylabel('Frequency', size=12)

    handles, labels = ax[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10, ncol=2, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')

def plot_distribution(vi_pos, vi_neg, filename):
    """plot vi score distribution figure"""
    sns.set(style='white')
    fig,ax = plt.subplots(1,1,figsize=(4.5,3))

    # plot vi score distribution without ground truth
    sns.distplot(vi_pos, hist=False, ax=ax, kde_kws={'fill': True}, color='#7F95D1', label='in-distribution')
    sns.distplot(vi_neg, hist=False, ax=ax, kde_kws={'fill': True}, color='#FF82A9', label='out-of-distribution')
    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.set_xlabel('VI Probability Score w/o Ground Truth', size=15)
    ax.set_ylabel('Frequency', size=15)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', fontsize=10,ncol=2, bbox_to_anchor=(0.55, 1.08))
    fig.tight_layout()
    fig.savefig(filename, bbox_inches='tight')

def show_performance(pos, neg, method_name='Ours', recall_level=recall_level_default):
    auroc, aupr, fpr = get_measures(pos[:], neg[:], recall_level)

    print('\t\t\t' + method_name)
    print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    print('AUROC:\t\t\t{:.2f}'.format(100 * auroc))
    print('AUPR:\t\t\t{:.2f}'.format(100 * aupr))
    
def get_gpu_memory_map():
    """Get the current gpu usage.
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = np.array([int(x) for x in result.strip().split('\n')])
    # gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    print(gpu_memory)
    
    return gpu_memory