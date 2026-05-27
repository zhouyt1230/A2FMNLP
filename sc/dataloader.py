from torch.utils.data import Dataset,DataLoader
import torch
import numpy as np
import pickle as pkl
import random
import os

class CustomDataset(Dataset):
    def __init__(self, idx, labels, node_num):
        self.idx = idx 
        self.labels = labels  
        self.node_num = node_num  

    def __len__(self):
        return len(self.idx) 

    def __getitem__(self, idx):
        ids = self.idx[idx]
        targets = self.labels[ids]
        row, col = divmod(ids, self.node_num)
        node_list_batch = torch.tensor([row, col])
        return [node_list_batch, targets]
def construct_loader(idx, label, layer_num, num_nodes,batch_num):
    data_loaders, max_len = [], 0
    for l in range(layer_num):
        dataset_loader = CustomDataset(idx[l], label[l], num_nodes)
        data_loader = DataLoader(dataset_loader, batch_size=batch_num, shuffle=False)
        data_loaders.append(data_loader)
        if max_len < len(data_loader):
            max_len = len(data_loader)
    return data_loaders, max_len

def get_weak_info(datsetname, idx_train, idx_val, idx_test, ori_adj, labels, layer_num, num_nodes):
    ''' search weak ties'''
    print('Start generate strong ties and weak ties.')
    #weak_dic = pkl.load(open('./community res/louvain/{}-dic.pkl'.format(datsetname), "rb"))
    current_dir = os.path.dirname(os.path.abspath(__file__))  
    project_root = os.path.dirname(current_dir) 
    file_path = os.path.join(
        project_root,
        "community res",  
        "louvain",
        f"{datsetname}-dic.pkl" 
    )
    with open(file_path, "rb") as f:
        weak_dic = pkl.load(f)
    wdic = weak_dic['dic_all']
    idx_test_weak, idx_test_wlen = [], []
    for l in range(layer_num):
        print(
            f'idx_train len: {len(idx_train[l])}  |  idx_val len: {len(idx_val[l])}  |  idx_test len: {len(idx_test[l])}')
        weak_true_edges, weak_false_edges = [],[]
        test_real_tie = []

        for item in idx_test[l]:
            e1, e2 = divmod(item, num_nodes)
            if ori_adj[l][e1][e2]==1 and e1 != e2:
                test_real_tie.append((l+1,e1,e2,1))
            #print('test_real_tie:',test_real_tie)
            if wdic[l][int(e1)] != wdic[l][int(e2)]:
                if ori_adj[l][int(e1)][int(e2)] == 1:
                    weak_true_edges.append(item)
                else:
                    weak_false_edges.append(item)
        print(
            f'init situation len(weak_true_edges):{len(weak_true_edges)}, len(weak_false_edges):{len(weak_false_edges)}')
        min_weak_num = min(len(weak_true_edges), len(weak_false_edges))
        weak_true_edges = random.sample(weak_true_edges, min_weak_num)
        weak_false_edges = random.sample(weak_false_edges, min_weak_num)
        print(
            f'after deal situation len(weak_true_edges):{len(weak_true_edges)}, len(weak_false_edges):{len(weak_false_edges)}')
        print('======================================')
        w_t = weak_true_edges + weak_false_edges
        random.shuffle(w_t)
        idx_test_weak.append(w_t)
        idx_test_wlen.append([len(weak_true_edges), len(weak_false_edges)])
    idx_test_weak_loaders, max_len_w = construct_loader(idx_test_weak, labels, layer_num, num_nodes, 1024)
    print('Generate strong ties and weak ties done!')
    return [[idx_test_weak_loaders, max_len_w],idx_test_wlen]

def get_dataset(dataname='ckm'):
    data_get = np.load(rf'C:\Users\zhouyitian\Desktop\3revise_model\data_pkl\{dataname}.npz')
    ori_adj, break_adj, feats = data_get['ori_adj'], data_get['adj'], data_get['feat_data']
    if dataname == 'ckm' or dataname == 'amazon' or dataname == 'lazega':
        idx_train = [data_get['idx_train'], data_get['idx_train1'], data_get['idx_train2']]
        idx_val = [data_get['idx_val'], data_get['idx_val1'], data_get['idx_val2']]
        idx_test = [data_get['idx_test'], data_get['idx_test1'], data_get['idx_test2']]
    elif dataname == 'imdb' or dataname == 'acm':
        idx_train = [data_get['idx_train'], data_get['idx_train1']]
        idx_val = [data_get['idx_val'], data_get['idx_val1']]
        idx_test = [data_get['idx_test'], data_get['idx_test1']]
    return ori_adj, break_adj, feats, idx_train, idx_val, idx_test
