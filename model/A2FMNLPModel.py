import sys

import torch
import torch.nn as nn
import numpy as np

from .GAT import GraphTrans, ImprovedFCStacked
from .MultiLayer import SemanticAttention, BitwiseMultipyLogis, ShareNetBottom
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from src.utils import accuracy, LogisticRegression


class MWTPModel(nn.Module):
    def __init__(self,layer_num, num_nodes, feat_data, adjs,  adj_lists, emb_dim,device,lr,ablation,model_type):
        super(MWTPModel, self).__init__()
        self.layer_num = layer_num
        self.num_nodes = num_nodes
        self.feat_data = feat_data
        self.adj_lists = adj_lists
        self.emb_dim = emb_dim
        self.device = device
        self.num_samples1 = 10
        self.num_samples2 = 10

        # self.enc = nn.ModuleList([GATLayer(torch.Tensor(feat_data[i]),adj_lists[i], self.emb_dim) for i in range(layer_num)])
        #self.enc = nn.ModuleList([GAT(torch.tensor(feat_data[i],dtype=torch.float),torch.tensor(adjs[i]),nfeat=feat_data.shape[2],nhid=8,nclass=emb_dim,dropout=0.6,nheads=8,alpha=0.2,device=device) for i in range(layer_num)])
        self.enc = nn.ModuleList([GraphTrans(torch.tensor(feat_data[i],dtype=torch.float),torch.tensor(adjs[i]),nfeat=feat_data.shape[2],nhid=8,nclass=128,dropout=0.5,heads=8,device=device) for i in range(layer_num)])
        self.enc_two = nn.ModuleList([ImprovedFCStacked(feat_data.shape[2], [256, 128, 64], emb_dim,
                                    torch.tensor(feat_data[l],dtype=torch.float), device=self.device)
                                      for l in range(self.layer_num)])

        self.MWTP = SupervisedGraphSage(self.enc, self.enc_two, emb_dim, layer_num,device,ablation,model_type )
        self.MWTP.to(device)
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.MWTP.parameters()), lr=lr)#, weight_decay=1e-4)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.MWTP.parameters()),
            lr=0.001,  # 推荐初始学习率比SGD小，通常设为 1e-3
            betas=(0.9, 0.999),
            weight_decay=0  # 可根据情况添加 L2 正则
        )
    def forward(self, nodes, targets,layer_predict,ori_adj,run_type='train'):
        #print('----------------------------------------------------------------------------------')
        #print('layer_predict',layer_predict)
        #print('nodes',nodes.shape)
        predict = self.MWTP(nodes,layer_predict,ori_adj)
        loss = self.MWTP.loss(predict,targets)
        acc = self.MWTP.acc(predict,targets)
        if run_type == 'valid':
            return loss, acc
        auc = self.MWTP.Auc(predict,targets)
        ap = self.MWTP.ap(predict,targets)
        f1 = self.MWTP.f1(predict,targets)
        return loss, acc, auc, ap, f1

class SupervisedGraphSage(nn.Module):

    def __init__(self, enc, enc2, embed_dim, layer_num, device,ablation,model_type ):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.enc2 = enc2
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.device = device
        self.criterion = nn.BCELoss()
        self.accuracy = accuracy
        self.ablation = ablation
        self.logis = nn.ModuleList(LogisticRegression(self.embed_dim, 1,self.device) for _ in range(layer_num))

        self.layerNodeAttention_weight = ShareNetBottom(self.layer_num, self.embed_dim, device,ablation,model_type).to(device)

        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(embed_dim * 2, embed_dim).to(device)) for _ in range(layer_num)
        ])
        for l in range(layer_num):
            nn.init.xavier_uniform_(self.W[l], 1.414)
    def forward(self, nodes,layer_predict,ori_adj ):
        embeds = []
        for l in range(self.layer_num):
            # embeds.append(self.enc[l](nodes)+self.enc2[l](nodes))
            if self.ablation == 'womlp':
                embeds.append(self.enc[l](nodes))
            elif self.ablation == 'wogat':
                embeds.append(self.enc2[l](nodes))
            else:
                embeds.append(torch.cat([self.enc[l](nodes),self.enc2[l](nodes)],dim=1) @ self.W[l])


            # print(embeds[l],embeds[l].shape)
        result = self.layerNodeAttention_weight(torch.stack(embeds),layer_predict)

        #predict = self.logis[layer_predict](result, nodes , ori_adj ,layer_predict)  #找出预测到的真实边
        predict = self.logis[layer_predict](result)
        return predict

    def loss(self, predict, targets):
        return self.criterion(predict, targets.to(self.device))
    def acc(self, predict, targets):
        return self.accuracy(predict, targets.to(self.device))
    def Auc(self, predict, targets):
        return roc_auc_score(targets, predict.cpu().detach().numpy())

    def ap(self, predict, targets):
        return average_precision_score(targets, predict.cpu().detach().numpy())
    def f1(self, predict, targets, threshold=0.5):
        predict_np = predict.cpu().detach().numpy()
        threshold = np.median(predict_np)
        binary_predictions = (predict_np >= threshold).astype(int)
        return f1_score(targets, binary_predictions)
