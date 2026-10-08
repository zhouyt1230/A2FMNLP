import sys
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from typing import Union

from pandas.core.interchange.from_dataframe import primitive_column_to_ndarray

from .GTN import GraphTrans, ImprovedFCStacked
from .MultiLayer import SemanticAttention, BitwiseMultipyLogis, ShareNetBottom
from sklearn.metrics import roc_auc_score, f1_score,average_precision_score
from src.utils import accuracy, LogisticRegression


class MWTPModel(nn.Module):
    def __init__(self,layer_num, num_nodes, feat_data, adjs, adj_lists, emb_dim,device):
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
        #self.enc = nn.ModuleList([GAT(torch.tensor(feat_data[i],dtype=torch.float),torch.tensor(adjs[i]),nfeat=feat_data.shape[2],nhid=8,nclass=128,dropout=0.6,nheads=8,alpha=0.2,device=device) for i in range(layer_num)])
        self.enc = nn.ModuleList([GraphTrans(torch.tensor(feat_data[i],dtype=torch.float),torch.tensor(adjs[i]),nfeat=feat_data.shape[2],nhid=8,nclass=128,dropout=0.5,heads=8,device=device) for i in range(layer_num)])
        self.enc_two = nn.ModuleList([ImprovedFCStacked(feat_data.shape[2], [256, 128, 64], emb_dim,
                                    torch.tensor(feat_data[l],dtype=torch.float), device=self.device)
                                      for l in range(self.layer_num)])

        self.MWTP = SupervisedGraphSage(self.enc, self.enc_two, emb_dim, layer_num,device)
        self.MWTP.to(device)
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.MWTP.parameters()), lr=0.7)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.MWTP.parameters()),
            lr=0.001,  # 推荐初始学习率比SGD小，通常设为 1e-3
            betas=(0.9, 0.999),
            weight_decay=0  # 可根据情况添加 L2 正则
        )

    def forward(self, nodes, targets,layer_predict,run_type):
        #print('nodes:',nodes)
        #print('nodes.shape:',nodes.shape)
        #print('layer_predict:',layer_predict)
        predict = self.MWTP(nodes,layer_predict,run_type)
        loss = self.MWTP.loss(predict,targets)
        acc = self.MWTP.acc(predict,targets)
        if run_type == 'valid':
            return loss, acc
        auc = self.MWTP.Auc(predict,targets)
        ap = self.MWTP.ap(predict,targets)
        f1 = self.MWTP.f1(predict,targets)
        return loss, acc, auc, ap, f1

    def forward1(self, nodes,targets,layer_predict):
        #print('nodes:',nodes)
        #print('nodes.shape:',nodes.shape)
        #print('layer_predict:',layer_predict)
        predict,feature= self.MWTP(nodes,layer_predict)
        loss = self.MWTP.loss(predict,targets)
        acc = self.MWTP.acc(predict,targets)

        return feature,loss, acc



class SupervisedGraphSage(nn.Module):

    def __init__(self, enc, enc2, embed_dim, layer_num, device):
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.enc2 = enc2
        self.embed_dim = embed_dim
        self.layer_num = layer_num
        self.device = device
        self.criterion = nn.BCELoss()
        self.accuracy = accuracy
        self.logis = nn.ModuleList(LogisticRegression(self.embed_dim, 1,self.device) for _ in range(layer_num))

        self.layerNodeAttention_weight = ShareNetBottom(self.layer_num, self.embed_dim, device).to(device)

        self.W = nn.ParameterList([
            nn.Parameter(torch.empty(embed_dim * 2, embed_dim).to(device)) for _ in range(layer_num)
        ])
        for l in range(layer_num):
            nn.init.xavier_uniform_(self.W[l], 1.414)
    # def forward2(self, nodes,layer_predict):
    #     embeds = []
    #     for l in range(self.layer_num):
    #         # embeds.append(self.enc[l](nodes)+self.enc2[l](nodes))
    #         embeds.append(torch.cat([self.enc[l](nodes),self.enc2[l](nodes)],dim=1) @ self.W[l])
    #         #embeds = torch.relu(embeds)
    #
    #
    #         # print(embeds[l],embeds[l].shape)
    #     result = self.layerNodeAttention_weight(torch.stack(embeds),layer_predict)
    #     predict = self.logis[layer_predict](result)
    #     #print("predict", predict)
    #     return predict

    def forward(self, nodes,layer_predict,run_type):
        embeds = []
        features_all = []
        for l in range(self.layer_num):
            # embeds.append(self.enc[l](nodes)+self.enc2[l](nodes))
            embeds.append(torch.relu(torch.cat([self.enc[l](nodes),self.enc2[l](nodes)],dim=1) @ self.W[l]))
            #embeds.append(torch.cat([self.enc[l](nodes), self.enc2[l](nodes)], dim=1) @ self.W[l])
            #embeds = torch.relu(embeds)
            # print(embeds[l],embeds[l].shape)
        result = self.layerNodeAttention_weight(torch.stack(embeds),layer_predict)
        #print('result',result.shape)

        features_layer = []
        for j in range(0, 246):
             indices = [i for i, val in enumerate(nodes) if val == j]
             if not indices:
                 continue
             # 选取对应的特征向量
             selected_features = [result[i] for i in indices]
             # 转换为 NumPy 数组（二维：n × 128）
             # selected_features = np.array(selected_features)
             selected_features = torch.stack(selected_features, dim=0)

             # 计算每一维特征的平均值（输出也是一个 128 维向量）
             # average_feature = np.mean(selected_features, axis=0)
             average_feature = selected_features.mean(dim=0)
             # print(average_feature)
             features_layer.append(average_feature)
        if run_type == 'train' and result.shape[0] == 512:
             with open(f'feature_layer_{layer_predict}.txt', 'w') as f:
                 for tensor in features_layer:
                     np_array = tensor.detach().cpu().numpy()  # 先转为 numpy array
                     np_array = np_array.flatten()  # 展平
                     line = ' '.join(map(str, np_array))  # 转为字符串
                     f.write(line + '\n')  # 每个 tensor 占一行


        #print('nodes',nodes.shape)
        #print('result',result.shape)
        #print('result[0]',result[0].shape)
        #self.save_result_to_txt(result, 'result.txt')
        predict = self.logis[layer_predict](result)
        #print('predict',predict)
        return predict

    def forward1(self, nodes,layer_predict):
        embeds = []
        features_all = []
        for l in range(self.layer_num):
            nodes1 = nodes[l]
            nodes2 = torch.tensor(nodes1, dtype=torch.long)
            #print("输入节点索引的最大值:", nodes2.max().item())
            #print("输入节点索引的最小值:", nodes2.min().item())
            # embeds.append(self.enc[l](nodes)+self.enc2[l](nodes))
            embeds.append(torch.cat([self.enc[l](nodes2),self.enc2[l](nodes2)],dim=1) @ self.W[l])
            embeds1 = embeds[l]
            features_layer = []
            for j in range(0,246):
                indices = [i for i, val in enumerate(nodes1) if val == j]
                if not indices:
                    continue
                # 选取对应的特征向量
                selected_features = [embeds1[i] for i in indices]
                # 转换为 NumPy 数组（二维：n × 128）
                #selected_features = np.array(selected_features)
                selected_features = torch.stack(selected_features, dim=0)

                # 计算每一维特征的平均值（输出也是一个 128 维向量）
                #average_feature = np.mean(selected_features, axis=0)
                average_feature = selected_features.mean(dim=0)
                #print(average_feature)
                features_layer.append(average_feature)
            features_all.append(torch.stack(features_layer))

        result = self.layerNodeAttention_weight(torch.stack(embeds), layer_predict)
        # print('nodes',nodes.shape)
        # print('result',result.shape)
        # print('result[0]',result[0].shape)
        # self.save_result_to_txt(result, 'result.txt')
        predict = self.logis[layer_predict](result)





        return predict,features_all



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

    def save_result_to_txt(self, result, filename):
        numpy_array = result.detach().cpu().numpy().flatten()
        with open(filename, 'w',newline='\n') as f:
            #for item in result:
                #f.write("%s\n" % item)
                f.write('\n'.join(numpy_array))
        print(f"Results saved to {filename}")
