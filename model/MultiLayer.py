import torch.nn as nn
import torch
import torch.nn.functional as F

class SemanticAttention(nn.Module):
    def __init__(self,layer_num, features_num,device, dropout, alpha):
        super(SemanticAttention, self).__init__()
        self.features_num = features_num
        self.dropout = dropout
        self.alpha = alpha
        self.device = device
        self.leakyReLU = nn.LeakyReLU(alpha)  # LeakyReLU
        self.trans = nn.Parameter(torch.eye(features_num, device=device))
        self.W = nn.Parameter(torch.empty(layer_num,features_num, features_num))
        self.b = nn.Parameter(torch.empty(1, features_num))
        self.q = nn.Parameter(torch.empty(features_num, 1))
        self.bias = nn.Parameter(torch.zeros(features_num, device=device))
        self.tanh = nn.Tanh()
        self.initParameter()
    def initParameter(self):
        nn.init.xavier_uniform_(self.trans.data, 1.414)
        for i in range(3):
            nn.init.xavier_uniform_(self.W[i], gain=1.414)
        nn.init.xavier_uniform_(self.b.data, 1.414)
        nn.init.xavier_uniform_(self.q.data, 1.414)
    def forward(self, node_features, layer_predict=0):
        projection_features = torch.tanh(torch.matmul(node_features, self.trans) + self.bias)


        aggregated_embeddings = projection_features[layer_predict]+(normalized_weights.unsqueeze(-1) * projection_features).sum(dim=0)  # (512, 128)

        return aggregated_embeddings


class BitwiseMultipyLogis(nn.Module):
    def __init__(self, layer_num, features_num,device):
        super(BitwiseMultipyLogis, self).__init__()
        self.features_num = features_num
        self.logis = LogisticVector(features_num, 1)
        self.trans = nn.Parameter(torch.eye(features_num).to(device))
        self.bias = nn.Parameter(torch.zeros(features_num, device=device))
        self.active = nn.Sigmoid()
        self.layer_num = layer_num
        self.theta = nn.Parameter(torch.randn(self.layer_num, features_num, features_num))
        self.initParameter()

    def initParameter(self):
        nn.init.xavier_uniform_(self.trans.data, 1.414)
        for i in range(self.layer_num):
            nn.init.xavier_uniform_(self.theta[i], 1.414)

    def forward(self, node_features, layer_predict=0):
        projection_features = torch.tanh(torch.matmul(node_features, self.trans) + self.bias)

        bitwise_features = projection_features * projection_features[layer_predict]
        bitwise_features = torch.bmm(bitwise_features, self.theta)

        bitwise_flat = bitwise_features.view(-1, bitwise_features.size(-1))  

        output_flat = self.logis(bitwise_flat)  

        output = output_flat.view(bitwise_features.size(0), bitwise_features.size(1)).squeeze(-1)
        bitwise_softmax_normalized = F.softmax(output, dim=0)
        aggregated_embeddings = projection_features[layer_predict] + (
                torch.sum(bitwise_softmax_normalized.unsqueeze(2) * projection_features, dim=0)) 
        return aggregated_embeddings
class LogisticVector(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.active = nn.Sigmoid() # output layer

    def forward(self, x):
        value = self.parameter(x)
        out = self.active(value)
        return out.squeeze()

class ShareNetBottom (nn.Module):
    def __init__(self, layer_num, features_num,device,ablation,model_type):
        super(ShareNetBottom, self).__init__()
        self.features_num = features_num
        self.logis = torch.nn.Linear(features_num, 1)#LogisticVector(features_num, 1)
        self.trans = nn.Parameter(torch.eye(features_num).to(device))
        self.bias = nn.Parameter(torch.zeros(features_num, device=device))
        self.active = nn.Sigmoid()
        self.layer_num = layer_num
        self.ablation = ablation
        self.model_type = model_type
        self.theta = nn.Parameter(torch.randn(self.layer_num, features_num, features_num))

        self.initParameter()

    def initParameter(self):
        nn.init.xavier_uniform_(self.trans.data, 1.414)
        for i in range(self.layer_num):
            nn.init.xavier_uniform_(self.theta[i], 1.414)

        self.W_g = nn.Linear(self.features_num, self.features_num, bias=False) 


    def forward(self, node_features, layer_predict=0):
        if self.model_type == 'gate':
            projection_features = node_features
        elif self.model_type == 'add':
            projection_features = torch.tanh(torch.matmul(node_features, self.trans) + self.bias)


        if self.ablation == 'wologit':
            shared_info = torch.mean(projection_features, dim=0)
            aggregated_embeddings = self.attention_aggregation(node_features[layer_predict] , shared_info)
            return aggregated_embeddings

        bitwise_features = projection_features * projection_features[layer_predict]
        bitwise_features = torch.bmm(bitwise_features, self.theta)


        bitwise_flat = bitwise_features.view(-1, bitwise_features.size(-1)) 

        output_flat = self.active(self.logis(bitwise_flat))#self.logis(bitwise_flat)  
        output_flat = output_flat.squeeze()

        output = output_flat.view(bitwise_features.size(0), bitwise_features.size(1)).squeeze(-1)
        bitwise_softmax_normalized = F.softmax(output, dim=0)
        shared_info = torch.sum(bitwise_softmax_normalized.unsqueeze(2) * projection_features, dim=0)
        if self.model_type == 'add':
            aggregated_embeddings = node_features[layer_predict] + shared_info
        elif self.model_type == 'gate':
            aggregated_embeddings = self.attention_aggregation(node_features[layer_predict] , shared_info)
        return aggregated_embeddings
    def attention_aggregation(self,A,B):
        G = torch.sigmoid(self.W_g(A) + self.W_g(B))  
        H = G * A + (1 - G) * B  
        return H
