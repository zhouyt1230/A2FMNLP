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
        self.leakyReLU = nn.LeakyReLU(alpha)  # LeakyReLU激活函数
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

        """
            features: (3, 512, 128) -> 3 layers of node embeddings, each of shape (512, 128)
            W: (128, 128) -> transformation matrix for embedding
            b: (1, 128) -> bias for transformation
            q: (128, 1) -> query vector for computing attention weights
            """

        # Step 1: Transformation: Apply W and b, then apply tanh activation
        # transformed_features = torch.tanh(torch.matmul(projection_features, self.W) + self.b)  # (3, 512, 128)
        transformed_features = torch.tanh(torch.matmul(projection_features, self.W.transpose(1, 2)) + self.b)

        # Step 2: Compute attention weights for each layer's node embeddings
        weights = torch.matmul(transformed_features, self.q)  # (3, 512, 1)

        # Step 3: Squeeze the last dimension to get the shape (3, 512)
        weights = weights.squeeze(-1)  # Now weights.shape = (3, 512)
        # print(weights)
        # print(weights.shape)
        weights = torch.exp(weights)
        # print(weights)


        # Step 4: Add weights of layers (2nd and 3rd) to the 1st layer's weights
        # Use broadcasting to add 1st layer's weights to the 2nd and 3rd layers
        adjusted_weights = weights + weights[layer_predict]  # Broadcast addition (3, 512)

        # Step 5: Normalize the adjusted weights
        # weight_sum = adjusted_weights.sum(dim=0, keepdim=True)  # (1, 512)
        # normalized_weights = adjusted_weights / weight_sum  # Normalize: (3, 512)
        normalized_weights = F.softmax(adjusted_weights, dim=0)
        # print(normalized_weights)
        # print(normalized_weights.shape)

        # Step 6: Aggregation - compute weighted sum of embeddings for each node
        # For each node, calculate weighted sum of embeddings from 3 layers
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

        # 将输入张量x的形状调整为 (3*512, 128) 进行批量计算
        bitwise_flat = bitwise_features.view(-1, bitwise_features.size(-1))  # 将 (3, 512, 128) 重塑为 (3*512, 128)
        # 批量执行回归操作，得到 (3*512, 1)
        output_flat = self.logis(bitwise_flat)  # 输出形状为 (3*512, 1)
        # 将输出重塑回 (3, 512)
        output = output_flat.view(bitwise_features.size(0), bitwise_features.size(1)).squeeze(-1)
        bitwise_softmax_normalized = F.softmax(output, dim=0)
        aggregated_embeddings = projection_features[layer_predict] + (
                torch.sum(bitwise_softmax_normalized.unsqueeze(2) * projection_features, dim=0))  # (512, 128)
        return aggregated_embeddings
class LogisticVector(torch.nn.Module):
    def __init__(self, n_feature, n_hidden):
        super(LogisticVector, self).__init__()
        self.n_feature = n_feature
        self.parameter = torch.nn.Linear(n_feature, n_hidden)  # hidden layer
        self.active = nn.Sigmoid()  ####  # output layer

    def forward(self, x):
        value = self.parameter(x)
        out = self.active(value)
        return out.squeeze()

class ShareNetBottom (nn.Module):
    def __init__(self, layer_num, features_num,device):
        super(ShareNetBottom, self).__init__()
        self.features_num = features_num
        self.logis = torch.nn.Linear(features_num, 1)#LogisticVector(features_num, 1)
        self.trans = nn.Parameter(torch.eye(features_num).to(device))
        self.bias = nn.Parameter(torch.zeros(features_num, device=device))
        self.active = nn.Sigmoid()
        self.layer_num = layer_num
        self.theta = nn.Parameter(torch.randn(self.layer_num, features_num, features_num))

        # self.W_Q = torch.nn.Parameter(torch.randn(features_num, features_num))
        # self.W_K = torch.nn.Parameter(torch.randn(features_num, features_num))
        # self.W_V = torch.nn.Parameter(torch.randn(features_num, features_num))

        self.initParameter()

    def initParameter(self):
        nn.init.xavier_uniform_(self.trans.data, 1.414)
        for i in range(self.layer_num):
            nn.init.xavier_uniform_(self.theta[i], 1.414)

        self.W_g = nn.Linear(self.features_num, self.features_num, bias=False)  # 门控参数

        # self.W_gate = nn.Sequential(
        #     nn.Linear(self.features_num * 2, self.features_num),
        #     nn.ReLU(),
        #     nn.Linear(self.features_num, 1),
        #     nn.Sigmoid()
        # )

    def forward(self, node_features, layer_predict):
        #projection_features = torch.tanh(torch.matmul(node_features, self.trans) + self.bias)
        projection_features = node_features
        bitwise_features = projection_features * projection_features[layer_predict]
        bitwise_features = torch.bmm(bitwise_features, self.theta)

        # 将输入张量x的形状调整为 (3*512, 128) 进行批量计算
        bitwise_flat = bitwise_features.view(-1, bitwise_features.size(-1))  # 将 (3, 512, 128) 重塑为 (3*512, 128)
        # 批量执行回归操作，得到 (3*512, 1)
        output_flat = self.active(self.logis(bitwise_flat))#self.logis(bitwise_flat)  # 输出形状为 (3*512, 1)
        output_flat = output_flat.squeeze()
        # 将输出重塑回 (3, 512)
        output = output_flat.view(bitwise_features.size(0), bitwise_features.size(1)).squeeze(-1)
        bitwise_softmax_normalized = F.softmax(output, dim=0)
        shared_info = torch.sum(bitwise_softmax_normalized.unsqueeze(2) * projection_features, dim=0)
        aggregated_embeddings = self.attention_aggregation(node_features[layer_predict] , shared_info)
        #aggregated_embeddings = node_features[layer_predict] + shared_info
        # aggregated_embeddings = projection_features[layer_predict] + \
        #                         torch.sum(bitwise_softmax_normalized.unsqueeze(2) * projection_features, dim=0)  # (512, 128)
        return aggregated_embeddings

    def attention_aggregation(self,A,B):
        G = torch.sigmoid(self.W_g(A) + self.W_g(B))  # 计算门控权重 G
        H = G * A + (1 - G) * B  # 门控融合
        return H
        # fusion_weight = self.W_gate(torch.cat([A, B], dim=-1))  # (10, 1)
        # H = fusion_weight * A + (1 - fusion_weight) * B  # (10, 128)
        # return H