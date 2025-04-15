
    
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import math

# 定义图卷积网络模型
class GCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GCN, self).__init__()
        # 第一层GCN卷积
        self.conv1 = GCNConv(in_channels, 64)
        # 第二层GCN卷积
        self.conv2 = GCNConv(64, out_channels)
        self.linear = nn.Linear(in_channels*out_channels, out_channels)

    def forward(self, x, edge_index):
        # 第一层卷积 + ReLU激活
        x = F.relu(self.conv1(x, edge_index))
        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = self.linear(x.view(-1))
        return x
    
# 定义图卷积网络模型
class AdaptGCN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AdaptGCN, self).__init__()
        # 第一层GCN卷积
        self.conv1 = GCNConv(in_channels, 64)
        # 第二层GCN卷积
        self.conv2 = GCNConv(64, out_channels)
        self.linear1 = nn.Linear(in_channels, in_channels)
        self.linear2 = nn.Linear(in_channels*out_channels, out_channels)

    def forward(self, x):
        # 第一层卷积 + ReLU激活
        ada_matrix = self.linear1(x)
        edge_index = ada_matrix.nonzero().t().contiguous()
        x = F.relu(self.conv1(x, edge_index))
        # 第二层卷积
        x = self.conv2(x, edge_index)
        x = self.linear2(x.view(-1))
        return x
    
    
class SelfAttentionFusion(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(SelfAttentionFusion, self).__init__()
        self.attn = nn.MultiheadAttention(embed_size, num_heads)

    def forward(self, vectors):
        # vectors: [num_vectors, batch_size, embed_size]
        #vectors = vectors.unsqueeze(1)  # Add batch dimension
        output, _ = self.attn(vectors[0].unsqueeze(0), vectors[1].unsqueeze(0), vectors[2].unsqueeze(0))
        return output.squeeze()  # [batch_size, embed_size]
    
class SelfAttention(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(SelfAttention, self).__init__()
        
        # 查询、键、值的线性变换
        self.query_linear = nn.Linear(input_dim, input_dim)
        self.key_linear = nn.Linear(input_dim, input_dim)
        self.value_linear = nn.Linear(input_dim, input_dim)
        
        # 最终的线性变换
        self.output_linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        
        # 线性变换得到Q, K, V
        Q = self.query_linear(x)  # (batch_size, seq_len, input_dim)
        K = self.key_linear(x)    # (batch_size, seq_len, input_dim)
        V = self.value_linear(x)  # (batch_size, seq_len, input_dim)
        
        # 计算注意力得分
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (input_dim ** 0.5)  # (batch_size, seq_len, seq_len)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (batch_size, seq_len, seq_len)

        # 根据注意力得分加权求和
        out = torch.matmul(attn_weights, V)  # (batch_size, seq_len, input_dim)

        # 最后的线性变换
        out = self.output_linear(out)  # (batch_size, seq_len, output_dim)
        
        # 聚合整个序列的特征，通常我们对整个序列进行池化（例如平均池化）
        out = out.mean(dim=1)  # (batch_size, output_dim) 平均池化
        
        return out
    

class Adaptive_MultiGraph_Module(nn.Module):
    def __init__(self, num_nodes=137, feature_dim=32, num_features=4, num_heads=4):
        super(Adaptive_MultiGraph_Module, self).__init__()
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.num_features = num_features
        self.num_heads = num_heads
        self.node_features = torch.eye(num_nodes)
        self.GCN_adj = GCN(num_nodes, feature_dim)
        self.GCN_con = GCN(num_nodes, feature_dim)
        self.GCN_dis = GCN(num_nodes, feature_dim)
        self.GCN_sim = GCN(num_nodes, feature_dim)
        self.GCN_ada = AdaptGCN(num_nodes, feature_dim)
        self.fusion = SelfAttentionFusion(feature_dim, num_heads)
        self.attention = SelfAttention(feature_dim*2, feature_dim)

    def forward(self, adj_matrix, con_matrix, dis_matrix, sim_matrix):
        self.node_features = self.node_features.to(adj_matrix.device)
        edge_index_adj = adj_matrix.nonzero().t().contiguous()
        edge_index_con = con_matrix.nonzero().t().contiguous()
        edge_index_dis = dis_matrix.nonzero().t().contiguous()
        edge_index_sim = sim_matrix.nonzero().t().contiguous()

        out_adj = self.GCN_adj(self.node_features, edge_index_adj)
        out_con = self.GCN_con(self.node_features, edge_index_con)
        out_dis = self.GCN_dis(self.node_features, edge_index_dis)
        out_sim = self.GCN_sim(self.node_features, edge_index_sim)

        features = torch.stack([out_adj, out_con, out_dis, out_sim], dim=0)
        fusion_result = self.fusion(features)

        out_ada = self.GCN_ada(self.node_features)
        cat_ada_fusion = torch.cat((fusion_result, out_ada), dim=0)
        out = self.attention(cat_ada_fusion.unsqueeze(0).unsqueeze(0))

        return out


def get_positional_encoding(length, dim):
    """
    生成正弦和余弦位置编码
    
    参数:
    - length: 位置编码的长度
    - dim: 位置编码的维度
    
    返回:
    - pos_encoding: 位置编码矩阵，形状为 (length, dim)
    """
    pos = np.arange(length)  # 位置索引
    i = np.arange(dim)  # 维度索引
    
    # 计算角度
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(dim))  # (2i/d)的部分
    
    # 计算正弦和余弦编码
    angle = np.outer(pos, angle_rates)  # pos * angle_rates 得到每个位置的编码
    pos_encoding = np.zeros((length, dim))
    
    # 偶数维度使用 sin，奇数维度使用 cos
    pos_encoding[:, 0::2] = np.sin(angle[:, 0::2])  # 偶数维度
    pos_encoding[:, 1::2] = np.cos(angle[:, 1::2])  # 奇数维度
    
    return torch.tensor(pos_encoding, dtype=torch.float32)

class MLP(nn.Module):
    def __init__(self, input_dim, out_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(64, out_dim)

    def forward(self, x):
        x = self.linear2(self.relu(self.linear1(x)))
        return x
    
class CrossAttention(torch.nn.Module):
    def __init__(self, input_dim=32):
        super(CrossAttention, self).__init__()
        # 定义线性变换层，用于将输入的两个向量映射为 Q, K, V
        self.query_projection = torch.nn.Linear(input_dim, input_dim)
        self.key_projection = torch.nn.Linear(input_dim, input_dim)
        self.value_projection = torch.nn.Linear(input_dim, input_dim)

    def forward(self, query, key_value):
        # 1. 计算 Query, Key, Value
        Q = self.query_projection(query)  # (batch_size, input_dim)
        K = self.key_projection(key_value)  # (batch_size, input_dim)
        V = self.value_projection(key_value)  # (batch_size, input_dim)
        
        # 2. 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, 1, 1) 或者 (batch_size, seq_len, seq_len)
        
        # 3. 缩放点积注意力分数
        d_k = Q.size(-1)  # 获取 key 的维度
        attention_scores = attention_scores / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # 4. 计算 attention 权重
        attention_weights = F.softmax(attention_scores, dim=-1)  # (batch_size, 1, 1)

        # 5. 加权求和 V
        output = torch.matmul(attention_weights, V)  # (batch_size, 1, input_dim)

        # 输出是加权求和后的向量
        return output.squeeze(1)  # (batch_size, input_dim)
    
class Time_Sensitive_Module(nn.Module):
    def __init__(self, length=96, feature_dim=32, num_heads=4):
        super(Time_Sensitive_Module, self).__init__()
        self.length = length
        self.feature_dim = feature_dim
        self.MLP_time = MLP(length, feature_dim)
        self.MLP_day = MLP(length, feature_dim)
        self.MLP_holiday = MLP(length, feature_dim)
        self.MLP_weather = MLP(length, feature_dim)
        self.fusion = SelfAttentionFusion(feature_dim, num_heads)
        self.cross_attention = CrossAttention(input_dim=feature_dim)
        self.linear = nn.Linear(length, feature_dim)
        self.relu = nn.ReLU()

    def tou(self, tou):
        positional_encoding = get_positional_encoding(self.length,1).squeeze().to(tou.device)
        
        tou_feature = tou + positional_encoding
        tou_feature_out = self.relu(self.linear(tou_feature))

        return tou_feature_out
    
    def forward(self, tou, time, day, holiday, weather):
        tou_out = self.tou(tou)
        feature_time = self.MLP_time(time)
        feature_day = self.MLP_day(day)
        feature_holiday = self.MLP_holiday(holiday)
        feature_weather = self.MLP_weather(weather)
        features = torch.stack([feature_time, feature_day, feature_holiday, feature_weather], dim=0)
        f = self.fusion(features)
        
        output = self.cross_attention(tou_out.unsqueeze(0), f.unsqueeze(0))
        return output

class Feature_Fusion_Module(nn.Module):
    def __init__(self, input_dim=64, output_dim=32) -> None:
        super(Feature_Fusion_Module, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x, y):
        out = torch.cat((x, y), dim=1)
        out = self.linear(out)

        return out
    
class SpatioTemporal_Feature_Extractor(nn.Module):
    def __init__(self, num_nodes=137, length=96, feature_dim=32, num_features=4, num_heads=4):
        super(SpatioTemporal_Feature_Extractor, self).__init__()
        self.graph_model = Adaptive_MultiGraph_Module(num_nodes=num_nodes, feature_dim=feature_dim, num_features=num_features, num_heads=num_heads)
        self.temporal_model = Time_Sensitive_Module(length=length, feature_dim=feature_dim, num_heads=num_heads)
        self.fusion = Feature_Fusion_Module(input_dim=feature_dim*2, output_dim=feature_dim)

    def spatio_feature(self, adj_matrix, con_matrix, dis_matrix, sim_matrix):
        feature = self.graph_model(adj_matrix, con_matrix, dis_matrix, sim_matrix)
        return feature

    def temporal_feature(self, tou, time, day, holiday, weather):
        feature = self.temporal_model(tou, time, day, holiday, weather)
        return feature
    
    def forward(self, x):
        adj_matrix, con_matrix, dis_matrix, sim_matrix, tou, time, day, holiday, weather = x
        spatiofeature = self.spatio_feature(adj_matrix, con_matrix, dis_matrix, sim_matrix)
        temporalfeature = self.temporal_feature(tou, time, day, holiday, weather)
        output = self.fusion(spatiofeature, temporalfeature)
        return output
    
def create_ST_feature(matrix, tou, x, model):
    adj_matrix, con_matrix, dis_matrix, sim_matrix = matrix
    batch_size = x.shape[0]
    st_features = []
    s_feature = model.graph_model(adj_matrix, con_matrix, dis_matrix, sim_matrix)
    for i in range(batch_size):
        day = x[i, :, 0].squeeze()
        holiday = x[i, :, 1].squeeze()
        weather = x[i, :, 2].squeeze()
        time = x[i, :, 3].squeeze()
        t_feature = model.temporal_model(tou, time, day, holiday, weather)
        st_feature = model.fusion(s_feature, t_feature)
        st_features.append(st_feature)
    st = torch.stack(st_features, dim=0)
    return st.squeeze()

    
class CustomGRUCell(nn.Module):
    def __init__(self, input_dim, feature_dim, hidden_dim):
        super(CustomGRUCell, self).__init__()

        # 输入到各个门的权重矩阵
        self.cat_linear1 = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.fusion_linear1 = nn.Linear(hidden_dim+feature_dim, hidden_dim)
        self.cat_linear2 = nn.Linear(hidden_dim + input_dim, hidden_dim)
        self.fusion_linear2 = nn.Linear(hidden_dim+feature_dim, hidden_dim)

    def forward(self, x, h_prev, st_feature):
        # 计算更新门 z_t 和 重置门 r_t
        cat_x_h_prev1 = torch.cat((x, h_prev), dim=1)
        cat_feature1 = self.cat_linear1(cat_x_h_prev1)
        cat_feature_1 = torch.cat((cat_feature1, st_feature), dim=1)
        fusion_feature1 = self.fusion_linear1(cat_feature_1)
        z_t = torch.sigmoid(fusion_feature1)  # 更新门
        r_t = torch.sigmoid(fusion_feature1)  # 重置门

        # 计算候选隐藏状态
        cat_x_h_prev2 = torch.cat((x, r_t * h_prev), dim=1)
        cat_feature2 = self.cat_linear2(cat_x_h_prev2)
        cat_feature_2 = torch.cat((cat_feature2, st_feature), dim=1)
        fusion_feature2 = self.fusion_linear2(cat_feature_2)
        h_tilde = torch.tanh(fusion_feature2)  # 候选隐藏状态

        # 计算最终的隐藏状态 h_t
        h_t = (1 - z_t) * h_prev + z_t * h_tilde  # GRU的最终隐藏状态

        return h_t

# GRU单元的使用示例
class CustomGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, matrix, tou, num_nodes=137, length=96, feature_dim=32, num_features=4, num_heads=4):
        super(CustomGRU, self).__init__()
        self.hidden_dim = hidden_dim
        self.matrix = matrix
        self.tou = tou
        self.model = SpatioTemporal_Feature_Extractor(num_nodes=num_nodes, length=length, feature_dim=feature_dim, num_features=num_features, num_heads=num_heads)

        # 定义一个GRU层
        self.gru_cell = CustomGRUCell(input_dim, feature_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size, seq_len, _, _ = x.size()
        
        # 初始化隐藏状态
        h = torch.zeros(batch_size, self.hidden_dim).to(x.device)

        # 逐步输入序列的每一个时间步
        for t in range(seq_len):
            st = create_ST_feature(self.matrix, self.tou, x[:, t, :, :].squeeze(), self.model)
            h = self.gru_cell(x[:, t, :, -1].squeeze(), h, st)

        out = self.linear(h)
        return out
