
    
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

def create_wo_fusion_feature(matrix, tou, x, model):
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
        st_feature = (s_feature + t_feature) / 2.0
        st_features.append(st_feature)
    st = torch.stack(st_features, dim=0)
    return st.squeeze()

def create_S_feature(matrix, model, batch_size):
    adj_matrix, con_matrix, dis_matrix, sim_matrix = matrix
    s_features = []
    s_feature = model(adj_matrix, con_matrix, dis_matrix, sim_matrix)
    for i in range(batch_size):
        s_features.append(s_feature)
    st = torch.stack(s_features, dim=0)
    return st.squeeze()

def create_T_feature(tou, x, model):
    #adj_matrix, con_matrix, dis_matrix, sim_matrix = matrix
    batch_size = x.shape[0]
    t_features = []
    for i in range(batch_size):
        day = x[i, :, 0].squeeze()
        holiday = x[i, :, 1].squeeze()
        weather = x[i, :, 2].squeeze()
        time = x[i, :, 3].squeeze()
        t_feature = model(tou, time, day, holiday, weather)
        t_features.append(t_feature)
    st = torch.stack(t_features, dim=0)
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

# GRU单元的使用示例
class CustomGRU_abla_Fusion(nn.Module):
    def __init__(self, input_dim, hidden_dim, matrix, tou, num_nodes=137, length=96, feature_dim=32, num_features=4, num_heads=4):
        super(CustomGRU_abla_Fusion, self).__init__()
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
            st = create_wo_fusion_feature(self.matrix, self.tou, x[:, t, :, :].squeeze(), self.model)
            h = self.gru_cell(x[:, t, :, -1].squeeze(), h, st)

        out = self.linear(h)
        return out
    
# GRU单元的使用示例
class CustomGRU_abla_S(nn.Module):
    def __init__(self, input_dim, hidden_dim, matrix, tou, num_nodes=137, length=96, feature_dim=32, num_features=4, num_heads=4):
        super(CustomGRU_abla_S, self).__init__()
        self.hidden_dim = hidden_dim
        self.matrix = matrix
        self.tou = tou
        self.model = Time_Sensitive_Module(length=length, feature_dim=feature_dim, num_heads=num_heads)
        #self.model.load_state_dict(all_model.model.temporal_model.state_dict())

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
            st = create_T_feature(self.tou, x[:, t, :, :].squeeze(), self.model)
            h = self.gru_cell(x[:, t, :, -1].squeeze(), h, st)

        out = self.linear(h)
        return out
    
# GRU单元的使用示例
class CustomGRU_abla_T(nn.Module):
    def __init__(self, input_dim, hidden_dim, matrix, tou, num_nodes=137, length=96, feature_dim=32, num_features=4, num_heads=4):
        super(CustomGRU_abla_T, self).__init__()
        self.hidden_dim = hidden_dim
        self.matrix = matrix
        self.tou = tou
        self.model = Adaptive_MultiGraph_Module(num_nodes=num_nodes, feature_dim=feature_dim, num_features=num_features, num_heads=num_heads)
        #self.model.load_state_dict(all_model.model.graph_model.state_dict())
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
            st = create_S_feature(self.matrix, self.model, batch_size)
            h = self.gru_cell(x[:, t, :, -1].squeeze(), h, st)

        out = self.linear(h)
        return out



# 定义 MLP 模型
class CustomMLP(nn.Module):
    def __init__(self, input_dim=672, hidden_dims=[256, 128], output_dim=96):
        super(CustomMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dims[0])
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRUModel, self).__init__()
        
        # GRU 层
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # 全连接层，用于将 GRU 的输出映射到预测结果
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # GRU 层的输出，h_n 是最后一个时间步的隐藏状态
        out, h_n = self.gru(x)
        
        # 取 GRU 输出的最后一个时间步的隐藏状态
        out = out[:, -1, :]  # 选择最后一个时间步的输出
        
        # 通过全连接层得到最终预测值
        out = self.fc(out)
        
        return out
    
class LSTMModel(nn.Module):
    def __init__(self, input_dim=1, hidden_size=128, num_layers=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 96)  # 预测未来 96 个时间步

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)  
        out, _ = self.lstm(x, (h0, c0))  
        out = self.fc(out[:, -1, :])  # 取最后一个时间步的隐藏状态
        return out  # 输出形状 (batch_size, 96)

# ConvLSTM模型定义
class ConvLSTM(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, output_channels):
        super(ConvLSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_channels = hidden_channels

        # 定义ConvLSTM层
        self.convlstm = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels
            out_channels = hidden_channels
            self.convlstm.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2)
            )
        
        # 全连接层，用于输出96个时间步的预测
        self.fc = nn.Linear(hidden_channels * 672, output_channels)  # Adjust based on your input size

    def forward(self, x):
        # x形状是(batch_size, 1, 672) 需要扩展为(batch_size, 1, 672, 1)
        x = x.unsqueeze(-1).unsqueeze(1)  # 添加一个维度，形成(batch_size, 1, 672, 1)
        
        # ConvLSTM层的前向传递
        h = x
        for conv in self.convlstm:
            h = torch.relu(conv(h))
        
        h = h.view(h.size(0), -1)  # Flatten before passing to fully connected layer
        output = self.fc(h)
        return output

# 定义时空特征提取模块
class SpatioTemporalEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(SpatioTemporalEncoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        
        x = x.permute(0, 2, 1)  # 调整形状为 (batch_size, input_dim, seq_length)
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.permute(0, 2, 1)  # 调整回 (batch_size, seq_length, hidden_dim)
        x, _ = self.lstm(x)
        return x

# 定义ST-SSL模型
class ST_SSL(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(ST_SSL, self).__init__()
        self.encoder = SpatioTemporalEncoder(input_dim, hidden_dim, num_layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.unsqueeze(-1)
        #print(x.shape)
        x = self.encoder(x)  # 提取时空特征
        x = self.fc(x[:, -1, :])  # 取最后一个时间步的特征
        return x

# DLinear模型
class DLinear(nn.Module):
    def __init__(self, input_len=672, output_len=96):
        super(DLinear, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        
        # 线性层分别用于趋势分量和残差分量
        self.trend_linear = nn.Linear(input_len, output_len, bias=False)
        self.trend_to_x = nn.Linear(output_len, input_len, bias=False)
        self.residual_linear = nn.Linear(input_len, output_len, bias=False)

    def forward(self, x):
        # x 形状: (batch_size, input_len)
        trend_part = self.trend_linear(x)  # 计算趋势分量 (batch_size, output_len)
        
        # 使用广播机制将趋势部分扩展到输入的维度
        trend_part_expanded = self.trend_to_x(trend_part)
        
        # 计算残差分量
        residual_part = self.residual_linear(x - trend_part_expanded)  # 计算残差 (batch_size, output_len)
        
        return trend_part + residual_part  # 合成最终预测结果

# Historical Inertia 模型
class HistoricalInertia(nn.Module):
    def __init__(self, input_len=672, output_len=96):
        super(HistoricalInertia, self).__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.linear = nn.Linear(output_len, output_len)

    def forward(self, x):
        # x 形状: (batch_size, input_len) = (batch_size, 672)
        # 直接使用 x 的最后 96 个时间步作为预测
        x = x[:, -self.output_len:]
        x = x + self.linear(x)
        return x # 预测未来的 96 个时间步

# 定义 U-Net 结构
class SimpleUNet(nn.Module):
    def __init__(self, input_dim=672, output_dim=96):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)  # 输出维度为 96
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# DDPM 训练框架
class DDPM(nn.Module):
    def __init__(self, timesteps=1000, input_dim=672, output_dim=96):
        super(DDPM, self).__init__()
        self.timesteps = timesteps
        self.unet = SimpleUNet(input_dim, output_dim)

        # 预定义扩散过程的 beta 系列
        self.beta = torch.linspace(0.0001, 0.02, timesteps).to('cuda')  # 线性增长的 beta
        self.alpha = 1.0 - self.beta
        self.alpha = self.alpha.to('cuda')
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)  # 累乘 alpha 以计算 α̅_t
        self.linear = nn.Linear(input_dim, output_dim)

    def forward_diffusion(self, x, t):
        """前向扩散过程"""
        noise = torch.randn_like(x).to(x.device)  # 生成与 x 相同形状的随机噪声
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1)  # 适应形状
        sqrt_beta = torch.sqrt(self.beta[t]).view(-1, 1)
        x_t = sqrt_alpha_hat * x + sqrt_beta * noise  # 扩散公式
        return x_t, noise

    def reverse_process(self, x_t, t):
        """逆向扩散过程，预测 y"""
        predicted_noise = self.unet(x_t)  # 预测噪声
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t]).view(-1, 1)
        sqrt_beta = torch.sqrt(self.beta[t]).view(-1, 1)
        x_1 = self.linear(x_t)
        x_0 = (x_1 - sqrt_beta * predicted_noise) / sqrt_alpha_hat  # 逆扩散公式
        return x_0  # 预测的 y

# 预测函数
def predict(ddpm, x, device='cuda'):
    """给定 x 预测 y"""
    ddpm.eval()
    x = x.to(device)

    with torch.no_grad():
        batch_size = x.shape[0]
        t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
        x_t, _ = ddpm.forward_diffusion(x, t)  # 添加噪声
        y_pred = ddpm.reverse_process(x_t, t)  # 预测 y

    return y_pred  # 转回 numpy

# 位置编码模块
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].unsqueeze(0)  # 修正位置编码添加方式
        return self.dropout(x)

class TimeSeriesTransformer(nn.Module):
    def __init__(self, 
                 input_size: int = 1,   # 每个时间步的特征维度
                 output_size: int = 1,  # 每个预测时间步的输出维度
                 d_model: int = 64,
                 nhead: int = 8,
                 num_encoder_layers: int = 3,
                 num_decoder_layers: int = 3,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1):
        super().__init__()
        
        # 输入输出参数
        self.input_size = input_size
        self.output_size = output_size
        self.d_model = d_model
        
        # 编码器部分（关键修改）
        self.encoder_embedding = nn.Linear(input_size, d_model)
        #self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # 解码器部分
        self.decoder_embedding = nn.Linear(output_size, d_model)
        #self.pos_decoder = PositionalEncoding(d_model, dropout)
        
        # Transformer核心（添加batch_first参数）
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True  # 使用batch_first格式
        )
        
        # 输出层
        self.fc_out = nn.Linear(d_model, output_size)
        
        # 初始化参数
        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.encoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder_embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        """
        Args:
            src: 输入序列 [batch_size, src_seq_len] -> 修改后的形状处理
        Returns:
            output: 预测序列 [batch_size, tgt_seq_len]
        """
        # 调整输入形状（关键修改）
        batch_size, src_seq_len = src.size()
        src = src.view(batch_size, src_seq_len, self.input_size)  # [32, 672] -> [32, 672, 1]
        
        # 编码器嵌入
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        #src_emb = self.pos_encoder(src_emb)  # [batch_size, seq_len, d_model]
        
        # 解码器输入（初始化为全零）
        tgt_seq_len = 96
        tgt = torch.zeros(batch_size, tgt_seq_len, self.output_size, device=src.device)
        
        # 解码器嵌入
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        #tgt_emb = self.pos_decoder(tgt_emb)  # [batch_size, tgt_seq_len, d_model]
        
        # 生成注意力mask
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        
        # Transformer前向传播（保持batch_first格式）
        output = self.transformer(
            src_emb, 
            tgt_emb,
            tgt_mask=tgt_mask
        )
        
        # 最终输出（调整形状）
        output = self.fc_out(output)  # [batch_size, tgt_seq_len, 1]
        return output.squeeze(-1)     # [batch_size, tgt_seq_len]