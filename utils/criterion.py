import torch
import torch.nn as nn
import torch.nn.functional as F

class RMSE(nn.Module):
    def __init__(self):
        super(RMSE, self).__init__()

    def forward(self, y_true, y_pred):
        # 计算均方误差（MSE）
        mse = torch.mean((y_true - y_pred) ** 2)
        # 返回均方根误差（RMSE）
        rmse = torch.sqrt(mse)
        return rmse
    

class SMAPE(nn.Module):
    def __init__(self):
        super(SMAPE, self).__init__()

    def forward(self, y_true, y_pred):
        # 防止除以0的情况，避免数值不稳定
        epsilon = 1e-8
        
        # 计算sMAPE
        diff = torch.abs(y_true - y_pred)
        denominator = (torch.abs(y_true) + torch.abs(y_pred)) / 2.0
        loss = torch.mean(diff / (denominator + epsilon))  # 加上epsilon防止分母为0

        return loss * 100  # 返回百分比

class JSD(nn.Module):
    def __init__(self):
        super(JSD, self).__init__()

    def forward(self, p, q):
        p = p / p.sum(dim=-1, keepdim=True)  # 每行元素归一化
        q = q / q.sum(dim=-1, keepdim=True)  # 每行元素归一化
        # 防止对数运算中的 NaN 或 -inf 错误，确保概率值在 (0, 1) 范围内
        p = p.clamp(min=1e-10, max=1.0)
        q = q.clamp(min=1e-10, max=1.0)
        
        # 计算M = (p + q) / 2
        m = 0.5 * (p + q)
        
        # 计算 D_KL(P || M) 和 D_KL(Q || M)
        kl_pm = torch.sum(p * torch.log(p / m), dim=-1)
        kl_qm = torch.sum(q * torch.log(q / m), dim=-1)
        
        # 返回 JSD = 0.5 * (D_KL(P || M) + D_KL(Q || M))
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd.mean()  # 取均值以适应批次大小
    

class MMDLoss(nn.Module):
    def __init__(self, kernel='rbf', sigma=1.0):
        super(MMDLoss, self).__init__()
        self.kernel = kernel
        self.sigma = sigma

    def rbf_kernel(self, X, Y=None):
        # 计算 RBF 核
        if Y is None:
            Y = X
        xx = torch.sum(X**2, dim=1, keepdim=True)
        yy = torch.sum(Y**2, dim=1, keepdim=True)
        dist_matrix = xx + yy.T - 2 * torch.matmul(X, Y.T)
        return torch.exp(-dist_matrix / (2 * self.sigma**2))

    def forward(self, x, y):
        # 计算 MMD 损失
        # x 和 y 是两个批次的样本，形状为 [N, D]，N 为样本数，D 为特征维度
        n = x.size(0)
        m = y.size(0)

        # 计算每对样本之间的 RBF 核
        Kxx = self.rbf_kernel(x, x)
        Kyy = self.rbf_kernel(y, y)
        Kxy = self.rbf_kernel(x, y)

        # MMD 损失计算公式
        loss = torch.mean(Kxx) + torch.mean(Kyy) - 2 * torch.mean(Kxy)
        return loss
    
class MAPE(nn.Module):
    def __init__(self):
        super(MAPE, self).__init__()

    def forward(self, y_pred, y_true):
        # 计算绝对百分比误差（MAPE）
        # 防止除以零，使用一个小的epsilon
        epsilon = 1e-8
        diff = torch.abs(y_true - y_pred)
        abs_percentage_error = torch.mean(diff / torch.abs(y_true + epsilon))
        # 计算平均绝对百分比误差
        return abs_percentage_error * 100
