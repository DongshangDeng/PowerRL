import csv
import torch
from torch import nn
import numpy as np
import pandas as pd
import torch.optim as optim
from tqdm import tqdm
from utils.network1 import CustomGRU, CustomMLP
from utils.dataset import create_dataloader
from utils.criterion import RMSE, SMAPE, JSD, MAPE

num_nodes = 50  # 节点数
input_dim = 96
hidden_dim = 64
feature_dim = 32  # 特征维度
num_features = 4  # 特征数量
num_heads = 4
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

a = np.load("./data/adjacency_matrix.npy")
b = np.load("./data/connectivity_matrix.npy")
c = np.load("./data/distance_matrix.npy")
d = np.load("./data/similarity_matrix.npy")
adj_matrix = torch.tensor(a[:num_nodes, :num_nodes]).to(device)
con_matrix = torch.tensor(b[:num_nodes, :num_nodes]).to(device)
dis_matrix = torch.tensor(c[:num_nodes, :num_nodes]).to(device)
sim_matrix = torch.tensor(d[:num_nodes, :num_nodes]).to(device)
matrix = (adj_matrix, con_matrix, dis_matrix, sim_matrix)


tou = np.load("./data/TOU.npy")
tou = torch.tensor(tou, dtype=torch.float32).to(device)
length = tou.shape[0]


window_height = 768
num_columns = 5
batch_size=32
df = pd.read_excel(f"./data/feature2_{num_nodes}.xlsx").to_numpy()
data_tensor = torch.tensor(df, dtype=torch.float32).to(device)
train_length = 96*80
train_samples = 5000
test_samples = 500
train_loader, test_loader = create_dataloader(data_tensor, train_length, train_samples, test_samples, window_height=window_height, num_columns=num_columns, batch_size=batch_size)



model = CustomGRU(input_dim, hidden_dim, matrix, tou, num_nodes=num_nodes).to(device)
criterion = nn.L1Loss()  # MAE损失
maeloss = nn.L1Loss()
rmseloss = RMSE()
smapeloss = SMAPE()
jsdloss = JSD()
mapeloss = MAPE()

#mse_loss_soft = nn.MSELoss()
lr=2e-4
optimizer = optim.Adam(model.parameters(), lr=lr)
alpha = 0.01

method = 'STGRU'
csv_filename = f"./results/training_results_{method}_{feature_dim}_{lr}_mae__{num_nodes}.csv"
header = ['epoch', 'train_MAE_loss', 'RMSE_loss', 'SMAPE_loss', 'JSD_loss', 'MAPE_loss', 'test_MAE_loss', 'RMSE', 'MAPE', 'JSD', 'MAPE']
with open(csv_filename, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(header)


num_epochs = 200
for epoch in tqdm(range(num_epochs), desc="Epochs"):
    model.train()  # 设置模型为训练模式
    losses1, losses2, losses3, losses4, losses5 = 0.0, 0.0, 0.0, 0.0, 0.0
    for inputs, labels in train_loader:
        # 清零优化器的梯度
        optimizer.zero_grad()
        inputs, labels = inputs.to(device), labels.to(device)
        # 前向传播
        student_outputs = model(inputs)  # 输出是形状为 [batch_size, 96] 的预测值
        
        # 计算损失
        loss1 = criterion(student_outputs, labels)
        loss2 = rmseloss(student_outputs, labels)
        loss3 = smapeloss(student_outputs, labels)
        loss4 = jsdloss(student_outputs, labels)
        loss5 = mapeloss(student_outputs, labels)
        
        # 总损失
        # 反向传播
        loss1.backward()
        
        # 更新参数
        optimizer.step()
        
        # 记录损失
        losses1 += loss1.item()
        losses2 += loss2.item()
        losses3 += loss3.item()
        losses4 += loss4.item()
        losses5 += loss5.item()
    
    # 打印每个epoch的损失
    losses1_avg = losses1 / len(train_loader)
    losses2_avg = losses2 / len(train_loader)
    losses3_avg = losses3 / len(train_loader)
    losses4_avg = losses4 / len(train_loader)
    losses5_avg = losses5 / len(train_loader)

    model.eval()  # 设置模型为评估模式
    #test_loss = 0.0
    mae_loss = 0.0
    rmse_loss = 0.0
    smape_loss = 0.0
    jsd_loss = 0.0
    mape_loss = 0.0
    with torch.no_grad():  # 禁用梯度计算以节省内存
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # 获取预测结果
            
            # 计算MAE损失
            #loss = criterion(outputs, labels)
            #test_loss += loss.item()
            mae_loss += maeloss(outputs, labels).item()
            rmse_loss += rmseloss(outputs, labels).item()
            smape_loss += smapeloss(outputs, labels).item()
            jsd_loss += jsdloss(outputs, labels).item()
            mape_loss += mapeloss(outputs, labels).item()
        
    
    # 打印每个epoch的测试损失和精度
    len_test_loader = len(test_loader)
    #test_loss = test_loss / len_test_loader
    mae_loss = mae_loss / len_test_loader
    rmse_loss = rmse_loss / len_test_loader
    smape_loss = smape_loss / len_test_loader
    jsd_loss = jsd_loss / len_test_loader
    mape_loss = mape_loss / len_test_loader
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {losses1_avg:.4f}-{losses2_avg:.4f}-{losses3_avg:.4f}-{losses4_avg:.4f}-{losses5_avg:.4f}, Test Loss: {mae_loss:.4f}-{rmse_loss:.4f}-{smape_loss:.4f}-{jsd_loss:.4f}-{mape_loss:.4f}")

    with open(csv_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([epoch+1, losses1_avg, losses2_avg, losses3_avg, losses4_avg, losses5_avg, rmse_loss, smape_loss, jsd_loss, mape_loss])

model_path = f"./model/mygru_{num_epochs}_{lr}_{feature_dim}_mae_{num_nodes}.pth"
torch.save(model.state_dict(), model_path)