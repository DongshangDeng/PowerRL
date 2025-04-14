import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def create_random_dataset(data_tensor, window_height, num_columns, num_samples):
    total_rows = data_tensor.shape[0]
    assert (num_columns <= data_tensor.shape[1])
    windows = []
    shuffled_sequence = torch.randperm(total_rows - window_height + 1)
    
    for i in range(num_samples):
        # 随机选择窗口的起始行位置
        start_row = shuffled_sequence[i]
        
        # 提取一个窗口
        window = data_tensor[start_row:start_row + window_height, :num_columns]
        
        # 将窗口添加到结果列表
        window = window.reshape([8, int(window_height/8), num_columns])
        #chunks = torch.chunk(window, 8, dim=0)
        windows.append(window)

    dataset = torch.stack(windows, dim=0)
    
    return dataset

def create_dataloader(data, train_length, train_samples, test_samples, window_height=768, num_columns=5, batch_size=32):
    train_tensor = data[:train_length, :]
    test_tensor = data[train_length:, :]
    trainset = create_random_dataset(train_tensor, window_height, num_columns, train_samples)
    testset = create_random_dataset(test_tensor, window_height, num_columns, test_samples)
    train_x = trainset[:, :-1, :, :]
    train_y = trainset[:, -1, :, -1].squeeze()
    train_dataset = TensorDataset(train_x, train_y)  # 训练集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_x = testset[:, :-1, :, :]
    test_y = testset[:, -1, :, -1].squeeze()
    test_dataset = TensorDataset(test_x, test_y)  # 训练集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def create_method_dataloader(data, train_length, train_samples, test_samples, window_height=768, num_columns=5, batch_size=32):
    train_tensor = data[:train_length, :]
    test_tensor = data[train_length:, :]
    trainset = create_random_dataset(train_tensor, window_height, num_columns, train_samples)
    testset = create_random_dataset(test_tensor, window_height, num_columns, test_samples)
    train_x = trainset[:, :-1, :, -1].squeeze()
    train_x = train_x.view(train_x.shape[0], -1)
    train_y = trainset[:, -1, :, -1].squeeze()
    train_dataset = TensorDataset(train_x, train_y)  # 训练集
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_x = testset[:, :-1, :, -1].squeeze()
    test_x = test_x.view(test_x.shape[0], -1)
    test_y = testset[:, -1, :, -1].squeeze()
    test_dataset = TensorDataset(test_x, test_y)  # 训练集
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

