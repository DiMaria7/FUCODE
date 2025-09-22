import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from dataset.data_utils import data_set, separate_data, split_proxy
import numpy as np
import os
import pickle as pkl
import pandas as pd
import tqdm
import random
from transformers import BertTokenizer

MAX_VOCAB_SIZE = 10000
UNK, PAD = '<UNK>', '<PAD>'

# Allocate data to users


def data_init(FL_params):
    # 根据设备类型设置DataLoader的额外参数
    # 如果设备是cuda，设置num_workers为0，pin_memory为True，以优化数据加载性能
    # 否则，设置为空字典
    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {}
    # 初始化存储训练数据特征的列表
    dataset_x = []
    # 初始化存储注意力掩码的列表（这里未使用，可能后续扩展用）
    dataset_at = []
    # 初始化存储训练数据标签的列表
    dataset_y = []

    # 调用data_set函数，根据FL_params中的数据名称获取训练集和测试集
    trainset, testset = data_set(FL_params.data_name)

    # 创建测试集的数据加载器，设置批量大小为test_batch_size，打乱数据，使用2个工作进程
    # 并传入之前设置的额外参数kwargs
    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs)
    # 创建训练集的数据加载器，设置批量大小为local_batch_size，打乱数据，使用2个工作进程
    # 并传入之前设置的额外参数kwargs
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2, **kwargs)

    # 遍历训练数据加载器中的每个批次
    for train_data in train_loader:
        # 解包每个批次的数据，得到特征和标签
        x_train, y_train = train_data
        # 将特征数据从张量转换为numpy数组，并添加到dataset_x列表中
        dataset_x.extend(x_train.cpu().detach().numpy())
        # 将标签数据从张量转换为numpy数组，并添加到dataset_y列表中
        dataset_y.extend(y_train.cpu().detach().numpy())
    # 如果遗忘范式是按客户端进行
    if FL_params.forget_paradigm == 'client':
        # 遍历测试数据加载器中的每个批次
        for test_data in test_loader:
            # 将测试集的特征数据从张量转换为numpy数组，并添加到dataset_x列表中
            x_test, y_test = test_data
            # 将测试集的特征数据从张量转换为numpy数组，并添加到dataset_x列表中
            dataset_x.extend(x_test.cpu().detach().numpy())
            # 将测试集的标签数据从张量转换为numpy数组，并添加到dataset_y列表中
            dataset_y.extend(y_test.cpu().detach().numpy())

    # 将dataset_x列表转换为numpy数组
    dataset_x = np.array(dataset_x)
    # 将dataset_y列表转换为numpy数组
    dataset_y = np.array(dataset_y)

    # 调用separate_data函数，将数据分割给不同的客户端
    # 参数包括数据、客户端数量、类别数量、其他参数、是否为非独立同分布、是否平衡数据、分区方式和每个客户端的类别数
    # 并返回分割后的数据集
    X, y, statistic = separate_data((dataset_x, dataset_y), FL_params.num_user, FL_params.num_classes, FL_params,
                                    FL_params.niid, FL_params.balance, FL_params.partition, class_per_client=2)

    # 调用split_proxy函数，将分割后的数据进一步划分为客户端加载器、测试加载器、代理客户端加载器和代理测试加载器
    client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = split_proxy(X, y, FL_params)
    # 计算每个客户端的数据大小，并存储在FL_params的datasize_ls属性中
    FL_params.datasize_ls = [len(k) for k in X]
    # 如果遗忘范式是按客户端进行
    if FL_params.forget_paradigm == 'client':
        # 保持测试加载器和代理测试加载器不变
        test_loaders = test_loaders
        proxy_test_loaders = proxy_test_loaders
    else:
        # 初始化存储代理测试集特征的列表
        proxy_test_x = []
        # 初始化存储代理测试集标签的列表
        proxy_test_y = []
        # 遍历每个客户端的测试加载器
        for i in range(FL_params.num_user):
            # 遍历每个客户端测试加载器中的每个批次
            for x, y in test_loaders[i]:
                 # 将特征数据添加到proxy_test_x列表中
                proxy_test_x.append(x)
                # 将标签数据添加到proxy_test_y列表中
                proxy_test_y.append(y)
         # 将proxy_test_x列表中的张量拼接成一个大的张量，并转换为numpy数组
        proxy_test_x = torch.cat(proxy_test_x).numpy()
        proxy_test_y = torch.cat(proxy_test_y).numpy()
        # 创建代理测试集的数据加载器，设置批量大小为test_batch_size，打乱数据
        proxy_test_loader = DataLoader(TensorDataset(torch.tensor(proxy_test_x), torch.tensor(proxy_test_y)), batch_size=FL_params.test_batch_size, shuffle=True)
        # 复制代理测试集的数据加载器，使其数量与客户端数量相同
        proxy_test_loaders = [proxy_test_loader for _ in range(FL_params.num_user)]
    # 返回客户端加载器、测试加载器、代理客户端加载器和代理测试加载器
    return client_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders

def cross_data_init(FL_params):

    kwargs = {'num_workers': 0, 'pin_memory': True} if FL_params.device == 'cuda' else {}
    dataset_x = []
    dataset_y = []

    trainset, testset = data_set(FL_params.data_name)

    test_loader = DataLoader(testset, batch_size=FL_params.test_batch_size, shuffle=True, num_workers=2, **kwargs)
    train_loader = DataLoader(trainset, batch_size=FL_params.local_batch_size, shuffle=True, num_workers=2,
                              **kwargs)

    for train_data in train_loader:
        x_train, y_train = train_data
        dataset_x.extend(x_train.cpu().detach().numpy())
        dataset_y.extend(y_train.cpu().detach().numpy())
    if FL_params.forget_paradigm == 'client':
        for test_data in test_loader:
            x_test, y_test = test_data
            dataset_x.extend(x_test.cpu().detach().numpy())
            dataset_y.extend(y_test.cpu().detach().numpy())

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    class_num = int(FL_params.num_classes/FL_params.num_user)
    X = []
    y = []
    idx_ls = []
    for user in range(FL_params.num_user):
        idx = []
        for i in range(class_num):
            item = user*class_num + i
            indices = [idx for idx, label in enumerate(dataset_y) if label == item]
            idx.extend(indices)
        idx_ls.append(idx)
    corss_idx = idx_ls[0][:int(len(idx_ls[0])*0.01)]
    idx_ls[0] = idx_ls[0][int(len(idx_ls[0])*0.01):]
    idx_ls[1] = corss_idx + idx_ls[1]
    remain_idx = []
    for idx in range(1, FL_params.num_user):
        remain_idx.extend(idx_ls[idx])
    random.shuffle(remain_idx)
    sublist_size = len(remain_idx) // (FL_params.num_user-len(FL_params.forget_client_idx))
    remainder = len(remain_idx) % (FL_params.num_user-len(FL_params.forget_client_idx))

    sublists = [remain_idx[i * sublist_size + min(i, remainder):(i + 1) * sublist_size + min(i + 1, remainder)] for i in
                range(9)]

    for idx in range(1, FL_params.num_user):
        idx_ls[idx] = sublists[idx-1]

    for user in range(FL_params.num_user):
        X.append(dataset_x[idx_ls[user]])
        y.append(dataset_y[idx_ls[user]])

    for i in range(FL_params.num_user):
        print('client {} data size {} lable {}'.format(i, len(X[i]),np.unique(y[i])))

    client_loaders, test_loaders, proxy_loader = split_proxy(X, y, FL_params)
    FL_params.datasize_ls = [len(k) for k in X]
    if FL_params.forget_paradigm == 'client':
        test_loaders = test_loaders
    else:
        test_loaders = [test_loader for _ in range(FL_params.num_user)]

    return client_loaders, test_loaders, proxy_loader
