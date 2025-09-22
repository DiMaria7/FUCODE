import time
import math
import pandas as pd
import torch

from models.Model_base import *
from models import LeNet_FashionMNIST, CNN_Cifar10, CNN_Cifar100, Model_adults, Model_purchase
from utils import init_network, test_class_forget, test_client_forget
from dataset.data_utils import *
from algs.fl_base import Base
import torch.optim as optim
import copy
import logging
import matplotlib.pyplot as plt
from utils import *
import random
from models.Model_base import *

class FUSED(Base):
    def __init__(self, args):
        super(FUSED, self).__init__(args)
        self.args = args
        self.log_dir = f"logs/fused_{self.args.data_name}_{self.args.alpha}"
        self.param_change_dict = {}
        self.param_size = {}

    # 执行正常的联邦学习流程，不涉及遗忘机制
    # 要步骤包括：按轮次选择客户端、在客户端本地训练模型、聚合客户端模型得到全局模型、
    # 测试模型性能（根据不同遗忘范式）、记录参数变化和训练结果，并最终保存模型和结果文件。
    def train_normal(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []            # 初始化存储检查点的列表
        result_list = []               # 初始化存储测试结果的列表
        param_list = []                # 初始化存储参数变化的列表
        # 遍历全局模型的参数，初始化参数变化字典和参数大小字典
        for name, param in global_model.named_parameters():
            # print(name)
            self.param_change_dict[name] = 0        # 记录参数变化量
            self.param_size[name] = 0               # 记录参数大小

        # 遍历全局训练轮次
        for epoch in range(self.args.global_epoch):
            # 随机选择部分客户端（数量为总客户端数 * 采样比例）
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user*self.args.fraction), replace=False))
            # 获取选中客户端的数据加载器
            select_client_loaders = [client_all_loaders[idx] for idx in selected_clients]
            # 执行一轮全局训练，返回客户端训练后的模型
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)

            # 使用联邦平均（FedAvg）聚合客户端模型，得到新的全局模型
            global_model = self.fedavg(client_models)

            # 根据不同的遗忘范式，执行对应的测试并记录结果
            if self.args.forget_paradigm == 'sample':
            
                # 测试样本级遗忘效果（精确率、零准确率、平均测试准确率等）
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model,
                                                                                              self.args, test_loaders)
                print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero,
                                                                                 avg_test_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'client':
                # 测试客户端级遗忘效果（遗忘客户端准确率、保留客户端准确率）
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)
            
            elif self.args.forget_paradigm == 'class':
                # 测试类别级遗忘效果（遗忘类别准确率、保留类别准确率）
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args, test_loaders)
                print('Epoch={}, avg_f_acc={}, avg_r_acc={}'.format(epoch, avg_f_acc, avg_r_acc))
                result_list.extend(test_result_ls)

            # 如果使用FUSED范式，记录参数变化
            if self.args.paradigm == 'fused':
                # 提取参数变化值并转换为浮点数
                diff_ls = list(self.param_change_dict.values())
                name = list(self.param_change_dict.keys())
                diff_ls_ = [float(i) for i in diff_ls]
                param_list.append(diff_ls_)                  # 存储当前轮次的参数变化
                # diff_ls_.append(list(self.param_size.values()))
        
        # 将参数变化记录保存为CSV文件
        df = pd.DataFrame(param_list, columns=name)
        df.to_csv('./results/param_change_{}_distri_{}.csv'.format(self.args.data_name, self.args.alpha))

        # 保存全局模型的状态字典
        torch.save(global_model.state_dict(), 'save_model/global_model_{}.pth'.format(self.args.data_name))
        
        # 根据不同的遗忘范式，将测试结果保存为CSV文件
        if self.args.forget_paradigm == 'sample':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        elif self.args.forget_paradigm == 'client':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        elif self.args.forget_paradigm == 'class':
            df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        if self.args.save_normal_result:
            df.to_csv('./results/Acc_loss_fl_{}_data_{}_distri_{}.csv'.format(self.args.forget_paradigm, self.args.data_name, self.args.alpha))

        # 返回训练完成的全局模型和客户端模型
        return global_model, client_models

    def forget_client_train(self, global_model, client_all_loaders, test_loaders):
        # 加载预训练的全局模型权重（正常训练阶段保存的模型）
        global_model.load_state_dict(torch.load('save_model/global_model_{}.pth'.format(self.args.data_name)))
        # 在遗忘开始前，测试当前全局模型在遗忘客户端和保留客户端上的初始性能
        avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, 1, global_model, self.args,
                                                                  test_loaders)
        # 打印初始性能日志（xxxx表示尚未开始遗忘训练的 epoch）
        print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format('xxxx', avg_r_acc,
                                                                                 avg_f_acc))

        # 初始化 FUSED 模型（基于 Lora 适配器，冻结原模型权重，仅训练适配器参数）
        fused_model = Lora(self.args, global_model)
        # 保存初始 FUSED 模型权重
        torch.save(fused_model.state_dict(), 'save_model/global_fusedmodel_{}.pth'.format(self.args.data_name))

        # 初始化变量：存储检查点、结果列表、训练耗时
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        # 开始遗忘训练的全局轮次迭代
        for epoch in range(self.args.global_epoch):
            # 将模型设为训练模式
            fused_model.train()

            # 选择参与训练的客户端：排除需要遗忘的客户端（只使用保留客户端的数据）
            selected_clients = [i for i in range(self.args.num_user) if i not in self.args.forget_client_idx]
            # 获取选中客户端的数据加载器（可能包含部分样本，由 cut_sample 参数控制）
            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)

            # 记录本轮训练开始时间
            std_time = time.time()
            # 执行一轮全局训练：每个选中的客户端本地训练模型，返回客户端模型列表
            client_models = self.global_train_once(epoch, fused_model, select_client_loaders, test_loaders, self.args, checkpoints_ls)
            # 记录本轮训练结束时间，累加耗时
            end_time = time.time()
            # 使用 FedAvg 算法聚合客户端模型，得到平均模型
            avg_model = self.fedavg(client_models)
            consume_time += end_time - std_time
            # 将聚合后的模型权重加载到 FUSED 模型中
            fused_model.load_state_dict(avg_model.state_dict())

            # 将模型设为评估模式
            fused_model.eval()

            # 测试当前 FUSED 模型在遗忘客户端和保留客户端上的性能
            avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, fused_model, self.args,
                                                                      test_loaders)

            # 将测试结果存入列表
            result_list.extend(test_result_ls)

            # 打印本轮训练后的性能日志
            print('FUSED-epoch-{}-client forget, Avg_r_acc: {}, Avg_f_acc: {}'.format(epoch, avg_r_acc,
                                                                                    avg_f_acc))

        # 将所有轮次的测试结果整理为 DataFrame（包含轮次、客户端ID、类别ID、样本数、准确率、损失）
        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss'])
        # 记录总训练耗时
        df['Comsume_time'] = consume_time

        # 根据是否使用部分数据（cut_sample 参数），将结果保存到不同的 CSV 文件
        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv('./results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm,
                                                                                              self.args.forget_paradigm,
                                                                                              self.args.data_name,
                                                                                              self.args.alpha,
                                                                                              len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(
                        self.args.forget_paradigm,
                        self.args.forget_paradigm,
                        self.args.data_name,
                        self.args.alpha,
                        len(self.args.forget_class_idx), self.args.cut_sample))

        # 返回训练完成的 FUSED 模型（已完成对指定客户端的遗忘）
        return fused_model

    def forget_class(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))

            select_client_loaders = select_part_sample(self.args, client_all_loaders, selected_clients)
            std_time = time.time()

            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time
            avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, Remember Test Acc={}, Forget Test Acc={}'.format(epoch, avg_r_acc, avg_f_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss'])
        df['Comsume_time'] = consume_time

        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                     self.args.forget_paradigm,
                                                                                     self.args.data_name,
                                                                                     self.args.alpha,
                                                                                     len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def forget_sample(self, global_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        result_list = []
        consume_time = 0

        fused_model = Lora(self.args, global_model)
        for epoch in range(self.args.global_epoch):
            fused_model.train()
            selected_clients = list(np.random.choice(range(self.args.num_user), size=int(self.args.num_user * self.args.fraction), replace=False))# 将需要遗忘的客户端排除在外

            self.select_forget_idx = list()
            select_client_loaders = list()
            record = -1
            for idx in selected_clients:
                select_client_loaders.append(client_all_loaders[idx])
                record += 1
                if idx in self.args.forget_client_idx:
                    self.select_forget_idx.append(record)
            std_time = time.time()
            client_models = self.global_train_once(epoch, fused_model,  select_client_loaders, test_loaders, self.args, checkpoints_ls)
            end_time = time.time()
            fused_model = self.fedavg(client_models)
            consume_time += end_time-std_time

            avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, fused_model, self.args, test_loaders)
            result_list.extend(test_result_ls)
            print('Epoch={}, jingdu={}, acc_zero={}, avg_test_acc={}'.format(epoch, avg_jingdu, avg_acc_zero, avg_test_acc))

        df = pd.DataFrame(result_list, columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc'])
        df['Comsume_time'] = consume_time
        if self.args.cut_sample == 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}.csv'.format(self.args.forget_paradigm, self.args.forget_paradigm, self.args.data_name, self.args.alpha, len(self.args.forget_class_idx)))
        elif self.args.cut_sample < 1.0:
            if self.args.save_normal_result:
                df.to_csv(
                    './results/{}/Acc_loss_fused_{}_data_{}_distri_{}_fnum_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                        self.args.forget_paradigm,
                                                                                        self.args.data_name,
                                                                                        self.args.alpha,
                                                                                        len(self.args.forget_class_idx), self.args.cut_sample))

        return fused_model

    def relearn_unlearning_knowledge(self, unlearning_model, client_all_loaders, test_loaders):
        checkpoints_ls = []
        all_global_models = list()
        all_client_models = list()
        global_model = unlearning_model
        result_list = []

        all_global_models.append(global_model)
        std_time = time.time()
        for epoch in range(self.args.global_epoch):
            if self.args.forget_paradigm == 'client':
                select_client_loaders = list()
                for idx in self.args.forget_client_idx:
                    select_client_loaders.append(client_all_loaders[idx])
            elif self.args.forget_paradigm == 'class':
                select_client_loaders = list()
                client_loaders = select_forget_class(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            elif self.args.forget_paradigm == 'sample':
                select_client_loaders = list()
                client_loaders = select_forget_sample(self.args, copy.deepcopy(client_all_loaders))
                for v in client_loaders:
                    if v is not None:
                        select_client_loaders.append(v)
            client_models = self.global_train_once(epoch, global_model, select_client_loaders, test_loaders, self.args,
                                                   checkpoints_ls)

            all_client_models += client_models
            global_model = self.fedavg(client_models)
            all_global_models.append(copy.deepcopy(global_model).to('cpu'))
            end_time = time.time()

            consume_time = end_time - std_time

            if self.args.forget_paradigm == 'client':
                avg_f_acc, avg_r_acc, test_result_ls = test_client_forget(self, epoch, global_model, self.args,
                                                                          test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Label_num', 'Test_acc', 'Test_loss',
                                           'Comsume_time'])
            elif self.args.forget_paradigm == 'class':
                avg_f_acc, avg_r_acc, test_result_ls = test_class_forget(self, epoch, global_model, self.args,
                                                                         test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Class_id', 'Test_acc', 'Test_loss', 'Comsume_time'])
            elif self.args.forget_paradigm == 'sample':
                avg_jingdu, avg_acc_zero, avg_test_acc, test_result_ls = test_backdoor_forget(self, epoch, global_model, self.args, test_loaders)
                for item in test_result_ls:
                    item.append(consume_time)
                result_list.extend(test_result_ls)
                df = pd.DataFrame(result_list,
                                  columns=['Epoch', 'Client_id', 'Jingdu', 'Acc_zero', 'Test_acc', 'Comsume_time'])

            global_model.to('cpu')

            print("Relearn Round = {}".format(epoch))
        
        if self.args.cut_sample == 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm), index=False)
        elif self.args.cut_sample < 1.0:
            df.to_csv('./results/{}/relearn_data_{}_distri_{}_fnum_{}_algo_{}_partdata_{}.csv'.format(self.args.forget_paradigm,
                                                                                          self.args.data_name,
                                                                                      self.args.alpha,
                                                                                      len(self.args.forget_class_idx),
                                                                                      self.args.paradigm, self.args.cut_sample), index=False)
        return