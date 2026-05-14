
import argparse
import copy

from dataset.generate_data import data_init, cross_data_init
import torch

from algs import fused_unlearning, fl_base
from utils import *
import random
import numpy as np

def get_args():
    parser = argparse.ArgumentParser(description='Chinese Text Classification')
    # TODO
    parser.add_argument('--model', type=str, required=False, default='LeNet_FashionMNIST', help= 'choose a model: LeNet_FashionMNIST,CNN_Cifar10,CNN_Cifar100')
    parser.add_argument('--data_name', type=str, required=False, default='fashionmnist', help= 'choose: mnist, fashionmnist, cifar10, cifar100')
    parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
    parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
    parser.add_argument('--distribution', default=True, type=bool, help='True means iid, while False means non-iid')
    parser.add_argument('--train_with_test', default=True, type=bool, help='')
    parser.add_argument('--temperature', default=0.5, type=float, help='the temperature for distillation loss')
    parser.add_argument('--max_checkpoints', default=3, type=int)
    # ======================unlearning setting==========================
    # TODO
    parser.add_argument('--forget_paradigm', default='class', type=str, help='choose from client or class')
    parser.add_argument('--paradigm', default='fused', type=str,
                        help='choose the training paradigm:fused, federaser, retrain, infocom22, exactfun, fl, eraseclient')
    parser.add_argument('--forget_client_idx', type=list, default=[0])
    parser.add_argument('--forget_class_idx', type=list, default=[0])
    parser.add_argument('--if_retrain', default=False, type=bool, help='')
    parser.add_argument('--if_unlearning', default=False, type=bool, help='')
    parser.add_argument('--baizhanting', default=True, type=bool, help='')
    parser.add_argument('--backdoor', default=False, type=bool, help='')
    parser.add_argument('--backdoor_frac', default=0.2, type=float, help='')
    # TODO
    parser.add_argument('--MIT', default=True, type=bool, help='whether to use membership inference attack')
    parser.add_argument('--n_shadow', default=5, type=int, help='the number of shadow model')
    parser.add_argument('--cut_sample', default=1.0, type=float, help='using part of the training data')
    parser.add_argument('--relearn', default=True, type=bool, help='whether to relearn the unlearned knowledge')
    parser.add_argument('--save_normal_result', default=True, type=bool, help='whether to save the normal result')
    # ======================batchsize setting===========================
    parser.add_argument('--local_batch_size', default=64, type=int)
    parser.add_argument('--test_batch_size', default=64, type=int)
    # ======================training epoch===========================
    # TODO
    parser.add_argument('--global_epoch', default=2, type=int)
    parser.add_argument('--local_epoch', default=1, type=int)
    parser.add_argument('--distill_epoch', default=10, type=int)
    parser.add_argument('--distill_pretrain_epoch', default=2, type=int)
    parser.add_argument('--fraction', default=1.0, type=float, help='the fraction of training data')
    parser.add_argument('--num_user', default=50, type=int)
    # ======================data process============================
    parser.add_argument('--niid', default=True, type=bool, help='')
    parser.add_argument('--balance', default=True, type=bool, help='')
    parser.add_argument('--partition', default='dir', type=str, help='choose from pat or dir')
    parser.add_argument('--alpha', default=1.0, type=float, help='for Dirichlet distribution')
    parser.add_argument('--proxy_frac', default=0.2, type=float, help='the fraction of training data')
    parser.add_argument('--seed', default=50, type=int)
    # ======================federaser========================
    parser.add_argument('--unlearn_interval', default=1, type=int, help='')
    parser.add_argument('--forget_local_epoch_ratio', default=0.2, type=float)

    # ======================eraseclient========================
    parser.add_argument('--epoch_unlearn', default=20, type=int, help='')
    parser.add_argument('--num_iterations', default=50, type=int, help='')

    args = parser.parse_args()
    return args
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    # hyperparameters setting 超参数设置
    args = get_args()
    # set_random_seed(args.seed)

    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', args.device)

    # 模型初始化
    model = model_init(args)

    # data preparation
    # client_all_loaders：所有客户端的训练数据, test_loaders：测试数据集
    # proxy_client_loaders：代理客户端的训练数据（用于成员推理攻击, proxy_test_loaders：代理测试数据
    client_all_loaders, test_loaders, proxy_client_loaders, proxy_test_loaders = data_init(args)
    print(test_loaders[0])
    # client_all_loaders, test_loaders, proxy_loader = cross_data_init(args)

    # 初始设置不进行遗忘操作
    args.if_unlearning = False
    # 初始化 FUSED 算法实例
    case = fused_unlearning.FUSED(args)

    # 客户端遗忘逻辑
    if args.forget_paradigm == 'client':
        # 1.准备拜占庭攻击数据
        client_all_loaders_process, test_loaders_process = baizhanting_attack(args, copy.deepcopy(client_all_loaders),
                                                                              copy.deepcopy(test_loaders))
        proxy_client_loaders_process, proxy_test_loaders_process = baizhanting_attack(args, copy.deepcopy(
            proxy_client_loaders), copy.deepcopy(proxy_test_loaders))
        # 2.正常训练
        model, all_client_models = case.train_normal(model, client_all_loaders_process, test_loaders_process)
        args.if_unlearning = True
        # 3.执行遗忘
        unlearning_model = case.forget_client_train(copy.deepcopy(model), copy.deepcopy(client_all_loaders),
                                                    test_loaders_process)
        # 4.执行成员推理攻击
        if args.MIT:
            args.save_normal_result = False
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_process,
                                        test_loaders, proxy_client_loaders_process, proxy_client_loaders,
                                        proxy_test_loaders_process)
            args.save_normal_result = True
        # 5.重新训练验证
        if args.relearn:
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_process, test_loaders_process)
    # 类别遗忘逻辑
    elif args.forget_paradigm == 'class':
        # 1.备份原始数据
        client_all_loaders_bk = copy.deepcopy(client_all_loaders)
        proxy_client_loaders_bk = copy.deepcopy(proxy_client_loaders)
        # 2.正常训练
        model, all_client_models = case.train_normal(model, copy.deepcopy(client_all_loaders), test_loaders)
        args.if_unlearning = True
        # 3. 创建遗忘数据集（标签重映射）
        # 将遗忘类别的标签随机替换为其他类别
        # 文本数据和图像数据分别处理
        for user in range(args.num_user):
            train_ls = []
            proxy_train_ls = []
            if args.data_name == 'text':
                for batch in client_all_loaders[user]:
                    data = batch[0]
                    at = batch[1]
                    targets = batch[2]
                    for idx, label in enumerate(targets):
                        if label in args.forget_class_idx:
                            label_ls = [i for i in range(args.num_classes)]
                            label_ls.remove(label)
                            inverse_label = np.random.choice(label_ls)
                            label = inverse_label
                        train_ls.append((torch.tensor(data[idx]), torch.tensor(at[idx]), torch.tensor(label)))
            else:
                for data, target in client_all_loaders[user]:
                    data = data.tolist()
                    targets = target.tolist()
                    for idx, label in enumerate(targets):
                        if label in args.forget_class_idx:
                            label_ls = [i for i in range(args.num_classes)]
                            label_ls.remove(label)
                            inverse_label = np.random.choice(label_ls)
                            label = inverse_label
                        train_ls.append((torch.tensor(data[idx]), torch.tensor(label)))
                for data, target in proxy_client_loaders[user]:
                    data = data.tolist()
                    targets = target.tolist()
                    for idx, label in enumerate(targets):
                        if label in args.forget_class_idx:
                            label_ls = [i for i in range(args.num_classes)]
                            label_ls.remove(label)
                            inverse_label = np.random.choice(label_ls)
                            label = inverse_label
                        proxy_train_ls.append((torch.tensor(data[idx]), torch.tensor(label)))
            train_loader = DataLoader(train_ls, batch_size=args.test_batch_size, shuffle=True)
            proxy_train_loader = DataLoader(proxy_train_ls, batch_size=args.test_batch_size, shuffle=True)
            client_all_loaders[user] = train_loader
            proxy_client_loaders[user] = proxy_train_loader
        
        # 4. 执行类别遗忘
        unlearning_model = case.forget_class(copy.deepcopy(model), client_all_loaders, test_loaders)

        if args.MIT:
            args.save_normal_result = False
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), copy.deepcopy(client_all_loaders_bk),
                                        test_loaders, proxy_client_loaders_bk, proxy_client_loaders,
                                        proxy_test_loaders)
            args.save_normal_result = True
        if args.relearn:
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_bk, test_loaders)

    # 样本遗忘逻辑
    elif args.forget_paradigm == 'sample':
        # 1. 复制数据加载器 准备后门攻击数据
        client_all_loaders_attack = backdoor_attack(args, copy.deepcopy(client_all_loaders))
        proxy_client_loaders_attack = backdoor_attack(args, copy.deepcopy(proxy_client_loaders))
        # 2. 正常训练（含后门）
        model, all_client_models = case.train_normal(model, client_all_loaders_attack, test_loaders)
        args.if_unlearning = True
        # 3. 清除后门样本
        client_all_loaders_process = erase_backdoor(args, copy.deepcopy(client_all_loaders))
        proxy_client_loaders_process = erase_backdoor(args, copy.deepcopy(proxy_client_loaders))
        # 4. 执行样本遗忘
        unlearning_model = case.forget_sample(copy.deepcopy(model), client_all_loaders_process, test_loaders)

        # 5.  安全验证
        if args.MIT:
            args.save_normal_result = False
            membership_inference_attack(args, unlearning_model, case, copy.deepcopy(model), client_all_loaders_attack,
                                        test_loaders, proxy_client_loaders_attack, proxy_client_loaders_process,
                                        proxy_test_loaders)
            args.save_normal_result = True
        if args.relearn:
            case.relearn_unlearning_knowledge(unlearning_model, client_all_loaders_attack, test_loaders)

## 测试冲突分支            

## 新增测试分支

## 新增测试分支——2

## 新增测试分支——3