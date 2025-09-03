import argparse
import time
import logging
import os
import math
import random
import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import copy
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torchmetrics import Accuracy

# 设置随机种子 (优化版)
def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

# 自定义数据集类 (优化版)
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# 神经网络模型定义 (优化版)
class MLP(nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        
        # 初始化权重
        nn.init.xavier_uniform_(self.layer_input.weight)
        nn.init.zeros_(self.layer_input.bias)
        nn.init.xavier_uniform_(self.layer_hidden.weight)
        nn.init.zeros_(self.layer_hidden.bias)

    def forward(self, x):
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return x

class CNNMnist(nn.Module):
    def __init__(self, args, num_classes):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class CNNCifar(nn.Module):
    def __init__(self, args, num_classes):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)
        
        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 模型选择函数 (优化版)
def get_model(model_name, num_classes, dataset='cifar', args=None):
    if model_name == 'mlp':
        if dataset == 'mnist':
            len_in = 1 * 28 * 28
        elif dataset == 'cifar':
            len_in = 3 * 32 * 32
        else:
            len_in = 3 * 24 * 24  # 对于裁剪后的CIFAR
        return MLP(len_in, 200, num_classes)
    elif model_name == 'cnn_mnist':
        return CNNMnist(args, num_classes)
    elif model_name == 'cnn_cifar':
        return CNNCifar(args, num_classes)
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
        return model
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    else:
        raise ValueError(f"未知模型名称: {model_name}")

# 联邦平均算法 (优化版)
def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for key in w_avg.keys():
        # 使用向量化操作加速
        w_avg[key] = torch.stack([w[i][key] for i in range(len(w))]).mean(dim=0)
    return w_avg

# 本地更新类 (优化版)
class LocalUpdate(object):
    def __init__(self, args, dataset, idxs):
        self.args = args
        self.trainloader = DataLoader(
            dataset, 
            batch_size=self.args.local_bs, 
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
        self.idxs = idxs
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = GradScaler()  # 混合精度训练

    def train(self, net):
        net.train()
        optimizer = optim.SGD(
            net.parameters(), 
            lr=self.args.lr, 
            momentum=self.args.momentum,
            nesterov=True
        )
        
        epoch_loss = []
        for _ in range(self.args.local_ep):
            batch_loss = []
            for data, target in self.trainloader:
                data, target = data.to(self.args.device, non_blocking=True), target.to(self.args.device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():  # 混合精度训练
                    output = net(data)
                    loss = self.criterion(output, target)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
                
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

# 测试函数 (优化版)
def test_img(net, dataloader, args):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad(), autocast():
        for data, target in dataloader:
            data, target = data.to(args.device, non_blocking=True), target.to(args.device, non_blocking=True)
            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()
            total += target.size(0)
    
    test_loss /= total
    accuracy = 100. * correct / total
    return accuracy, test_loss

# 自适应alpha计算 (优化版)
def compute_alpha_factor(args, worker_delay):
    if args.alpha_type == 'power':
        alpha_factor = 1 / math.pow(worker_delay + 1.0, args.alpha_adaptive) if args.alpha_adaptive > 0 else 1
    elif args.alpha_type == 'exp':
        alpha_factor = math.exp(-worker_delay * args.alpha_adaptive)
    elif args.alpha_type == 'sigmoid':
        a = args.alpha_adaptive2
        b = math.exp(-args.alpha_adaptive * worker_delay)
        alpha_factor = (1 - b) / a + b
    elif args.alpha_type == 'hinge':
        alpha_factor = 1 if worker_delay <= args.alpha_adaptive2 else 1.0 / ((worker_delay - args.alpha_adaptive2) * args.alpha_adaptive + 1)
    else:
        alpha_factor = 1
    return alpha_factor

# 数据加载函数 (优化版)
def load_fed_data(args):
    train_data_list = []
    train_dir = os.path.join(args.dir, 'train')
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练目录不存在: {train_dir}")
    
    # 预加载所有文件名
    filenames = sorted([f for f in os.listdir(train_dir) if f.endswith('.pkl')])
    
    for filename in filenames:
        filepath = os.path.join(train_dir, filename)
        try:
            with open(filepath, "rb") as f:
                B, L = pickle.load(f)
            # 使用更快的转换方式
            B = torch.as_tensor(np.transpose(B, (0, 3, 1, 2)), dtype=torch.float32)
            L = torch.as_tensor(L, dtype=torch.long).squeeze()
            dataset = CustomDataset(B, L)
            # 优化DataLoader配置
            dataloader = DataLoader(
                dataset, 
                batch_size=args.batchsize, 
                shuffle=True,
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
            train_data_list.append(dataloader)
        except Exception as e:
            print(f"加载文件 {filename} 失败: {str(e)}")
            continue
    
    return train_data_list

# 同步联邦学习训练 (优化版)
def train_fed_sync(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    # 加载标准数据集 (优化数据转换)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)) if args.dataset == 'cifar' 
        else transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    if args.dataset == 'mnist':
        dataset_train = datasets.MNIST(
            './data/mnist/', 
            train=True, 
            download=True, 
            transform=transform
        )
        dataset_test = datasets.MNIST(
            './data/mnist/', 
            train=False, 
            download=True, 
            transform=transform
        )
    elif args.dataset == 'cifar':
        dataset_train = datasets.CIFAR10(
            './data/cifar', 
            train=True, 
            download=True, 
            transform=transform
        )
        dataset_test = datasets.CIFAR10(
            './data/cifar', 
            train=False, 
            download=True, 
            transform=transform
        )
    else:
        raise ValueError("不支持的dataset类型")
    
    # 构建模型 (优化初始化)
    if args.model == 'cnn' and args.dataset == 'cifar':
        net_glob = CNNCifar(args, args.classes).to(device)
    elif args.model == 'cnn' and args.dataset == 'mnist':
        net_glob = CNNMnist(args, args.classes).to(device)
    elif args.model == 'mlp':
        len_in = 1
        img_size = dataset_train[0][0].shape
        for x in img_size:
            len_in *= x
        net_glob = MLP(len_in, 200, args.classes).to(device)
    else:
        net_glob = get_model(args.model, args.classes, args.dataset, args).to(device)
    
    if torch.cuda.device_count() > 1:
        print(f"使用 {torch.cuda.device_count()} 个GPU!")
        net_glob = nn.DataParallel(net_glob)
    
    print(net_glob)
    net_glob.train()
    w_glob = net_glob.state_dict()
    
    # 划分用户数据 (优化划分过程)
    if args.iid:
        dict_users = {}
        all_idxs = list(range(len(dataset_train)))
        num_items = int(len(dataset_train) / args.num_users)
        for i in range(args.num_users):
            dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
            all_idxs = list(set(all_idxs) - dict_users[i])
    else:
        dict_users = {}
        idxs = np.arange(len(dataset_train))
        labels = np.array([dataset_train[i][1] for i in range(len(dataset_train))])
        idxs_labels = np.vstack((idxs, labels))
        idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
        idxs = idxs_labels[0, :]
        
        num_shards = args.num_users * 2
        num_imgs = len(dataset_train) // num_shards
        for i in range(args.num_users):
            rand_set = set(np.random.choice(range(num_shards), 2, replace=False))
            dict_users[i] = []
            for rand in rand_set:
                dict_users[i] = set(dict_users[i]) | set(idxs[rand*num_imgs:(rand+1)*num_imgs])
            dict_users[i] = list(dict_users[i])
    
    # 训练循环 (优化训练过程)
    loss_train = []
    best_acc = 0.0
    scaler = GradScaler()
    
    for epoch in range(args.epochs):
        w_locals, loss_locals = [], []
        
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        
        for idx in idxs_users:
            local = LocalUpdate(args, dataset_train, dict_users[idx])
            w, loss = local.train(copy.deepcopy(net_glob))
            w_locals.append(copy.deepcopy(w))
            loss_locals.append(copy.deepcopy(loss))
        
        w_glob = FedAvg(w_locals)
        net_glob.load_state_dict(w_glob)
        
        loss_avg = sum(loss_locals) / len(loss_locals)
        loss_train.append(loss_avg)
        
        if epoch % args.interval == 0:
            acc_test, loss_test = test_img(
                net_glob, 
                DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=4), 
                args
            )
            print(f'Epoch {epoch}: Test Accuracy {acc_test:.2f}%, Test Loss {loss_test:.4f}')
            
            if acc_test > best_acc:
                best_acc = acc_test
                torch.save(net_glob.state_dict(), f'./log/best_model_{args.model}.pth')
    
    # 最终测试
    net_glob.eval()
    acc_test, loss_test = test_img(
        net_glob, 
        DataLoader(dataset_test, batch_size=1000, shuffle=False, num_workers=4), 
        args
    )
    print(f'Final Test Accuracy: {acc_test:.2f}%')
    
    return loss_train, acc_test

# 异步联邦学习训练 (优化版)
def train_fed_async(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    args.device = device
    
    # 加载数据 (优化数据加载)
    train_data_list = load_fed_data(args)
    
    # 加载验证数据 (优化加载过程)
    val_dir = os.path.join(args.valdir, 'val')
    with open(os.path.join(val_dir, "val_data.pkl"), "rb") as f:
        B, L = pickle.load(f)
        B = torch.as_tensor(np.transpose(B, (0, 3, 1, 2)), dtype=torch.float32)
        L = torch.as_tensor(L, dtype=torch.long).squeeze()
        val_data = DataLoader(
            CustomDataset(B, L), 
            batch_size=1000, 
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
    
    # 构建模型 (优化初始化)
    model = get_model(args.model, args.classes, 'cifar', args).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler()
    
    # 历史模型池 (优化存储)
    params_prev_list = [torch.cat([param.data.clone().flatten() for param in model.parameters()]).cpu()]
    ts_list = [0]
    sum_delay = 0
    
    # 训练循环 (优化训练过程)
    loss_history = []
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # alpha衰减
        if epoch in [int(e) for e in args.alpha_decay_epoch.split(',')]:
            args.alpha *= args.alpha_decay
        
        # 从历史模型池中随机选择 (优化选择过程)
        model_idx = random.randint(0, len(params_prev_list)-1)
        selected_params = params_prev_list[model_idx].to(device)
        
        # 恢复选中的模型参数 (优化参数恢复)
        with torch.no_grad():
            start_idx = 0
            for param in model.parameters():
                end_idx = start_idx + param.numel()
                param.data.copy_(selected_params[start_idx:end_idx].reshape(param.shape))
                start_idx = end_idx
        
        # 正则化参数准备 (优化计算)
        params_prev = torch.cat([param.data.clone().flatten() for param in model.parameters()])
        if args.rho > 0:
            params_prev = params_prev * args.rho
        
        worker_ts = ts_list[model_idx]
        
        # 本地训练 (优化训练循环)
        train_loader = random.choice(train_data_list)
        optimizer = optim.SGD(
            model.parameters(), 
            lr=args.lr, 
            momentum=args.momentum,
            nesterov=True
        )
        
        for _ in range(args.iterations):
            for data, labels in train_loader:
                data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                
                with autocast():
                    outputs = model(data)
                    loss = criterion(outputs, labels)
                    
                    if args.rho > 0:
                        current_params = torch.cat([param.data.flatten() for param in model.parameters()])
                        loss += 0.5 * torch.sum((current_params - params_prev) ** 2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        
        # 计算worker延迟
        worker_delay = epoch - worker_ts
        sum_delay += worker_delay
        
        # 计算自适应alpha
        alpha_factor = compute_alpha_factor(args, worker_delay)
        alpha_scaled = args.alpha * alpha_factor
        
        # 模型聚合 (优化聚合过程)
        with torch.no_grad():
            global_params = params_prev_list[-1].to(device)
            new_params = torch.cat([param.data.clone().flatten() for param in model.parameters()])
            updated_params = global_params * (1 - alpha_scaled) + new_params * alpha_scaled
            
            # 更新模型池 (优化存储)
            params_prev_list.append(updated_params.cpu())
            ts_list.append(epoch + 1)
            
            if len(params_prev_list) > args.max_delay:
                params_prev_list.pop(0)
                ts_list.pop(0)
        
        # 验证 (优化验证过程)
        if epoch % args.interval == 0:
            model.eval()
            val_loss, correct, total = 0, 0, 0
            with torch.no_grad(), autocast():
                for data, labels in val_data:
                    data, labels = data.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                    outputs = model(data)
                    val_loss += criterion(outputs, labels).item() * data.size(0)
                    _, predicted = outputs.max(1)
                    correct += predicted.eq(labels).sum().item()
                    total += labels.size(0)
            
            val_loss /= total
            val_acc = 100. * correct / total
            loss_history.append(val_loss)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'./log/best_model_{args.model}_async.pth')
            
            print(f'Epoch {epoch}: Val Accuracy {val_acc:.2f}%, Val Loss {val_loss:.4f}, '
                  f'Delay {worker_delay}, Alpha {alpha_scaled:.4f}')
    
    return loss_history

# 主函数 (优化版)
def main():
    parser = argparse.ArgumentParser()
    
    # 通用参数
    parser.add_argument("--mode", type=str, default="sync", choices=["sync", "async"], help="训练模式")
    parser.add_argument("--dataset", type=str, default="cifar", choices=["mnist", "cifar"])
    parser.add_argument("--model", type=str, default="cnn_cifar")
    parser.add_argument("--num_users", type=int, default=100)
    parser.add_argument("--frac", type=float, default=0.1)
    parser.add_argument("--local_ep", type=int, default=5)
    parser.add_argument("--local_bs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--momentum", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--iid", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--classes", type=int, default=10)
    
    # 异步特定参数
    parser.add_argument("--dir", type=str, help="数据目录")
    parser.add_argument("--valdir", type=str, help="验证数据目录")
    parser.add_argument("--batchsize", type=int, default=50)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--rho", type=float, default=0.00001)
    parser.add_argument("--alpha", type=float, default=0.9)
    parser.add_argument("--alpha-decay", type=float, default=0.5)
    parser.add_argument("--alpha-decay-epoch", type=str, default='400')
    parser.add_argument("--alpha-adaptive", type=float, default=0)
    parser.add_argument("--alpha-adaptive2", type=float, default=1)
    parser.add_argument("--alpha-type", type=str, default='none')
    parser.add_argument("--max-delay", type=int, default=10)
    
    args = parser.parse_args()
    
    # 创建日志目录
    os.makedirs('./log', exist_ok=True)
    
    if args.mode == "sync":
        print("开始同步联邦学习训练...")
        loss_history, final_acc = train_fed_sync(args)
        print(f"同步训练完成，最终准确率: {final_acc:.2f}%")
    else:
        print("开始异步联邦学习训练...")
        if not args.dir or not args.valdir:
            raise ValueError("异步训练需要指定数据目录(--dir)和验证数据目录(--valdir)")
        loss_history = train_fed_async(args)
        print("异步训练完成")
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history)
    plt.title(f'{args.mode.upper()} Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(f'./log/{args.mode}_training_loss.png')

if __name__ == '__main__':
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU instead.")
    
    main()