import argparse
import time
import logging
import os
import math
import random
import numpy as np
from scipy import stats

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from utils.options_sync import args_parser
from utils.sampling import SplitDataset
from models.Nets import get_model

import pickle
from os import listdir
import matplotlib.pyplot as plt 

def setup_logging(log_file):
    logger = logging.getLogger('')
    logger.setLevel(logging.INFO)
    
    file_handler = logging.FileHandler(log_file)
    stream_handler = logging.StreamHandler()
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    stream_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    
    return logger

class FedSyncTrainer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() and args.gpu != -1 else 'cpu')
        
        # 设置随机种子
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        
        # 加载模型
        self.net = get_model(args.model, num_classes=args.classes).to(self.device)
        self.global_model = get_model(args.model, num_classes=args.classes).to(self.device)
        
        # 准备数据
        self.prepare_data()
        
        # 学习率衰减设置
        self.lr_decay_epoch = [int(i) for i in args.lr_decay_epoch.split(',')] if hasattr(args, 'lr_decay_epoch') else []
        
        self.logger = setup_logging('./log/fedsync.log')
        
        # 用于保存训练历史的变量
        self.train_history = {
            'epochs': [],
            'top1_acc': [],
            'top5_acc': [],
            'loss': [],
            'client_accuracies': []
        }
        
    def get_params_copy(self):
        """获取模型参数的深拷贝"""
        return {k: v.clone().detach() for k, v in self.net.state_dict().items()}
    
    def set_params(self, params_dict):
        """设置模型参数"""
        self.net.load_state_dict(params_dict)
    
    def prepare_data(self):
        """准备训练和验证数据"""
        data_dir = os.path.join(self.args.dir, f'dataset_split_{self.args.nsplit}')
        train_dir = os.path.join(data_dir, 'train')
        val_dir = os.path.join(data_dir, 'val')
        
        # 训练数据
        self.train_loaders = []
        training_files = []
        for filename in sorted(listdir(train_dir)):
            if filename.endswith('.pkl'):
                absolute_filename = os.path.join(train_dir, filename)
                training_files.append(absolute_filename)
        
        for file_path in training_files[:self.args.nsplit]:
            dataset = SplitDataset(file_path)
            loader = DataLoader(dataset, batch_size=self.args.batchsize, 
                              shuffle=True, num_workers=2)
            self.train_loaders.append(loader)
        
        # 验证数据
        val_train_file = os.path.join(val_dir, 'train_data.pkl')
        val_val_file = os.path.join(val_dir, 'val_data.pkl')
        
        self.val_train_loader = DataLoader(SplitDataset(val_train_file), 
                                         batch_size=1000, shuffle=False)
        self.val_val_loader = DataLoader(SplitDataset(val_val_file), 
                                       batch_size=1000, shuffle=False)
    
    def warm_up(self):
        """预热训练"""
        self.logger.info("Starting warm up...")
        optimizer = optim.SGD(self.net.parameters(), lr=0.01, 
                            momentum=self.args.momentum, weight_decay=0.0001)
        
        for local_epoch in range(5):
            for data, label in self.train_loaders[0]:
                data, label = data.to(self.device), label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.net(data)
                loss = F.cross_entropy(outputs, label)
                loss.backward()
                optimizer.step()
        
        # 初始化全局模型
        self.global_model.load_state_dict(self.get_params_copy())
    
    def local_train(self, train_loader, global_params):
        """本地训练过程 - 同步版本"""
        # 加载全局模型参数
        self.set_params(global_params)
        
        # 设置优化器
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr,
                            momentum=self.args.momentum, weight_decay=0.0001)
        
        local_losses = []
        
        # 本地训练多个epoch
        for local_epoch in range(self.args.local_epochs):  # 使用 local_epochs
            epoch_loss = 0
            batch_count = 0
            
            for data, label in train_loader:
                data, label = data.to(self.device), label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.net(data)
                loss = F.cross_entropy(outputs, label)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                batch_count += 1
            
            local_losses.append(epoch_loss / batch_count)
        
        # 返回更新后的参数和平均损失
        return self.get_params_copy(), sum(local_losses) / len(local_losses)
    
    def evaluate_client(self, client_idx):
        """评估客户端模型性能"""
        self.net.eval()
        
        client_loader = self.train_loaders[client_idx]
        correct, total = 0, 0
        
        with torch.no_grad():
            for data, label in client_loader:
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.net(data)
                _, pred = torch.max(outputs, 1)
                correct += (pred == label).sum().item()
                total += label.size(0)
        
        self.net.train()
        return correct / total if total > 0 else 0
    
    def evaluate(self):
        """评估全局模型性能"""
        self.global_model.eval()
        
        # 计算验证准确率
        top1_correct, top5_correct, total = 0, 0, 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, label in self.val_val_loader:
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.global_model(data)
                
                # Top-1准确率
                _, pred = torch.max(outputs, 1)
                top1_correct += (pred == label).sum().item()
                
                # Top-5准确率
                _, top5_pred = outputs.topk(5, 1, True, True)
                top5_correct += torch.eq(top5_pred, label.view(-1, 1)).sum().item()
                
                total += label.size(0)
                
                # 计算损失
                loss = F.cross_entropy(outputs, label)
                total_loss += loss.item() * label.size(0)
        
        top1_acc = top1_correct / total
        top5_acc = top5_correct / total
        avg_loss = total_loss / total
        
        self.global_model.train()
        return top1_acc, top5_acc, avg_loss
    
    def save_training_history(self):
        """保存训练历史到文件并绘制图表"""
        # 确保日志目录存在
        log_dir = './log'
        os.makedirs(log_dir, exist_ok=True)
        
        # 保存训练历史数据
        history_file = os.path.join(log_dir, 'training_history.pkl')
        with open(history_file, 'wb') as f:
            pickle.dump(self.train_history, f)
        
        # 绘制准确率图表
        plt.figure(figsize=(15, 5))
        
        # Top-1 准确率图
        plt.subplot(1, 3, 1)
        plt.plot(self.train_history['epochs'], self.train_history['top1_acc'], 'b-', label='Top-1 Accuracy')
        plt.plot(self.train_history['epochs'], self.train_history['top5_acc'], 'r-', label='Top-5 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Global Model Accuracy')
        plt.legend()
        plt.grid(True)
        
        # Loss 图
        plt.subplot(1, 3, 2)
        plt.plot(self.train_history['epochs'], self.train_history['loss'], 'g-', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Global Model Loss')
        plt.legend()
        plt.grid(True)
        
        # 客户端准确率热力图
        plt.subplot(1, 3, 3)
        client_acc_matrix = np.array(self.train_history['client_accuracies'])
        plt.imshow(client_acc_matrix.T, aspect='auto', cmap='viridis')
        plt.colorbar(label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Client ID')
        plt.title('Client Accuracies Over Time')
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history saved to {log_dir}/")
    
    def fedavg_aggregate(self, client_params_list):
        """FedAvg聚合算法"""
        aggregated_params = {}
        
        # 假设所有客户端数据量相同，等权重平均
        num_clients = len(client_params_list)
        
        # 初始化聚合参数
        for key in client_params_list[0].keys():
            aggregated_params[key] = torch.zeros_like(client_params_list[0][key])
        
        # 累加所有客户端的参数
        for params in client_params_list:
            for key in params.keys():
                aggregated_params[key] += params[key]
        
        # 计算平均值
        for key in aggregated_params.keys():
            aggregated_params[key] = aggregated_params[key] / num_clients
        
        return aggregated_params
    
    def train(self):
        """主训练循环 - 同步版本"""
        self.logger.info("Starting FedSync training...")
        self.warm_up()
        
        tic = time.time()
        
        for epoch in range(self.args.epochs):
            # 学习率衰减
            if epoch in self.lr_decay_epoch:
                self.args.lr = self.args.lr * self.args.lr_decay
                self.logger.info(f"Learning rate decayed to {self.args.lr}")
            
            # 选择参与本轮训练的客户端
            num_selected = max(self.args.min_clients, int(self.args.frac * len(self.train_loaders)))  # 使用 min_clients
            selected_clients = random.sample(range(len(self.train_loaders)), num_selected)
            
            client_params_list = []
            client_losses = []
            client_accuracies = []
            
            # 每个选中的客户端进行本地训练
            for client_idx in selected_clients:
                # 获取全局模型参数
                global_params = self.global_model.state_dict()
                
                # 本地训练
                client_params, client_loss = self.local_train(
                    self.train_loaders[client_idx], global_params
                )
                
                client_params_list.append(client_params)
                client_losses.append(client_loss)
                
                # 评估客户端性能
                client_acc = self.evaluate_client(client_idx)
                client_accuracies.append(client_acc)
            
            # 服务器端聚合 (FedAvg)
            aggregated_params = self.fedavg_aggregate(client_params_list)
            self.global_model.load_state_dict(aggregated_params)
            
            # 更新当前网络为全局模型
            self.set_params(aggregated_params)
            
            # 验证和日志
            if epoch % self.args.interval == 0 or epoch == self.args.epochs - 1:
                top1_acc, top5_acc, avg_loss = self.evaluate()
                mean_client_loss = sum(client_losses) / len(client_losses)
                mean_client_acc = sum(client_accuracies) / len(client_accuracies)
                
                # 保存训练历史
                self.train_history['epochs'].append(epoch)
                self.train_history['top1_acc'].append(top1_acc)
                self.train_history['top5_acc'].append(top5_acc)
                self.train_history['loss'].append(avg_loss)
                self.train_history['client_accuracies'].append(client_accuracies)
                
                self.logger.info(
                    f'[Epoch {epoch}] Global - acc-top1={top1_acc:.4f} '
                    f'acc-top5={top5_acc:.4f}, loss={avg_loss:.4f}\n'
                    f'Clients - mean_acc={mean_client_acc:.4f}, '
                    f'mean_loss={mean_client_loss:.4f}, '
                    f'selected={num_selected}/{len(self.train_loaders)}, '
                    f'lr={self.args.lr}, time={time.time()-tic:.2f}s'
                )
                tic = time.time()
        
        # 训练结束后保存历史数据
        self.save_training_history()

def main():
    args = args_parser()
    trainer = FedSyncTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()