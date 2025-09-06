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

from utils.options_async import args_parser
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

class FedAsyncTrainer:
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
        
        # 初始化参数列表和时间戳
        self.params_prev_list = [self.get_params_copy()]
        self.ts_list = [0]
        
        # 准备数据
        self.prepare_data()
        
        # 解析alpha衰减轮数
        self.alpha_decay_epoch = [int(i) for i in args.alpha_decay_epoch.split(',')]
        
        self.alpha = args.alpha
        self.rho = args.rho
        
        self.logger = setup_logging(args.log if hasattr(args, 'log') else 'fedasync.log')
        
        # 用于保存训练历史的变量
        self.train_history = {
            'epochs': [],
            'top1_acc': [],
            'top5_acc': [],
            'loss': [],
            'mean_delay': []
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
        
        # 更新参数列表
        self.params_prev_list = [self.get_params_copy()]
        self.ts_list = [0]
    
    def compute_alpha_factor(self, delay):
        """计算自适应alpha因子"""
        if self.args.alpha_type == 'power':
            if self.args.alpha_adaptive > 0:
                return 1 / math.pow(delay + 1.0, self.args.alpha_adaptive)
            else:
                return 1
        elif self.args.alpha_type == 'exp':
            return math.exp(-delay * self.args.alpha_adaptive)
        elif self.args.alpha_type == 'sigmoid':
            a = self.args.alpha_adaptive2
            c = self.args.alpha_adaptive
            b = math.exp(-c * delay)
            return (1 - b) / a + b
        elif self.args.alpha_type == 'hinge':
            if delay <= self.args.alpha_adaptive2:
                return 1
            else:
                return 1.0 / ((delay - self.args.alpha_adaptive2) * self.args.alpha_adaptive + 1)
        else:
            return 1
    
    def local_train(self, train_loader, params_prev):
        """本地训练过程"""
        # 设置优化器
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr,
                            momentum=self.args.momentum, weight_decay=0.0001)
        
        # 将参数prev转换为可用于正则化的形式
        if self.rho > 0:
            rho_params = {k: v * self.rho for k, v in params_prev.items()}
        else:
            rho_params = None
        
        # 本地训练多个epoch
        for local_epoch in range(self.args.iterations):
            for data, label in train_loader:
                data, label = data.to(self.device), label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.net(data)
                loss = F.cross_entropy(outputs, label)
                loss.backward()
                
                # 应用正则化（FedProx风格）
                if self.rho > 0 and rho_params is not None:
                    with torch.no_grad():
                        for name, param in self.net.named_parameters():
                            if param.requires_grad:
                                # w = w * (1-rho) + param_prev * rho
                                param.data = param.data * (1 - self.rho) + rho_params[name].to(self.device)
                
                optimizer.step()
    
    def evaluate(self):
        """评估模型性能"""
        self.net.eval()
        
        # 计算验证准确率
        top1_correct, top5_correct, total = 0, 0, 0
        total_loss = 0.0
        
        with torch.no_grad():
            for data, label in self.val_val_loader:
                data, label = data.to(self.device), label.to(self.device)
                outputs = self.net(data)
                
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
        
        self.net.train()
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
        plt.figure(figsize=(12, 5))
        
        # Top-1 准确率图
        plt.subplot(1, 2, 1)
        plt.plot(self.train_history['epochs'], self.train_history['top1_acc'], 'b-', label='Top-1 Accuracy')
        plt.plot(self.train_history['epochs'], self.train_history['top5_acc'], 'r-', label='Top-5 Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy vs Epoch')
        plt.legend()
        plt.grid(True)
        
        # Loss 图
        plt.subplot(1, 2, 2)
        plt.plot(self.train_history['epochs'], self.train_history['loss'], 'g-', label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss vs Epoch')
        plt.legend()
        plt.grid(True)
        
        # 保存图表
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Training history saved to {log_dir}/")
    
    def train(self):
        """主训练循环"""
        self.logger.info("Starting FedAsync training...")
        self.warm_up()
        
        sum_delay = 0
        tic = time.time()
        
        for epoch in range(self.args.epochs):
            # Alpha衰减
            if epoch in self.alpha_decay_epoch:
                self.alpha = self.alpha * self.args.alpha_decay
            
            # 随机选择历史模型
            model_idx = random.randint(0, len(self.params_prev_list) - 1)
            params_prev = self.params_prev_list[model_idx]
            
            # 加载历史参数
            self.set_params(params_prev)
            
            # 复制当前参数用于正则化
            current_params = self.get_params_copy()
            
            # 获取对应的时间戳和计算延迟
            worker_ts = self.ts_list[model_idx]
            worker_delay = epoch - worker_ts
            sum_delay += worker_delay
            
            # 随机选择客户端数据
            train_loader = random.choice(self.train_loaders)
            
            # 本地训练
            self.local_train(train_loader, current_params)
            
            # 计算自适应alpha
            alpha_factor = self.compute_alpha_factor(worker_delay)
            alpha_scaled = self.alpha * alpha_factor
            
            # 服务器端聚合
            server_params = self.params_prev_list[-1]
            current_net_params = self.get_params_copy()
            
            # 加权平均: w = server * (1-alpha_scaled) + current * alpha_scaled
            aggregated_params = {}
            for key in server_params.keys():
                aggregated_params[key] = (server_params[key] * (1 - alpha_scaled) + 
                                        current_net_params[key] * alpha_scaled)
            
            # 更新全局模型
            self.set_params(aggregated_params)
            
            # 保存历史参数
            self.params_prev_list.append(self.get_params_copy())
            self.ts_list.append(epoch + 1)
            
            # 限制历史列表长度
            if len(self.params_prev_list) > self.args.max_delay:
                del self.params_prev_list[0]
                del self.ts_list[0]
            
            # 验证和日志
            if epoch % self.args.interval == 0 or epoch == self.args.epochs - 1:
                top1_acc, top5_acc, avg_loss = self.evaluate()
                mean_delay = sum_delay / (epoch + 1)
                
                # 保存训练历史
                self.train_history['epochs'].append(epoch)
                self.train_history['top1_acc'].append(top1_acc)
                self.train_history['top5_acc'].append(top5_acc)
                self.train_history['loss'].append(avg_loss)
                self.train_history['mean_delay'].append(mean_delay)
                
                self.logger.info(
                    f'[Epoch {epoch}] validation: acc-top1={top1_acc:.4f} '
                    f'acc-top5={top5_acc:.4f}, loss={avg_loss:.4f}, '
                    f'lr={self.args.lr}, rho={self.rho}, alpha={self.alpha:.6f}, '
                    f'max_delay={self.args.max_delay}, mean_delay={mean_delay:.2f}, '
                    f'time={time.time()-tic:.2f}s'
                )
                tic = time.time()
        
        # 训练结束后保存历史数据
        self.save_training_history()

def main():
    args = args_parser()
    trainer = FedAsyncTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()