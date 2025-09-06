import argparse
import time
import logging
import os
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from utils.options_ca2fl import args_parser
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

class CA2FLTrainer:
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
        
        # CA2FL特有参数
        self.M = args.M  # 缓冲区大小
        self.eta = args.eta  # 全局学习率
        
        # 初始化缓存和缓冲区
        self.h_cache = [torch.zeros_like(self.model2tensor()) for _ in range(args.nsplit)]
        self.buffer = []  # 存储客户端更新差异
        self.buffer_clients = []  # 存储对应的客户端ID
        
        # 准备数据
        self.prepare_data()
        
        self.logger = setup_logging(args.log if hasattr(args, 'log') else 'ca2fl.log')

        # 用于保存训练历史的变量
        self.train_history = {
            'epochs': [],
            'top1_acc': [],
            'top5_acc': [],
            'loss': []
        }
        
    def get_params_copy(self):
        """获取模型参数的深拷贝"""
        return {k: v.clone().detach() for k, v in self.net.state_dict().items()}
    
    def set_params(self, params_dict):
        """设置模型参数"""
        self.net.load_state_dict(params_dict)
    
    def model2tensor(self):
        """将模型参数转换为张量"""
        return torch.cat([p.data.view(-1) for p in self.net.parameters()])
    
    def tensor2model(self, tensor):
        """将张量转换回模型参数"""
        pointer = 0
        for param in self.net.parameters():
            numel = param.numel()
            param.data = tensor[pointer:pointer+numel].view_as(param.data)
            pointer += numel
    
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
    
    def local_train(self, train_loader):
        """本地训练过程"""
        # 记录训练前的模型参数
        w_last = self.model2tensor()
        
        # 设置优化器
        optimizer = optim.SGD(self.net.parameters(), lr=self.args.lr,
                            momentum=self.args.momentum, weight_decay=0.0001)
        
        # 本地训练多个epoch
        self.net.train()
        for local_epoch in range(self.args.iterations):
            for data, label in train_loader:
                data, label = data.to(self.device), label.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.net(data)
                loss = F.cross_entropy(outputs, label)
                loss.backward()
                optimizer.step()
        
        # 计算本地模型更新量 dW = 训练后参数 - 训练前参数
        dW = self.model2tensor() - w_last
        return dW
    
    def aggregate(self, client_id, dW):
        """CA2FL聚合算法"""
        # 存入"缓存差值"：新更新 - 上一次的缓存
        delta = dW - self.h_cache[client_id]
        self.buffer.append(delta)
        self.buffer_clients.append(client_id)
        
        # 更新该客户端的缓存为这次的新更新
        self.h_cache[client_id] = dW.clone()
        
        # 当收集到 M 个更新时，触发一次全局聚合
        if len(self.buffer) >= self.M:
            # 1. 计算所有客户端缓存更新的均值 h_t
            h_cache_tensor = torch.stack(self.h_cache)
            h_t = torch.mean(h_cache_tensor, dim=0)
            
            # 2. 计算校正项
            calibration = torch.zeros_like(h_t)
            for delta, cid in zip(self.buffer, self.buffer_clients):
                calibration += delta - self.h_cache[cid]
            calibration = calibration / len(self.buffer)
            
            # 3. 最终的全局更新向量 v_t
            v_t = h_t + calibration
            
            # 4. 应用到全局模型： w_g = w_g + eta * v_t
            current_params = self.model2tensor()
            new_params = current_params + self.eta * v_t
            self.tensor2model(new_params)
            
            # 清空缓冲区
            self.buffer.clear()
            self.buffer_clients.clear()
            
    
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
        self.logger.info("Starting CA2FL training...")
        self.logger.info(f"Model parameters: {self.model2tensor().shape[0]}")
        self.logger.info(f"Number of clients: {len(self.train_loaders)}")
        
        tic = time.time()
        
        for epoch in range(self.args.epochs):
            # 随机选择客户端
            client_id = random.randint(0, len(self.train_loaders) - 1)
            train_loader = self.train_loaders[client_id]
            
            # 本地训练
            dW = self.local_train(train_loader)
            
            # 服务器端聚合
            self.aggregate(client_id, dW)
            
            # 验证和日志
            if epoch % self.args.interval == 0 or epoch == self.args.epochs - 1:
                top1_acc, top5_acc, avg_loss = self.evaluate()

                # 保存训练历史
                self.train_history['epochs'].append(epoch)
                self.train_history['top1_acc'].append(top1_acc)
                self.train_history['top5_acc'].append(top5_acc)
                self.train_history['loss'].append(avg_loss)
                
                self.logger.info(
                    f'[Epoch {epoch}] validation: acc-top1={top1_acc:.4f} '
                    f'acc-top5={top5_acc:.4f}, loss={avg_loss:.4f}, '
                    f'lr={self.args.lr}, eta={self.eta}, M={self.M}, '
                    f'buffer_size={len(self.buffer)}, '
                    f'time={time.time()-tic:.2f}s'
                )
                tic = time.time()
            
        # 训练结束后保存历史数据
        self.save_training_history()

def main():
    args = args_parser()
    trainer = CA2FLTrainer(args)
    trainer.train()

if __name__ == '__main__':
    main()