from alg.asyncbase import AsyncBaseClient, AsyncBaseServer, Status
from utils.time_utils import time_record

import torch


def add_args(parser):
    # 新增CA2FL的参数
    parser.add_argument('--M', type=int, default=10, help='buffer size M for CA2FL')
    parser.add_argument('--eta', type=float, default=0.01, help='gloal lr')
    return parser.parse_args()


class Client(AsyncBaseClient):
    def __init__(self, id, args):
        super().__init__(id, args)
        
    @time_record
    def run(self):# 记录本地训练前的模型参数
        w_last = self.model2tensor()
        # 执行本地训练（在客户端数据集上）
        self.train()
         # 计算本地模型更新量 dW = 训练后参数 - 训练前参数
        self.dW = self.model2tensor() - w_last


class Server(AsyncBaseServer):
    def __init__(self, id, args, clients):
        super().__init__(id, args, clients)
        self.buffer = []           # 缓冲区，用于存放客户端的“缓存差值”更新
        self.buffer_clients = []   # 记录本批次收集到的更新对应的客户端ID
        # h_cache[i]：客户端 i 的最新缓存更新向量
        self.h_cache = [torch.zeros_like(self.model2tensor()) for _ in self.clients]
        self.momentum_cache = torch.zeros_like(self.model2tensor())
        self.M = args.M              # 缓冲区大小
        self.eta = args.eta          # 全局学习率

    def run(self):
        self.sample()                   # 选择一个客户端
        self.downlink()                  # 下发当前全局模型参数给该客户端
        self.client_update()             # 客户端执行本地训练          
        self.uplink()                   # 接收客户端更新（dW）
        self.aggregate()                 # 聚合（如果收集到M个更新就更新全局模型）
        self.update_status()             # 更新客户端状态与延迟信息

    def aggregate(self):
        # 获取当前客户端 ID
        c_id = self.cur_client.id
        # 存入“缓存差值”：新更新 - 上一次的缓存
        self.buffer.append(self.cur_client.dW - self.h_cache[c_id])
        # 更新该客户端的缓存为这次的新更新
        self.h_cache[c_id] = self.cur_client.dW
        

        # 当收集到 M 个更新时，触发一次全局聚合
        if len(self.buffer) == self.M:
             # 1. 计算所有客户端缓存更新的均值 h_t
            h_t = torch.mean(torch.stack(self.h_cache), dim=0)
            # 2. 计算校正项：用本批次 buffer 中的差值减去缓存，平均得到
            calibration = sum(delta - self.h_cache[cid] for delta, cid in zip(self.buffer, self.buffer_clients)) / len(
                self.buffer)
            # 3. 最终的全局更新向量 v_t
            v_t = h_t + calibration
            

            # 4. 应用到全局模型： w_g = w_g + eta * v_t
            t_g = self.model2tensor()
            t_g_new = t_g + self.eta * v_t
            self.tensor2model(t_g_new)

    def update_status(self):
        # 当前客户端设为空闲
        self.cur_client.status = Status.IDLE

        # 如果完成了一个批次的聚合：
        if len(self.buffer) >= self.M:
            # 对仍处于活跃状态的客户端，延迟值加1
            for c in filter(lambda x: x.status == Status.ACTIVE, self.clients):
                self.staleness[c.id] += 1
            # 清空 buffer 和客户端记录，开始下一轮批次收集
            self.buffer.clear()
            self.buffer_clients.clear()