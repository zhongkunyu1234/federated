import numpy as np
import pickle
from torch.utils.data import Dataset
import torch

class SplitDataset(Dataset):
    def __init__(self, file_path):
        with open(file_path, "rb") as f:
            images, labels = pickle.load(f)
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2).float()
        if len(labels.shape) > 1 and labels.shape[1] > 1:
            # 假设是one-hot编码，转换为类别索引
            self.labels = torch.from_numpy(np.argmax(labels, axis=1)).long()
        else:
            # 已经是类别索引
            self.labels = torch.from_numpy(labels).long().squeeze()
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]