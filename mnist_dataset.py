import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class Mydataset(Dataset):
    def __init__(self, pic_path, label_path, device):
        super(Mydataset, self).__init__()
        pics = open(pic_path, 'r', encoding='utf-8').readlines()
        labels = open(label_path, 'r', encoding='utf-8').readlines()
        self.pics = pics
        self.labels = labels
        self.device = device

    # 索引你的数据和标签并返回
    def __getitem__(self, index):
        pic = np.array(Image.open(self.pics[index].strip()))
        label = int(self.labels[index].strip())

        # 转成tensor格式
        pic = torch.tensor(pic).to(self.device)
        pic = torch.unsqueeze(pic, 0).float()
        label = torch.tensor(label).to(self.device)
        # print(label)
        return pic, label

    # 返回数据长度
    def __len__(self):
        return len(self.pics)
