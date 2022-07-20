import torch
import seaborn as sn
from matplotlib import pyplot as plt
from model import ConvNet
from MnistDataset import Mydataset
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd


torch.manual_seed(13)

def get_score(confusion_mat):
    smooth = 0.0001  #防止出现除数为0而加上一个很小的数
    tp = np.diagonal(confusion_mat)
    fp = np.sum(confusion_mat, axis=0)
    fn = np.sum(confusion_mat, axis=1)
    precision = tp / (fp + smooth)
    recall = tp / (fn + smooth)
    f1 = 2 * precision * recall / (precision + recall + smooth)
    return precision, recall, f1

def get_confusion(confusion_matrix, out, label):
    idx = np.argmax(out.detach().numpy())
    confusion_matrix[idx, label] += 1
    return confusion_matrix

def main():
    confusion_matrix = np.zeros((10, 10))

    net = ConvNet()
    net.load_state_dict(torch.load('model_parameter\\parameter_epo90.pth'))
    test_path = ['test.txt', r'dataset/test_label.txt']
    test_dataset = Mydataset(test_path[0], test_path[1], 'cpu')
    test_dataloader = DataLoader(test_dataset, 1, True)
    for i, (pic, label) in enumerate(test_dataloader):
        out = net(pic)
        confusion_matrix = get_confusion(confusion_matrix, out, label)

    precision, recall, f1 = get_score(confusion_matrix)
    print(f'precision: {np.average(precision)}\trecall: {np.average(recall)}\tf1: {np.average(f1)}')
    confusion_mat = pd.DataFrame(confusion_matrix)
    confusion_df = pd.DataFrame(confusion_mat, index=[i for i in range(10)], columns=[i for i in range(10)])
    sn.heatmap(data=confusion_df, cmap='RdBu_r')
    plt.show()
    confusion_df.to_csv(r'confusion.csv', encoding='ANSI')


if __name__ == '__main__':
    main()
