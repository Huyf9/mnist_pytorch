import os
from tqdm import tqdm

from typing import List
# 训练集与验证集比例 = 9 : 1
# 由于训练集与测试集已经区分开，所以不需要再划分
trainval_percent = 0.9

def split_sets(train_path, test_path):
    f_train = open(r'dataset\train.txt', 'w')
    f_test = open(r'dataset\test.txt', 'w')
    f_val_label = open(r'dataset\val_label.txt', 'w')
    f_train_label = open(r'dataset\tr_label.txt', 'w')

    # 将 dataset\\test 下的图片用os.listdir()函数获取到文件路径并写入txt文件
    test_pic_paths = os.listdir(test_path)
    for test_pic_path in tqdm(test_pic_paths):
        f_test.write(test_pic_path + '\n')
    print('Generate test.txt done!')

    # 将 dataset\\train 下的图片与标签划分为训练与验证，并依次存入txt文件
    train_pic_path = os.listdir(train_path[0])
    num = len(train_pic_path)
    train_num = int(num * trainval_percent)
    for i in tqdm(train_pic_path[0:train_num]):
        f_train.write(i + '\n')
    print('Generate train.txt done!')

    with open(train_path[1], 'r+') as f:
        train_labels = f.readlines()
        for train_label in tqdm(train_labels[0:train_num]):
            f_train_label.write(train_label)
        print('Generate tr_label.txt done!')

        for val_label in train_labels[train_num:]:
            f_val_label.write(val_label)
        print('Generate val_label.txt done!')

    f.close()
    f_train.close()
    f_test.close()
    f_val_label.close()
    f_train_label.close()


train_path = ['dataset\\train', 'dataset\\train_label.txt']
test_path = 'dataset\\test'
split_sets(train_path, test_path)
