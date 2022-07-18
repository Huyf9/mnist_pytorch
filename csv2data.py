mport numpy
import numpy as np
from tqdm import tqdm
from PIL import Image

'''
标签直接保存到txt文件中
图片先reshape成28x28大小，再将其名称长度统一为7，方便后续按顺序读取
名称格式为00....+idx.png
idx表示为第idx张图片，前面的0的数量为使其长度为7时所需的数量
'''

def convert(mnist, save_path):
    idx = 0

    f_tr_label = open(r'dataset\\train_label.txt', 'w')  # 保存train标签文件
    f_test_label = open(r'dataset\\test_label.txt', 'w')  # 保存test标签文件
    f_label = [f_tr_label, f_test_label]

    for i in range(len(mnist)):
        mnist_docs = open(mnist[i], 'r').readlines()
        for mnist_doc in tqdm(mnist_docs):
            mnist_doc = mnist_doc.strip().split(',')  # csv文件用逗号分隔数据，因此利用split()来以逗号划分成列表
            f_label[i].write(mnist_doc[0] + '\n')

            '''
            由于读取的pixel是字符型，需要将其转为整型
            利用map()函数来进行实现
            map()的语法规则为
            map(function, iterable, ...) --function为一个函数 --iterable为一个或多个序列
            因此使用map(int list_name)就可以将一个列表中的元素转为整型
            但是其返回的是迭代器，因此还需要用list()将其再转化为列表
            '''
            # print(mnist_doc[1:])
            img = list(map(int, mnist_doc[1:]))
            img = np.array(img).reshape((28, 28))  # mnist图片尺寸为28 x 28的
            img = Image.fromarray(img).convert('L')  # 将numpy矩阵转为image格式并且转化为灰度图，方便保存图片

            if idx >= 10000:
                img.save(save_path[i] + '00' + str(idx) + '.png')
            elif idx >= 1000:
                img.save(save_path[i] + '000' + str(idx) + '.png')
            elif idx >= 100:
                img.save(save_path[i] + '0000' + str(idx) + '.png')
            elif idx >= 10:
                img.save(save_path[i] + '00000' + str(idx) + '.png')
            else:
                img.save(save_path[i] + '000000' + str(idx) + '.png')

            idx += 1

        print('done!')


mnist = ['mnist_train.csv', 'mnist_test.csv']
save_path = [r'dataset\\train\\', r'dataset\\test\\']
convert(mnist, save_path)
