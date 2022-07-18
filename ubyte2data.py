from PIL import Image
import numpy as np
from tqdm import tqdm

'''
标签直接保存到txt文件中
图片先reshape成28x28大小，再将其名称长度统一为7，方便后续按顺序读取
名称格式为00....+idx.png
idx表示为第idx张图片，前面的0的数量为使其长度为7时所需的数量
'''
def convert(image_path, label_path, n):

    f_images = open(image_path[0], 'rb')
    f_labels = open(label_path[0], 'rb')
    f_out = open(label_path[1], 'w')  # 标签路径

    f_images.read(16)
    f_labels.read(1)

    images = []
    labels = []
    for i in range(n):
        image = []
        labels.append(ord(f_labels.read(1)))
        for j in range(28*28):
            image.append(ord(f_images.read(1)))
        images.append(image)
    idx = 0
    for image in tqdm(images):
        img = Image.fromarray(np.array(image).reshape((28, 28))).convert('L')
        if idx >= 10000:
            img.save(image_path[1] + '00' + str(idx) + '.png')
        elif idx >= 1000:
            img.save(image_path[1] + '000' + str(idx) + '.png')
        elif idx >= 100:
            img.save(image_path[1] + '0000' + str(idx) + '.png')
        elif idx >= 10:
            img.save(image_path[1] + '00000' + str(idx) + '.png')
        else:
            img.save(image_path[1] + '000000' + str(idx) + '.png')

        idx += 1

    for label in labels:
        f_out.write(str(label) + '\n')

    f_images.close()
    f_labels.close()
    f_out.close()


train_image_path = ['train-images.idx3-ubyte', 'dataset\\train\\']  # 训练集图片读取路径；训练集图片保存路径
train_label_path = ['train-labels.idx1-ubyte', 'dataset\\train.txt']  # 训练集标签读取路径；训练集标签保存路径
test_image_path = ['t10k-images.idx3-ubyte', 'dataset\\test\\']  # 测试集图片读取路径；测试集图片保存路径
test_label_path = ['t10k-labels.idx1-ubyte', 'dataset\\test.txt']  # 测试集标签读取路径；测试集标签保存路径
convert(train_image_path, train_label_path, 60000)
print('Generate the train sets done!')
convert(test_image_path, test_label_path, 10000)
print('Generate the test sets done!')
