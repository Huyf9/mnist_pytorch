from model import ConvNet
from PIL import Image
import numpy as np
import torch

net = ConvNet()
net.load_state_dict(torch.load(r'model_parameter\parameter_epo10.pth'))

pic_path = input("请输入图片路径: ")
pic = Image.open(pic_path).convert('L')
pic = pic.resize((28, 28))  # [H, W]
pic = torch.tensor(np.array(pic)).float()
pic = torch.unsqueeze(pic, 0).float()  # [C, H, W]
pic = torch.unsqueeze(pic, 0)  # [B, C, H, W]

out = net(pic)
out = out.detach().numpy()
num = np.argmax(out)

print(f'图片中的数字被预测为：{num}')
