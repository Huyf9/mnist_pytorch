import torch
import torch.nn as nn
from MnistDataset import Mydataset
from torch.utils.data import DataLoader
from model import ConvNet
from tqdm import tqdm

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(64)  # 设置一个随机种子，保证每次训练的结果一样

train_path = ['train.txt', 'tr_label.txt']
val_path = ['val.txt', 'val_label.txt']
train_dataset = Mydataset(train_path[0], train_path[1], device)
val_dataset = Mydataset(val_path[0], val_path[1], device)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=32, shuffle=True, drop_last=True, num_workers=4)

net = ConvNet().to(device)
CRITERION = nn.NLLLoss()
LR = 0.001
OPTIMIZER = torch.optim.SGD(net.parameters(), lr=LR)
EPOCHS = 100

def train(epoch, epoch_loss, batch_num):

    for i, (pic, label) in tqdm(enumerate(train_dataloader)):

        batch_num += 1

        net.zero_grad()
        out = net(pic)
        # print(out)
        loss_value = loss(out, label)
        epoch_loss += loss_value

        loss_value.backward()
        optimizer.step()

    print(f'epoch: {epoch}\ttrain_loss: {epoch_loss/batch_num}')

def val(epoch, epoch_loss, batch_num):
    for i, (pic, label) in tqdm(enumerate(val_dataloader)):
        epoch_num += 1
        out = net(pic)
        loss_value = loss(out, label)
        epoch_loss += loss_value

    print(f'epoch: {epoch}\tval_loss: {epoch_loss/batch_num}')


def main():
    for epoch in range(EPOCHS):
        epoch_loss, batch_num = 0, 0
        train(epoch, epoch_loss, batch_num)
        val(epoch, epoch_loss, batch_num)

        if (epoch+1) % 10 == 0:
            torch.save(net.state_dict(), f'model_parameter\\parameter_epo{epoch}.pth')


if __name__ == '__main__':
    main()
