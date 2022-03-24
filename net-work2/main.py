# -*- coding: UTF-8 -*-
"""
@Project ：手写数字识别 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/3/23 23:51
"""
import struct
import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.cnov2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.relu2 = nn.ReLU()
        self.fc1 = nn.Linear(320, 50)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """
        向前输入
        :param x:输入的数据集
        :return: 映射到的十个类别
        """
        # x = x.view(-1, 28 * 28)
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.relu1(x)
        x = self.cnov2(x)
        x = self.conv2_drop(x)
        x = F.max_pool2d(x, 2)
        x = self.relu2(x)
        x = x.view(-1, 320)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return F.log_softmax(x)

def load_mnist(file_dir, is_img):
    bin_file = open(file_dir, 'rb')
    bin_data = bin_file.read()
    bin_file.close()
    if is_img:
        fmt_head = '>iiii'
        magic, num_images, num_rows, num_cols = struct.unpack_from(fmt_head, bin_data, 0)
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from(">" + str(data_size) + 'B', bin_data, struct.calcsize(fmt_head))
        mat_data = np.reshape(mat_data, [num_images, 1, num_rows, num_cols])
    else:
        fmt_head = '>ii'
        magic, num_images = struct.unpack_from(fmt_head, bin_data, 0)
        num_rows, num_cols = 1, 1
        data_size = num_images * num_rows * num_cols
        mat_data = struct.unpack_from(">" + str(data_size) + 'B', bin_data, struct.calcsize(fmt_head))
        mat_data = np.reshape(mat_data, [num_images, 1])
    return mat_data


def load_data(MNIST_DIR):
    train_images = load_mnist(os.path.join(MNIST_DIR, "train-images.idx3-ubyte"), True)
    train_labels = load_mnist(os.path.join(MNIST_DIR, "train-labels.idx1-ubyte"), False)
    test_images = load_mnist(os.path.join(MNIST_DIR, "t10k-images.idx3-ubyte"), True)
    test_labels = load_mnist(os.path.join(MNIST_DIR, "t10k-labels.idx1-ubyte"), False)
    F_train_data = torch.utils.data.TensorDataset(
        torch.tensor(train_images).to(torch.float32),
        torch.tensor(train_labels)
    )
    F_test_data = torch.utils.data.TensorDataset(
        torch.tensor(test_images).to(torch.float32),
        torch.tensor(test_labels)
    )
    return F_train_data, F_test_data


def train(epoch):
    """
    训练函数
    :return:
    """
    model.train()
    for index, (X_train, Y_train) in enumerate(train_loader):
        X_train = X_train.to('cpu') / 255.
        Y_train = Y_train.to('cpu')
        Y_train = Y_train.squeeze()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        if index % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%]\tLoss: {:.6f}'.format(
                    epoch,
                    index * len(X_train),
                    len(train_loader.dataset),
                    100. * index / len(train_loader),
                    loss.data.item()
                )
            )


def evaluation(loss_vector, accuracy_vector):
    """
    评估模型效果
    :return:
    """
    model.eval()
    val_loss, correct = 0, 0
    for X_test, Y_test in test_loader:
        X_test = X_test.to('cpu')
        Y_test = Y_test.to('cpu')
        Y_test = Y_test.squeeze()
        output = model(X_test)
        val_loss += criterion(output, Y_test).data.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(Y_test.data).cpu().sum()
    val_loss /= len(test_loader)
    loss_vector.append(val_loss)
    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(accuracy)
    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            val_loss,
            correct,
            len(test_loader.dataset),
            accuracy
        )
    )


if __name__ == "__main__":
    train_set, test_set = load_data("./data")
    train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    test_loader = torch.utils.data.DataLoader(train_set, shuffle=True, batch_size=64)
    model = Net()
    model.to('cpu')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
    criterion = nn.CrossEntropyLoss()

    plt_size = 2
    plt.figure(figsize=(10*plt_size, plt_size))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.axis("off")
        for (X_train, Y_train) in train_loader:
            plt.imshow(X_train[i, :].numpy().reshape(28, 28), cmap="gray_r")
            plt.title('Class: ' + str(Y_train[i].item()))
            break
    # plt.show()

    epochs = 10

    lossv, accv = [], []

    for epoch in range(1, epochs + 1):
        train(epoch)
        evaluation(lossv, accv)

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), lossv)
    plt.title('validation loss')
    plt.show()

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), accv)
    plt.title('validation accuracy')
    plt.show()
