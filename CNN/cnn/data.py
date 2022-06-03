# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：data.py
@Author ：AnthonyZ
@Date ：2022/6/2 14:58
"""
import torchvision
import torch
import torchvision.transforms as transforms


mean = [0.5070751592371323, 0.48654887331495095, 0.4409178433670343]
std = [0.2673342858792401, 0.2564384629170883, 0.27615047132568404]
num_workers = 2


def cifar100_dataset(args):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),  # 数据增强
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean, std)])

    cifar10_training = torchvision.datasets.CIFAR10(root=args.data_path, train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(cifar10_training, batch_size=args.batch_size, shuffle=True, num_workers=num_workers)

    cifar10_testing = torchvision.datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(cifar10_testing, batch_size=100, shuffle=False, num_workers=num_workers)

    return train_loader, test_loader, cifar10_training.classes

