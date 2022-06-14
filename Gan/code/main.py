# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/6/14 14:58
"""

from model import *
from utils import *

import argparse

import torchvision.datasets
import torchvision.transforms as transforms


def train(args):
    for epoch in range(args.epochs):  # 3 epochs
        for i, data in enumerate(dataloader, 0):
            # STEP 1: Discriminator optimization step
            x_real, _ = next(iter(dataloader))
            x_real = x_real.to(args.device)
            # reset accumulated gradients from previous iteration
            optimizerD.zero_grad()

            D_x = D(x_real)
            lossD_real = criterion(D_x, lab_real.to(args.device))

            z = torch.randn(args.batch_size, 100, device=args.device)  # random noise, 64 samples, z_dim=100
            x_gen = G(z).detach()
            D_G_z = D(x_gen)
            lossD_fake = criterion(D_G_z, lab_fake.to(args.device))

            lossD = lossD_real + lossD_fake
            lossD.backward()
            optimizerD.step()

            # STEP 2: Generator optimization step
            # reset accumulated gradients from previous iteration
            optimizerG.zero_grad()

            z = torch.randn(args.batch_size, 100, device=args.device)  # random noise, 64 samples, z_dim=100
            x_gen = G(z)
            D_G_z = D(x_gen)
            lossG = criterion(D_G_z, lab_real)  # -log D(G(z))

            lossG.backward()
            optimizerG.step()
            if i % 100 == 0:
                x_gen = G(fixed_noise)
                show_imgs(x_gen, new_fig=False)
                fig.canvas.draw()
                print('e{}.i{}/{} last mb D(x)={:.4f} D(G(z))={:.4f}'.format(
                    epoch, i, len(dataloader), D_x.mean().item(), D_G_z.mean().item()))
        # End of epoch
        x_gen = G(fixed_noise)
        collect_x_gen.append(x_gen.detach().clone())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="./FashionMNIST/", type=str, help="The input data dir")
    parser.add_argument("--batch_size", default=64, type=int, help="The batch size of training")
    parser.add_argument("--device", default='mps', type=str, help="The training device")
    parser.add_argument("--learning_rate", default=0.01, type=int, help="learning rate")
    parser.add_argument("--epochs", default=10, type=int, help="Training epoch")

    args = parser.parse_args()

    dataset = torchvision.datasets.FashionMNIST(root=args.data_path,
                                                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
                                                download=True)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    D = Discriminator().to(args.device)
    G = Generator().to(args.device)

    optimizerD = torch.optim.SGD(D.parameters(), lr=args.learning_rate)
    optimizerG = torch.optim.SGD(G.parameters(), lr=args.learning_rate)

    criterion = nn.BCELoss()

    lab_real = torch.ones(args.batch_size, 1, device=args.device)
    lab_fake = torch.zeros(args.batch_size, 1, device=args.device)

    collect_x_gen = []
    fixed_noise = torch.randn(args.batch_size, 100, device=args.device)
    fig = plt.figure()  # keep updating this one
    plt.ion()

    train(args)

    for x_gen in collect_x_gen:
        show_imgs(x_gen)


