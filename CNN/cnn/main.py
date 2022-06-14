# -*- coding: UTF-8 -*-
"""
@Project ：base 
@File ：main.py
@Author ：AnthonyZ
@Date ：2022/6/2 14:58
"""
import argparse

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from data import *
from utils import *
from Net import *


def train(args, epoch):
    # running_loss = 0.0
    net.train()
    train_tqdm = tqdm(train_loader, desc="Epoch " + str(epoch))
    for index, (inputs, labels) in enumerate(train_tqdm):
        # print(inputs.shape, labels.shape)
        # get the inputs; data is a list of [inputs, labels]
        # inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs.to(args.device))
        loss = criterion(outputs, labels.to(args.device))
        loss.backward()
        optimizer.step()

        # print statistics
        # running_loss += loss.item()
        writer.add_scalar("loss/train", loss, (epoch + 1) * (1 + index))
        train_tqdm.set_postfix({"loss": "%.3g" % loss.item()})

        # if index % 20 == 19:    # print every 2000 mini-batches
        #     logger.info(f'[{epoch + 1}, {index + 1:5d}] loss: {running_loss / 2000:.3f}')
        #     running_loss = 0.0


def validate(args, epoch, loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct = 0, 0
    for index, (data, target) in enumerate(test_loader):
        data = data.to(args.device)
        target = target.to(args.device)
        output = net(data)
        val_loss += criterion(output, target.to(args.device)).data.item()

        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(test_loader)
    loss_vector.append(val_loss)
    writer.add_scalar("loss/validation", val_loss, epoch)

    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    accuracy_vector.append(accuracy)
    writer.add_scalar("accuracy/validation", accuracy, epoch)

    logger.info("***** Eval results *****")
    logger.info('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset), accuracy))


def main(args, loss_vector, accuracy_vector):
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_loader) * args.batch_size)
    logger.info("  Num Epochs = %d", args.epochs)
    logger.info("  Batch size = %d", args.batch_size)
    for epoch in range(args.epochs):
        train(args, epoch)
        PATH = os.path.join(args.logdir, 'cifar_net.pth')
        torch.save(net.state_dict(), PATH)
        with torch.no_grad():
            validate(args, epoch, loss_vector, accuracy_vector)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", default="../data", type=str, help="The input data dir")
    parser.add_argument("--batch_size", default=4, type=int, help="The batch size of training")
    parser.add_argument("--device", default='cpu', type=str, help="The training device")
    parser.add_argument("--learning_rate", default=0.0004, type=int, help="learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="Training epoch")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--model", default="Densenet", type=str, help="Resnet Base Densenet")

    args = parser.parse_args()

    train_loader, test_loader, classes = cifar100_dataset(args)

    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # imshow(torchvision.utils.make_grid(images))

    writer = SummaryWriter(os.path.join(args.logdir, "tensorboard"))
    # writer2 = SummaryWriter(args.logdir)
    # writer3 = SummaryWriter(args.logdir)
    if args.model == "Resnet":
        net = ResNet().to(args.device)
    if args.model == "Base":
        net = Net().to(args.device)
    if args.model == "Densenet":
        net = DenseNet().to(args.device)
    else:
        print("没有该模型，error")
        exit()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)

    if not os.path.exists(args.logdir):
        os.makedirs('./log')
    logger = init_logger()
    # args_logger(args, os.path.join(args.logdir, "args.txt"))

    lossv, accv = [], []
    main(args, lossv, accv)

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, args.epochs + 1), lossv)
    plt.title('validation loss')
    plt.savefig(os.path.join(args.logdir, 'validation_loss'))

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, args.epochs + 1), accv)
    plt.title('validation accuracy')
    plt.savefig(os.path.join(args.logdir, 'validation_accuracy'))
