# -*- coding: UTF-8 -*-
"""
@Project ：code 
@File ：main.py.py
@Author ：AnthonyZ
@Date ：2022/6/14 13:31
"""

from data import *
from model import *
from utils import *

import argparse
import time

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def train(args, category_tensor, line_tensor):
    # hidden = rnn.initHidden()
    #
    # rnn.zero_grad()
    # print(line_tensor.shape)
    # for i in range(line_tensor.size()[0]):
    #     output, hidden = rnn(line_tensor[i], hidden)
    optimizer.zero_grad()
    output = rnn(line_tensor)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()

    # Add parameters' gradients to their values, multiplied by learning rate
    # for p in rnn.parameters():
    #     p.data.add_(p.grad.data, alpha=-args.learning_rate)

    return output, loss.item()


def main(args, current_loss):
    correct_num = 0
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(args, category_tensor, line_tensor)
        current_loss += loss

        guess, guess_i = categoryFromOutput(output)
        if guess == category:
            correct_num += 1

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            # guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            all_accurancy.append(correct_num / plot_every)
            correct_num = 0
            current_loss = 0


# Just return an output given a line
def evaluate(line_tensor):
    # hidden = rnn.initHidden()
    # for i in range(line_tensor.size()[0]):
    #     output, hidden = rnn(line_tensor[i], hidden)
    output = rnn(line_tensor)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=4, type=int, help="The batch size of training")
    parser.add_argument("--device", default='mps', type=str, help="The training device")
    parser.add_argument("--learning_rate", default=0.08, type=float, help="learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="Training epoch")
    parser.add_argument("--logdir", default="./log", type=str)
    parser.add_argument("--hidden", default=128, type=int, help="The number of hidden state")

    args = parser.parse_args()

    for i in range(10):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        print('category =', category, '/ line =', line)

    criterion = nn.NLLLoss()

    # rnn = RNN(n_letters, args.hidden, n_categories)=
    rnn = LSTM(n_letters, args.hidden, n_categories)#.to(args.device)
    optimizer = torch.optim.SGD(rnn.parameters(), lr=args.learning_rate)

    current_loss = 0
    all_losses = []
    all_accurancy = []
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    start = time.time()

    main(args, current_loss)

    plt.figure()
    plt.plot(all_losses)
    plt.savefig(
            './validation_loss')
    plt.figure()
    plt.plot(all_accurancy)
    plt.savefig(
            './validation_accuracy')

    confusion = torch.zeros(n_categories, n_categories)
    n_confusion = 10000

    # Go through a bunch of examples and record which are correctly guessed
    for i in range(n_confusion):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output = evaluate(line_tensor)
        guess, guess_i = categoryFromOutput(output)
        category_i = all_categories.index(category)
        confusion[category_i][guess_i] += 1

    # Normalize by dividing every row by its sum
    for i in range(n_categories):
        confusion[i] = confusion[i] / confusion[i].sum()

    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(confusion.numpy())
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + all_categories, rotation=90)
    ax.set_yticklabels([''] + all_categories)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2
    plt.show()
    fig.savefig(
            './1')




