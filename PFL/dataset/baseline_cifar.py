#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from PFL.system.flcore.trainmodel.models import FedAvgCNN
from Read_Data import read_total
from method.FedAvg.models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar, VGG
from method.FedAvg.update import test_inference

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def args_parser():
    parser = argparse.ArgumentParser()

    # federated arguments (Notation for the arguments followed from paper)
    parser.add_argument('--epochs', type=int, default=10,
                        help="number of rounds of training")
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate')
    # model arguments
    parser.add_argument('--model', type=str, default='mlp', help='model name')

    # other arguments
    parser.add_argument('--dataset', type=str, default='Cifar10',choices=['mnist', 'Cifar10'], help="name \
                        of dataset")
    parser.add_argument('--num_classes', type=int, default=10, help="number \
                        of classes")
    parser.add_argument('--gpu', default=True)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = args_parser()
    logging.info(torch.cuda.is_available())
    device = 'cuda' if args.gpu else 'cpu'
    logging.info(device)
    # load datasets
    # train_dataset, test_dataset, _ = get_dataset(args)

    train_dataset, test_dataset = read_total(args.dataset)
    # BUILD MODEL
    if args.model == 'cnn':
        if args.dataset == 'Cifar10':
            global_model = FedAvgCNN(in_features=3, num_classes=args.num_classes, dim=1600)
        if args.dataset == 'mnist':
            global_model = FedAvgCNN(in_features=1, num_classes=args.num_classes, dim=1024)
    else:
        exit('Error: unrecognized model')

    # Set the model to train and send it to device.
    global_model.to(device)
    global_model.train()
    logging.info(global_model)

    # Training
    # Set optimizer and criterion
    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr,
                                    momentum=0.5)


    trainloader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    epoch_loss = []

    for epoch in tqdm(range(args.epochs)):
        batch_loss = []

        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = global_model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if batch_idx % 50 == 0:
                logging.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch + 1, batch_idx * len(images), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.item()))
            batch_loss.append(loss.item())

        loss_avg = sum(batch_loss) / len(batch_loss)
        logging.info('\nTrain loss:', loss_avg)
        epoch_loss.append(loss_avg)
        if epoch % 10 == 0:
            params = global_model.state_dict()
            torch.save(params, f'../../save/baseline_{args.model}_{args.dataset}_NonIID_Unbalance_DIR_gr_{epoch}.pth')
    # Plot loss
    plt.figure()
    plt.plot(range(len(epoch_loss)), epoch_loss)
    plt.xlabel('epochs')
    plt.ylabel('Train loss')
    plt.savefig('../../save/nn_{}_{}_{}.png'.format(args.dataset, args.model,
                                                    args.epochs))

    # testing
    test_acc, test_loss = test_inference(args, global_model, test_dataset)
    logging.info('Test on', len(test_dataset), 'samples')
    logging.info("Test Accuracy: {:.2f}%".format(100 * test_acc))

    # saving_model
    params = global_model.state_dict()
    torch.save(params, f'../../save/baseline_{args.model}_{args.dataset}_NonIID_Unbalance_DIR.pth')
