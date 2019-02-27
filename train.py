import sys
import os
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim

from eval import eval_net
from unet import UNet

from utils import dataset
# from utils import get_ids, split_ids, split_train_val, get_imgs_and_masks, batch

def train_net(net, epochs=5, batch_size=1, lr=0.1, val_percent=0.2, save_cp=True, gpu=False):

    data_dir = r'E:\workspace\dataset\RVSC\TrainingSet'
    dir_checkpoint = ''


    trainloader = dataset.get_dataloader(data_dir,batch_size,trainset = True, validation_split = val_percent,
                                         mask='inner', shuffle=True, normalize_images=True)

    valloader = dataset.get_dataloader(data_dir, batch_size=1, trainset=False, validation_split=val_percent,
                                         mask='inner', shuffle=True, normalize_images=True)
    _imgs = trainloader.dataset.images
    _imgsv = valloader.dataset.images

    print('''
    Starting training:
        Epochs: {}
        Batch size: {}
        Learning rate: {}
        Training size: {}
        Validation size: {}
        Checkpoints: {}
        CUDA: {}
    '''.format(epochs, batch_size, lr, len(_imgs),
               len(_imgsv), str(save_cp), str(gpu)))

#优化函数
    optimizer = optim.Adam(net.parameters(), lr=lr)

#二进制交叉熵
    criterion = nn.BCELoss()

    N_train = len(_imgs)

    for epoch in range(epochs):
        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        net.train()


        epoch_loss = 0

        for i, (images,masks) in enumerate(trainloader):

            images = images.type(torch.FloatTensor)
            masks = masks.type(torch.FloatTensor)
            if gpu:
                images = images.cuda()
                masks = masks.cuda()

            masks_pred = net(images)
            masks_probs_flat = masks_pred.view(-1)

            true_masks_flat = masks.view(-1)

            loss = criterion(masks_probs_flat, true_masks_flat)
            epoch_loss += loss.item()

            print('{0:.4f} --- loss: {1:.6f}'.format(i * batch_size / N_train, loss.item()))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print('Epoch finished ! Loss: {}'.format(epoch_loss / i))

        if 1:
            val_dice = eval_net(net, valloader, gpu)
            print('Validation Dice Coeff: {}'.format(val_dice))

        if epoch%100==0:
            torch.save(net.state_dict(),
                       dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))


def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=500, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=16,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=0.001,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=True, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')
    parser.add_option('-s', '--scale', dest='scale', type='float',
                      default=0.5, help='downscaling factor of the images')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    args = get_args()

    net = UNet(n_channels=1, n_classes=1)

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        net.cuda()
        # cudnn.benchmark = True # faster convolutions, but more memory

    try:
        train_net(net=net, epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, gpu=args.gpu)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)