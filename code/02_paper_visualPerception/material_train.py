# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:13:06 2021

@author: richie bao-paper data processing
ref_material classification:pytorch-material-classification https://github.com/jiaxue1993/pytorch-material-classification
"""
from config import config
import shutil
import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchvision.models import resnet
from tqdm import tqdm

from material_dataset_loader import Dataloder
from material_network import FinetuneNet

# global variable
best_pred = 100.0
errlist_train = []
errlist_val = []

# refer to https://github.com/xternalz/WideResNet-pytorch
def save_checkpoint(state, config, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    if not os.path.exists(config.snapshot_dir):
        os.makedirs(config.snapshot_dir)
    filename = config.snapshot_dir + filename
    print(filename)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, config.snapshot_dir + 'model_best.pth.tar')

def adjust_learning_rate(optimizer, config, epoch):
    lr = config.lr * (0.1 ** ((epoch - 1) // config.lr_decay))
    if (epoch - 1) % config.lr_decay == 0:
        print('LR is set to {}'.format(lr))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
 
def train(epoch,model,optimizer,train_loader,criterion):
    model.train()
    global best_pred, errlist_train
    train_loss, correct, total = 0, 0, 0
    adjust_learning_rate(optimizer, config, epoch)
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(train_loader)), file=sys.stdout,bar_format=bar_format)
    data_loader = iter(train_loader)

    for batch_idx in pbar:
        data, target = data_loader.next()
        if config.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum().numpy()
        total += target.size(0)
        err = 100 - 100. * correct / total
        print_str = 'Epoch: {}/{}  '.format(epoch, config.nepochs) \
                    + 'Iter: {}/{}  '.format(batch_idx, len(train_loader)) \
                    + 'Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                    (train_loss / (batch_idx + 1),
                     err, total - correct, total)
        pbar.set_description(print_str)
    errlist_train += [err]        

def test(epoch,model,optimizer,test_loader,criterion):
    model.eval()
    global best_pred, errlist_train, errlist_val
    test_loss, correct, total = 0, 0, 0
    is_best = False
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(len(test_loader)), file=sys.stdout,bar_format=bar_format)
    data_loader = iter(test_loader)
    with torch.no_grad():
        for batch_idx in pbar:
            data, target = data_loader.next()
            if config.cuda:
                data, target = data.cuda(), target.cuda()
            output = model(data)
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum().numpy()
            total += target.size(0)

            err = 100 - 100. * correct / total
            print_str = 'Epoch: {}/{}  '.format(epoch, config.nepochs) \
                        + 'Iter: {}/{}  '.format(batch_idx, len(test_loader)) \
                        + 'Loss: %.3f | Err: %.3f%% (%d/%d)' % \
                        (test_loss / (batch_idx + 1),
                         err, total - correct, total)
            pbar.set_description(print_str)

    if config.eval:
        print('Error rate is %.3f' % err)
        return
    # save checkpoint
    errlist_val += [err]
    if err < best_pred:
        best_pred = err
        is_best = True
    print('Best Accuracy: %.3f' % (100 - best_pred))
    save_checkpoint({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'best_pred': best_pred,
        'errlist_train': errlist_train,
        'errlist_val': errlist_val,
    }, config, is_best=is_best)
    

def main():
    # init the config
    global best_pred, errlist_train, errlist_val
    config.cuda = config.cuda and torch.cuda.is_available()
    torch.manual_seed(config.seed)
    if config.cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
        torch.cuda.manual_seed(config.seed)
    # init dataloader
    classes, train_loader, test_loader = Dataloder(config).getloader()    

    # init the model
    backbone = resnet.resnet18(pretrained=False) 
    model = FinetuneNet(len(classes), backbone)
    # print(model)
    # criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay)
    if config.cuda:
        model.cuda()
        # Please use CUDA_VISIBLE_DEVICES to control the number of gpus
        model = torch.nn.DataParallel(model)    

    # check point
    if config.resume is not None:
        if os.path.isfile(config.resume):
            print("=> loading checkpoint '{}'".format(config.resume))
            checkpoint = torch.load(config.resume)
            config.start_epoch = checkpoint['epoch'] + 1
            best_pred = checkpoint['best_pred']
            errlist_train = checkpoint['errlist_train']
            errlist_val = checkpoint['errlist_val']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})".format(config.resume, checkpoint['epoch']))
        else:
            print("=> no resume checkpoint found at '{}'".format(config.resume))
                
    if config.eval:
        test(config.start_epoch,model,optimizer,test_loader,criterion)
        return

    for epoch in range(config.start_epoch, config.nepochs + 1):
        print('Epoch:', epoch)
        train(epoch,model,optimizer,train_loader,criterion)
        test(epoch,model,optimizer,test_loader,criterion)                    

if __name__ == "__main__":
    main()