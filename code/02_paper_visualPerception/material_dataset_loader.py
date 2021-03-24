# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 20:08:42 2021

@author: richie bao-paper data processing
ref_material classification:pytorch-material-classification https://github.com/jiaxue1993/pytorch-material-classification
"""
import os
from config import config
from torchvision import datasets, transforms
from material_augument import Lighting
import torch

class Dataloder():
    def __init__(self, config):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.4,0.4,0.4),
            transforms.ToTensor(),
            Lighting(0.1),
            normalize,
        ])
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])

        trainset = datasets.ImageFolder(os.path.join(config.dataset_path, 'train'), transform_train)
        testset = datasets.ImageFolder(os.path.join(config.dataset_path, 'test'), transform_test)


        kwargs = {'num_workers': 8, 'pin_memory': True} if config.cuda else {}
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        testloader = torch.utils.data.DataLoader(testset, batch_size=config.batch_size, shuffle=False, **kwargs)


        self.trainloader = trainloader 
        self.testloader = testloader
        self.classes = trainset.classes
    
    def getloader(self):
        return self.classes, self.trainloader, self.testloader

if __name__ == "__main__":
    data_dir=config.dataset_path
    trainset=datasets.ImageFolder(os.path.join(data_dir, 'train'))
    classes, train_loader, test_loader=Dataloder(config).getloader()
