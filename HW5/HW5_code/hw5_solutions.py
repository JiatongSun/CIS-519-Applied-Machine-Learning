import torch
import torchvision

import torchvision.transforms.functional as TF
import torch.nn.functional as F

from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from os import path
import time
import copy

class SuperTuxDataset(Dataset):
    def __init__(self, image_path,data_transforms=None):
        self.df = pd.read_csv(path.join(image_path,'labels.csv'), header=0)
        self.path = image_path
        self.files = self.df.iloc[:,0].values
        self.labels = self.df.iloc[:,1].astype('category').cat.codes.values
        self.tracks = self.df.iloc[:,2].values
        
        if data_transforms is None:
            self.transform = torchvision.transforms.Compose([
                          torchvision.transforms.ToTensor()
            ])
        else:
            self.transform = data_transforms
        
        
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = self.transform(Image.open(path.join(self.path,img_name)))
        label = self.labels[idx]
        sample = (image, label)
        return sample

class ClassificationLoss(torch.nn.Module):
    def forward(self, inputs, target):
        X = torch.exp(inputs)/torch.exp(inputs).sum(axis=1).reshape(-1,1)
        Y = -torch.log(X[torch.arange(len(target)),target.long()])
        loss = torch.sum(Y)/len(target)
        return loss

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        super(CNNClassifier,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.BatchNorm2d(6),
            torch.nn.Conv2d(6,12,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.BatchNorm2d(12),
            torch.nn.Conv2d(12,24,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(24,24,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.Conv2d(24,20,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1280,400),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.15),
            torch.nn.Linear(400,100),
            torch.nn.ReLU(True),
            torch.nn.Dropout2d(0.15),
            torch.nn.Linear(100,6)
        )

    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

def test_logging(train_logger, valid_logger):
    global_step = 0
    size_train = 20
    size_valid = 10
    for epoch in range(10):
        torch.manual_seed(epoch)
        train_acc = []
        valid_acc = []
        for iteration in range(size_train):
            dummy_train_loss = 0.9**(epoch+iteration/20.)
            dummy_train_accuracy = epoch/10. + torch.randn(10)
            train_logger.add_scalar('train loss',
                                    dummy_train_loss,
                                    global_step)
            train_acc.append(torch.mean(dummy_train_accuracy))
            global_step += 1
        
        train_acc_iter = torch.mean(torch.FloatTensor(train_acc))
        train_logger.add_scalar('train accuracy',
                                train_acc_iter,
                                epoch)
        
        torch.manual_seed(epoch)
        for iteration in range(size_valid):
            dummy_validation_accuracy = epoch / 10. + torch.randn(10)
            valid_acc.append(torch.mean(dummy_validation_accuracy))
            
        valid_acc_iter = torch.mean(torch.FloatTensor(valid_acc))
        valid_logger.add_scalar('valid accuracy',
                                valid_acc_iter,
                                epoch)