import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
import time
import os
import copy
from tqdm import tqdm

from dataset import SuperTuxDataset
from model import ClassificationLoss, CNNClassifier
from save_and_load import save_model, load_model

if __name__ == '__main__':
    data_transforms = {
        'train': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
        'valid': torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ]),
    }
    
    data_dir = 'data'
    image_datasets = {x: SuperTuxDataset(os.path.join(data_dir, x),
                                         data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=64,
                                 shuffle=True)
                  for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = 'cpu'
    
    since = time.time()
    
    model = CNNClassifier().to(device)
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    num_epochs=25
    criterion = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr=0.05)
        
    for epoch in range(num_epochs):
        print('\nEpoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase],position=0):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, axis=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('\n{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    save_model(model)

