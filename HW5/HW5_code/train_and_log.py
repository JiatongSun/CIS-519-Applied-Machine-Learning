import torch
import torchvision
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
from os import path
import time
import copy
from tqdm import tqdm

from dataset import SuperTuxDataset
from model import ClassificationLoss, CNNClassifier
from save_and_load import save_model, load_model


def load_data(dataset_path, data_transforms=None, 
              num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path,data_transforms)
    return DataLoader(dataset, num_workers=num_workers, 
                      batch_size=batch_size, shuffle=True)

class Args(object):
    def  __init__(self):
        self.learning_rate = 0.005
        self.momentum = 0.9
        self.step_size = 5
        self.gamma = 0.5
        self.log_dir = './logdir'
        self.epoch = 200
        self.data_dir = 'data'
        self.data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]),
      }

def train(args):
    data_transforms = args.data_transforms
    data_dir = args.data_dir
    
    image_datasets = {x: SuperTuxDataset(path.join(data_dir, x),
                                         data_transforms[x])
                      for x in ['train', 'valid']}
    dataloaders = {x: DataLoader(image_datasets[x], 
                                 batch_size=64,
                                 shuffle=True)
                  for x in ['train', 'valid']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    since = time.time()
    
    model = CNNClassifier().to(device)
    # model = load_model()
    
    best_epoch = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    num_epochs=args.epoch
    criterion = ClassificationLoss()
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum = args.momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                        step_size = args.step_size,
                                        gamma = args.gamma)
    
    global_step_train = 0
    global_step_valid = 0
    
    train_flag = True
    
    if args.log_dir is not None:
        train_logger = tb.SummaryWriter(path.join(args.log_dir, 'train'))
        valid_logger = tb.SummaryWriter(path.join(args.log_dir, 'valid'))
    
    for epoch in range(num_epochs):
        if train_flag is False:
            print('stop training!')
            break
        
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
                
                if phase == 'train':
                    train_logger.add_scalar('train loss',
                                            loss.item(),
                                            global_step_train)
                    global_step_train += 1
                elif phase == 'valid':
                    valid_logger.add_scalar('valid loss',
                                            loss.item(),
                                            global_step_valid)
                    global_step_valid += 1
            
            if phase == 'train':
                scheduler.step()
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            
            if phase == 'train':
                train_logger.add_scalar('train accuracy',
                                        epoch_acc,
                                        epoch)
            elif phase == 'valid':
                valid_logger.add_scalar('valid accuracy',
                                        epoch_acc,
                                        epoch)

            print('\n{} Loss: {:.4f} Acc: {:.4f}\n'.format(
                phase, epoch_loss, epoch_acc))

            if phase == 'valid' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            
            if epoch-best_epoch > 5:
                print('Early stopping!')
                train_flag = False


    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid Acc: {:4f}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    save_model(model)
    
if __name__ == '__main__':
    # %load_ext tensorboard
    # %reload_ext tensorboard
    # %tensorboard --logdir="./logdir" --host=127.0.0.1
    args = Args()
    train(args)