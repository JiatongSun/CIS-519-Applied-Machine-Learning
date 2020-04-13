import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torch.utils.tensorboard as tb
from os import path
import time
import copy
from glob import glob
from tqdm import tqdm


class Args(object):
    def  __init__(self):
        self.data_path = './data'
        self.num_expert_episode = 100
        self.learning_rate = 0.005
        self.momentum = 0.9
        self.step_size = 5
        self.gamma = 0.5
        self.epoch = 200
        self.data_transforms = {
            'train': torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(),
                torchvision.transforms.ToTensor()
            ]),
            'valid': torchvision.transforms.Compose([
                torchvision.transforms.ToTensor()
            ]),
      }
        
def load_initial_data(args):
    num_file = 0
    for file in glob(path.join(args.data_path, '*.npz')):
        if num_file >= args.num_expert_episode: break
        data = np.load(file)
        data_len = len(data['observations'])
        cur_observations = data['observations'].reshape(data_len,-1)
        cur_actions = data['actions'].reshape(data_len,-1)
        if num_file == 0:
            training_observations = cur_observations.copy()
            training_actions = cur_actions.copy()
        else:
            training_observations = np.vstack((training_observations, 
                                               cur_observations))
            training_actions = np.vstack((training_actions,
                                          cur_actions))
        num_file += 1
    return training_observations, training_actions


if __name__ == '__main__':
    args = Args()
    training_observations, training_actions = load_initial_data(args)