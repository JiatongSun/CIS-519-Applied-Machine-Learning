import numpy as np
import gym
import torch
import torchvision.transforms as tf
from torch.utils.data import Dataset, DataLoader
from os import path
from glob import glob

from args import Args
from resize_observation import ResizeObservation

        
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

def load_dataset(args, observations, actions, batch_size=64, data_transforms=None, num_workers=0):
    class dataset(Dataset):
        def __init__(self,transform,observations,actions):
            self.observations = observations
            self.actions = actions
            self.transform = transform
    
        def __len__(self):
            return len(self.observations)
    
        def __getitem__(self,idx):
            observations = self.transform(self.observations).float()
            actions = torch.tensor(self.actions.reshape(-1))
            item = {'observations': observations[:,idx], 
                    'actions': actions[idx]}
            return item
        
    if data_transforms is None:
        transform = args.transform
    else:
        transform = data_transforms
    
    train_dataset = dataset(transform,observations,actions)
    
    dataloader = DataLoader(train_dataset, 
                            num_workers=num_workers, 
                            batch_size=batch_size)
    return dataloader

def process_individual_observation(args,observation):
    
    data = args.transform(observation.reshape(1,-1)).float()
    return data


if __name__ == '__main__':
    args = Args()
    expert_observations, expert_actions = load_initial_data(args)
    dataloader = load_dataset(args, expert_observations, expert_actions)