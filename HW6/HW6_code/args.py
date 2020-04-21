import gym
import numpy as np
import torch
import torchvision.transforms as tf

from resize_observation import ResizeObservation
from states_network import StatesNetwork

class Args(object):
    def  __init__(self):
        # model
        self.env = ResizeObservation(gym.make('MountainCar-v0'), 100)
        self.model = StatesNetwork(self.env)
        self.transform = tf.Compose([tf.ToTensor()])
        
        # data & log
        self.datapath = './data'
        self.log_dir = './logdir'
        
        # expert
        self.Q = np.load('./expert_Q.npy')
        self.discretization = np.array([10,100])
        
        # train
        self.initial_episodes_to_use = 20
        self.batch_size = 128
        self.lr = 0.1
        self.num_epochs = 2
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.lr)
        self.criterion = torch.nn.CrossEntropyLoss() 
        
        # dagger
        self.do_dagger = False
        if self.do_dagger:
            self.max_dagger_iterations = 18
        else:
            self.max_dagger_iterations = 1
        
        # valid
        self.record_frames = True
        self.num_valid_episode = 5

def get_args():
    args = Args()
    return args

if __name__ == '__main__':
    args = Args()
        
        