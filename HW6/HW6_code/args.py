import gym
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
        self.data_path = './data'
        self.log_dir = './logdir'
        
        # train
        self.num_expert_episode = 100
        self.batch_size = 64
        self.lr = 0.005
        self.epoch = 5
        self.optimizer = torch.optim.Adam(self.model.parameters(),self.lr)
        self.criterion = torch.nn.CrossEntropyLoss() 
        
        # dagger
        self.do_dagger = False
        self.max_dagger_iterations = 1
        
        # valid
        self.record_frames = False
        self.num_valid_episode = 5
    
        
        