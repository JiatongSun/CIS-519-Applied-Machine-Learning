import torch.nn as nn

class StatesNetwork(nn.Module):
    def __init__(self, env):
        super(StatesNetwork,self).__init__()
        self.observation_space = env.observation_space.high -\
                                 env.observation_space.low
        self.action_space = env.action_space.high -\
                            env.action_space.low
        self.fc = nn.Sequential(
            nn.Linear(64,128),
            nn.ReLU(True),
            nn.Linear(128,128),
            nn.ReLU(True),
            nn.Linear(128,64)
        )
    
    def forward(self, x):    
        forward_pass = self.fc(x)
        return forward_pass