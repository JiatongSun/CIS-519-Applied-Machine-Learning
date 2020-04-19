import torch.nn as nn

class StatesNetwork(nn.Module):
    def __init__(self, env):
        super(StatesNetwork,self).__init__()
        num_observation = env.observation_space.shape[0]
        num_action = env.action_space.n
        self.fc = nn.Sequential(
            nn.Linear(num_observation,50),
            nn.ReLU(True),
            nn.Linear(50,num_action)
        )
    
    def forward(self, x):    
        x = x.view(x.size(0),-1)
        forward_pass = self.fc(x)
        return forward_pass