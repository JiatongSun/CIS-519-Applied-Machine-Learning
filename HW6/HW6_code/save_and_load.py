from torch import save
from torch import load
from os import path
from states_network import StatesNetwork

def save_model(model):
    return save(model.state_dict(), 
            path.join(path.dirname(path.abspath(__file__)), 
                      'states_net.th'))
    
def load_model(env):
    r = StatesNetwork(env)
    r.load_state_dict(load(
        path.join(path.dirname(path.abspath(__file__)), 'states_net.th'), 
        map_location='cpu'))
    return r