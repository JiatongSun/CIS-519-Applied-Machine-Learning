from torch import save
from torch import load
from os import path
from model import CNNClassifier

def save_model(model):
    return save(model.state_dict(), 
            path.join(path.dirname(path.abspath(__file__)), 'cnn.th'))
    
def load_model():
    r = CNNClassifier()
    r.load_state_dict(load(
        path.join(path.dirname(path.abspath(__file__)), 'cnn.th'), 
        map_location='cpu'))
    return r