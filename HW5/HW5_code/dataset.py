import torch
import torchvision
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from os import path

LABEL_NAMES = {'background':0, 'kart':1, 'pickup':2, 'nitro':3, 'bomb':4, 'projectile':5}

LABEL_=['background','kart','pickup','nitro','bomb','projectile']

class SuperTuxDataset(Dataset):
    def __init__(self, image_path,data_transforms=None):
        """
        Your code here
        Hint: Use the python csv library to parse labels.csv
        """
        
        self.df = pd.read_csv(path.join(image_path,'labels.csv'), header=0)
        self.path = image_path
        self.files = self.df.iloc[:,0].values
        self.labels = self.df.iloc[:,1].astype('category').cat.codes.values
        self.tracks = self.df.iloc[:,2].values
        
        if data_transforms is None:
            self.transform = torchvision.transforms.Compose([
                          torchvision.transforms.ToTensor()
            ])
        else:
            self.transform = data_transforms
        
        
    def __len__(self):
        """
        Your code here
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.files[idx]
        image = self.transform(Image.open(path.join(self.path,img_name)))
        label = self.labels[idx]
        sample = (image, label)
        return sample


def visualize_data():

    Path_to_your_data= 'data/train/'
    dataset = SuperTuxDataset(image_path=Path_to_your_data)

    f, axes = plt.subplots(3, len(LABEL_NAMES))

    counts = [0]*len(LABEL_NAMES)

    for img, label in dataset:
        c = counts[label]

        if c < 3:
            ax = axes[c][label]
            ax.imshow(img.permute(1, 2, 0).numpy())
            ax.axis('off')
            ax.set_title(LABEL_[label])
            counts[label] += 1
        
        if sum(counts) >= 3 * len(LABEL_NAMES):
            break

    plt.show()

if __name__ == '__main__':
    visualize_data()