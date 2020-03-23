import torch
import numpy as np
import torch.nn.functional as F

class ClassificationLoss(torch.nn.Module):
    def forward(self, inputs, target):
        """
        Your code here
        Compute mean(-log(softmax(input)_label))
        @input:  torch.Tensor((B,C)), where B = batch size, C = number of classes
        @target: torch.Tensor((B,), dtype=torch.int64)
        @return:  torch.Tensor((,))
        Hint: Don't be too fancy, this is a one-liner
        """
        X = torch.exp(inputs)/torch.exp(inputs).sum(axis=1).reshape(-1,1)
        Y = -torch.log(X[torch.arange(len(target)),target.long()])
        loss = torch.sum(Y)/len(target)
        return loss

class CNNClassifier(torch.nn.Module):
    def __init__(self):
        """
        Your code here
        (n+2p-k)/s+1
        """
        super(CNNClassifier,self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,6,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(6,12,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2),
            torch.nn.Conv2d(12,24,kernel_size=3,stride=1,padding=1),
            torch.nn.ReLU(True),
            torch.nn.MaxPool2d(2,2),
            # torch.nn.Conv2d(24,48,kernel_size=3,stride=1,padding=1),
            # torch.nn.ReLU(True),
            # torch.nn.MaxPool2d(2,2)
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(1536,100),
            torch.nn.Linear(100,100),
            torch.nn.Linear(100,6)
        )
        for m in self.modules():
            if isinstance(m,torch.nn.Linear):
                torch.nn.init.normal_(m.weight,mean=0,std=0.1)
                torch.nn.init.constant_(m.bias,0.1)
            if isinstance(m,torch.nn.Conv2d):
                torch.nn.init.normal_(m.weight,mean=0,std=0.1)

    def forward(self, x):
        """
        Your code here
        @x: torch.Tensor((B,3,64,64))
        @return: torch.Tensor((B,6))
        """
        out = self.conv(x)
        out = out.view(out.size(0),-1)
        out = self.fc(out)
        return out

if __name__ == '__main__':
    inputs = torch.Tensor([[0.07,0.22,0.28],[0.35,0.78,1.12]])
    target = torch.Tensor([0,1])
    criterion = ClassificationLoss()
    print(criterion(inputs,target))