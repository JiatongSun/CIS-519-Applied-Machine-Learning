import torch
import numpy as np

class ClassificationLoss(torch.nn.Module):
    def forward(self, input, target):
        """
        Your code here
        Compute mean(-log(softmax(input)_label))
        @input:  torch.Tensor((B,C)), where B = batch size, C = number of classes
        @target: torch.Tensor((B,), dtype=torch.int64)
        @return:  torch.Tensor((,))
        Hint: Don't be too fancy, this is a one-liner
        """
        input_exp = np.exp(input)
        input_exp /= input_exp.sum(axis=1).reshape(-1,1)
        return -np.log(input_exp[np.arange(len(input_exp)),target])