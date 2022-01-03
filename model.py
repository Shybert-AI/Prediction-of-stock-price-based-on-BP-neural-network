# -*- coding:utf-8 -*-
import torch.nn as nn
import torch

class Swish(nn.Module):
    """Swish activation function: x * sigmoid(x)"""

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)
    
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size1,hidden_size2, output_size,p=0,active_func="relu"):
        super(NeuralNet, self).__init__()
        self.p = p
        self.active_func = active_func
        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.dropout1 = nn.Dropout(self.p)
        self.relu1 = nn.ReLU() if self.active_func.lower() == "relu" else Swish()
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.dropout2 = nn.Dropout(self.p)
        self.relu2 = nn.ReLU() if self.active_func.lower() == "relu" else Swish()
        self.fc3 = nn.Linear(hidden_size2, 1)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu1(out)
        out = self.dropout1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        out = self.fc3(out)
        return out
    
