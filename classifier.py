import numpy as np
import json
import time
import os
import random
# import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
from torchvision import datasets, transforms, models

class Classifier(nn.Module):
    
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Add input layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Add hidden layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Add output layer
        self.output = nn.Linear(hidden_layers[-1], output_size)

        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)

    def forward(self, x):
        # Flaten tensor input
        x = x.view(x.shape[0], -1)

        # Add dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))        

        # Output so no dropout here
        x = F.log_softmax(self.output(x), dim=1)

        return x
    