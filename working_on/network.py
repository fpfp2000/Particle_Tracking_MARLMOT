"""
Generic Feedforward NN for actor and Critic
"""

import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self, input_dim=18, output_dim=5):
        """
            Input: (batch_size, 18)
            Output: (batch_size, 5)
        """
        super(Net, self).__init__()

        self.fc1 = nn.Linear(input_dim, 128)  
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, output_dim)

    def forward(self, x):

        # if torch.isnan(x).any() or torch.isinf(x).any():
        #     nan_cols = torch.isnan(x).any(dim=0)
        #     print(f"Columns with NaN values {nan_cols.nonzero()}")
        #     raise ValueError("Invalid input: NaN or Inf values detected")
        # print(f"Input chape to network is: {x.shape}")
        x = F.elu(self.fc1(x))  
        x = F.elu(self.fc2(x))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)

        # print(f"Logits: {x}")
        # return logits and compute softmax with loss function?
        return x
