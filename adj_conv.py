import torch
import torch.nn as nn

class NLPConv(nn.Conv2d):
    def forward(self, weights, input_tensor):
        self.weight = nn.Parameter(weights)
        return super().forward(input_tensor)
