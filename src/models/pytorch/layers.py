import torch
import torch.nn as nn
import torch.nn.functional as F


class ExtendMask(nn.Module):

    def __init__(self, add_epsilon=False):
        super(ExtendMask, self).__init__()
        self.add_epsilon = add_epsilon

    def forward(self, x, mask):
        return x

    def compute_mask(self, mask):
        if self.add_epsilon:
            return mask + torch.finfo(torch.float32).eps
        return mask
