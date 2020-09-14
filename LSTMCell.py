import torch
import torch.nn as nn
import torch.functional as F


class LSTMCell(nn.Module):

    def __init__(self):
        super(LSTMCell).__init__()
