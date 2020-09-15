import torch
import torch.nn as nn


class LSTMCell(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(LSTMCell).__init__()

        self.W_f = nn.Linear(in_dim, out_dim)
        self.U_f = nn.Linear(out_dim, out_dim)

        self.W_i = nn.Linear(in_dim, out_dim)
        self.U_i = nn.Linear(out_dim, out_dim)

        self.W_o = nn.Linear(in_dim, out_dim)
        self.U_o = nn.Linear(out_dim, out_dim)

        self.W_c = nn.Linear(in_dim, out_dim)
        self.U_c = nn.Linear(out_dim, out_dim)

    def forward(self, x, c, h):
        f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_tag = torch.tanh(self.W_c(x) + self.U_c(h))

        c = f * c + i * c_tag
        h = o * torch.tanh(c)

        return h, torch.cat((c, h))
