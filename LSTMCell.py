import torch
import torch.nn as nn


class LSTMCell(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(LSTMCell, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_f = nn.Linear(in_dim, out_dim)
        self.U_f = nn.Linear(out_dim, out_dim)

        self.W_i = nn.Linear(in_dim, out_dim)
        self.U_i = nn.Linear(out_dim, out_dim)

        self.W_o = nn.Linear(in_dim, out_dim)
        self.U_o = nn.Linear(out_dim, out_dim)

        self.W_c = nn.Linear(in_dim, out_dim)
        self.U_c = nn.Linear(out_dim, out_dim)

    def forward(self, x, hx=None):
        if hx is None:
            zeros = torch.zeros(x.size(0), self.out_dim, dtype=x.dtype, device=x.device)
            hx = (zeros, zeros)

        h, c = hx

        f = torch.sigmoid(self.W_f(x) + self.U_f(h))
        i = torch.sigmoid(self.W_i(x) + self.U_i(h))
        o = torch.sigmoid(self.W_o(x) + self.U_o(h))
        c_tag = torch.tanh(self.W_c(x) + self.U_c(h))

        c = f * c + i * c_tag
        h = o * torch.tanh(c)

        return h, (c, h)


if __name__ == "__main__":
    test_in_dim = 100
    test_out_dim = 1000
    batch_size = 50

    lstm_cell = LSTMCell(test_in_dim, test_out_dim)

    rand = torch.rand(batch_size, test_out_dim)
    hx = (rand, rand)

    test_tensor = torch.rand((batch_size, test_in_dim))

    h_test, hx_test = lstm_cell(test_tensor, hx)

    print(h_test.size())
    print(hx_test[0].size())
    print(hx_test[1].size())
