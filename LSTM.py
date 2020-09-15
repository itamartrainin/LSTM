import torch
import torch.nn as nn
from LSTMCell import LSTMCell


class LSTM(nn.Module):
    def __init__(self, dim_in, dim_out, batch_first=False):
        super(LSTM, self).__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.batch_first = batch_first

        self.lstm_cell = LSTMCell(self.dim_in, self.dim_out)

    def forward(self, x, hx=None):
        assert len(x.size()) == 3

        if self.batch_first:
            x = x.transpose(0, 1)

        seq_len = x.size(0)
        batch_size = x.size(1)

        hx_i = hx
        h_out = torch.zeros(seq_len, batch_size, self.dim_out, dtype=x.dtype, device=x.device)
        hx_out = []
        for i, x_i in enumerate(x):
            h_i, hx_i = self.lstm_cell(x_i, hx_i)
            h_out[i] = h_i
            hx_out.append(hx_i)

        return h_out, hx_out


if __name__ == "__main__":
    test_in_dim = 100
    test_out_dim = 1000
    batch_size = 50
    seq_len = 10

    lstm = LSTM(test_in_dim, test_out_dim)

    rand = torch.rand(batch_size, test_out_dim)
    hx_0 = (rand, rand)

    test_tensor = torch.rand((seq_len, batch_size, test_in_dim))

    h_test, hx_test = lstm(test_tensor, hx_0)

    assert h_test.size() == torch.Size([10, 50, 1000])
    assert len(hx_test) == 10
    assert len(hx_test[0]) == 2
    assert hx_test[0][0].size() == torch.Size([50, 1000])
