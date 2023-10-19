import torch
import torch.nn as nn
import numpy as np
from .attention import Seq_Transformer



class FC(nn.Module):
    def __init__(self, configs, device):
        super(FC, self).__init__()
        self.num_channels = configs.final_out_channels
        self.f_step = configs.TC.f_steps
        self.Wk = nn.ModuleList([nn.Linear(configs.TC.hidden_dim, self.num_channels) for i in range(self.f_step)])
        self.lsoftmax = nn.LogSoftmax()
        self.device = device

        self.seq_transformer = Seq_Transformer(patch_size=self.num_channels, dim=configs.TC.hidden_dim, depth=4, heads=4, mlp_dim=64)

    def forward(self, features):# features are (batch_size, #channels, seq_len)
        seq_len = features.shape[2]
        features = features.transpose(1, 2)

        batch = features.shape[0]
        t_samples = torch.randint(seq_len - self.f_step, size=(1,)).long().to(self.device)  # randomly pick time stamps
        nce = 0  # average over f_step and batch
        encode_samples = torch.empty((self.f_step, batch, self.num_channels)).float().to(self.device)

        for i in np.arange(1, self.f_step + 1):
            encode_samples[i - 1] = features[:, t_samples + i, :].view(batch, self.num_channels)
        forward_seq = features[:, :t_samples + 1, :]

        c_t = self.seq_transformer(forward_seq)

        pred = torch.empty((self.f_step, batch, self.num_channels)).float().to(self.device)
        for i in np.arange(0, self.f_step):
            linear = self.Wk[i]
            pred[i] = linear(c_t)
        for i in np.arange(0, self.f_step):
            total = torch.mm(encode_samples[i], torch.transpose(pred[i], 0, 1))
            nce += torch.sum(torch.diag(self.lsoftmax(total)))
        nce /= -1. * batch * self.f_step
        return nce, c_t, self.seq_transformer(features)