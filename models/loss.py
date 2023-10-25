import torch
import torch.nn as nn
import numpy as np

class swavloss(torch.nn.Module):

    def __init__(self, device, final_out_channels):
        super(swavloss, self).__init__()
        self.device = device
        self.prototypes = nn.Sequential(
            nn.Linear(final_out_channels, final_out_channels // 2),
            nn.BatchNorm1d(final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(final_out_channels // 2, final_out_channels // 4),
        )
        self.softmax = torch.nn.Softmax(dim=1).to(self.device)
        self.temperature = 0.1

    def sinkhorn(self, scores, eps = 0.05, nmb_iters =3):
        with torch.no_grad():
            Q = torch.exp(scores / eps).T
            sum_Q = torch.sum(Q)
            Q /= sum_Q
            
            r = torch.ones(Q.shape[0]).to(self.device, non_blocking=True) / Q.shape[0]
            c = torch.ones(Q.shape[1]).to(self.device, non_blocking=True) / (-1 * Q.shape[1])

            curr_sum = torch.sum(Q, dim=1)

            for it in range(nmb_iters):
                u = curr_sum
                Q *= (r / u).unsqueeze(1)
                Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
                curr_sum = torch.sum(Q, dim=1)
            return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()

    def forward(self, c_t,c_f):
        assign_t = self.prototypes(c_t) ##add after T encoder
        assign_f = self.prototypes(c_f) ##add after F encoder
        q_t = self.sinkhorn(assign_t)
        q_f = self.sinkhorn(assign_f)
        p_t = self.softmax(q_t / self.temperature)
        p_f = self.softmax(q_f / self.temperature)
        
        swavloss = -0.5 * torch.mean(q_t * torch.log(p_f) + q_f * torch.log(p_t))
        return swavloss
