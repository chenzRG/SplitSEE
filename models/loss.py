import torch
import torch.nn as nn
import numpy as np

class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)
    
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
