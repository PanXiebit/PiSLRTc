import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time


def mask_local_mask(size, local_ws=16):
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(local_ws/2)) | (1 - torch.triu(tmp, diagonal=-int(local_ws/2-1)))
    return (1 - mask).unsqueeze(0)


class TemporalAttention3(nn.Module):
    def __init__(self, feat_dim=512, window_size=12, dropout=0.2):
        super(TemporalAttention3, self).__init__()
        self.feat_dim = feat_dim
        self.window_size = window_size
        self.relu = nn.ReLU()
        # self.k_tcn = TemporalConvNet(feat_dim, [feat_dim, feat_dim], kernel_size=3, dropout=dropout)
        self.rnn = nn.GRU(feat_dim, feat_dim)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):
        """
        :param x:  [batch, t, 512]
        :return:  [batch, t, 512]
        """
        scores = torch.matmul(x, x.transpose(-2, -1)) / math.sqrt(self.feat_dim)  # [batch, t, t]
        local_mask = mask_local_mask(size=x.size(1), local_ws=2 * self.window_size).to(x.device) # [batch, t, t]

        scores = scores.masked_fill(local_mask == 0, -1e9)  # [batch, t, t]
        ids = scores.topk(k=self.window_size, dim=-1)[1].sort(-1)[0].detach_()  # require_grad=False, [batch, t, k]

        # print(ids)

        feature = []
        for i in range(x.size(0)):  # batch
            batch_t = []
            for j in range(x.size(1)):  # t
                t = x[i].index_select(0, ids[i, j, :]).unsqueeze(0)  # [1, k, 512]
                batch_t.append(t)
            batch_t = torch.cat(batch_t, dim=0).unsqueeze(0)   # [1, t, k, 512]
            feature.append(batch_t)
        feature = torch.cat(feature, dim=0) # [bs, t, k, 512]
        # print("feature: ", feature.shape)
        feature = feature.reshape(-1, self.window_size, self.feat_dim)  # [bs*t, k, 512]

        # tcn
        # feature = feature.permute(0, 2, 1).contiguous()  # [bs*t, 512, k]
        # out = self.k_tcn(feature) # [bs*t, 512, k]

        # rnn
        feature = feature.permute(1,0,2).contiguous()  # [k, bs*t, 512]
        _, out = self.rnn(feature)  # [1, bs*t, 512]
        # print(out.shape)

        out = out.squeeze().reshape(x.size(0), x.size(1), -1)  # [batch, t, 512]
        del feature

        # residual connection
        out += x
        return out, scores
