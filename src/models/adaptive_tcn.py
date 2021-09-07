import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math



def non_local_mask(size, local_ws=16):
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(local_ws/2 + 1)) | (1 - torch.triu(tmp, diagonal=-int(local_ws/2-1)))
    
    mask = (1 - mask) - torch.eye(size).long()    
    return mask.unsqueeze(0)

def get_rl_mask(size, local_ws=16):
    tmp = torch.ones(size, size).long()
    r_mask = torch.triu(tmp, diagonal=int(local_ws/2+1)) | (1 - torch.triu(tmp, diagonal=1))
    l_mask = torch.triu(tmp, diagonal=0) | (1 - torch.triu(tmp, diagonal=-int(local_ws/2-1)))
    return (1 - r_mask).unsqueeze(0), (1 - l_mask).unsqueeze(0)

    
class AdaptiveTemporalConv(nn.Module):
    def __init__(self, feat_dim=512, window_size=16):
        super(AdaptiveTemporalConv, self).__init__()
        self.feat_dim = feat_dim
        self.window_size = window_size
        self.rel_embedding = nn.Embedding(2 * window_size, feat_dim)
        self.tcn = TemporalConv(feat_dim)
        self.eps = 1e-8

    def forward(self, x):
        """
        :param x:  [batch, t, 512]
        :return:  [batch, t, 512]
        """
        bs, num_f, _ = x.size()
        scores = torch.matmul(x, x.transpose(-2, -1))
        scores /= (torch.sqrt((x.transpose(-2, -1) ** 2).sum(1)).unsqueeze(1) * torch.sqrt((x ** 2).sum(-1)).unsqueeze(-1)) + self.eps

        local_mask = non_local_mask(size=x.size(1), local_ws=2 * self.window_size).to(x.device)
        scores = scores.masked_fill(local_mask == 0, -math.inf)  # [batch, t, t]
        scores = scores.softmax(-1)

        rmask, lmask = get_rl_mask(size=x.size(1), local_ws=2 * self.window_size) # [batch, t, t]
        r_score = (scores * rmask.to(x.device).float()).sum(-1)
        r_span = (self.window_size * r_score).long()

        r_max = torch.clamp(torch.arange(num_f).unsqueeze(0).repeat(bs, 1).to(x.device) + r_span, 0, num_f)
        l_min = torch.clamp(r_max - self.window_size + 1, 0, num_f)
        
        min_span = torch.min((r_max - l_min))

        ids = torch.arange(num_f).unsqueeze(0).repeat(num_f, 1).unsqueeze(0).repeat(bs, 1, 1).to(x.device)
        rmask = (r_max.unsqueeze(-1) - ids) >= 0
        lmask = (ids - l_min.unsqueeze(-1)) >= 0
        ids = ids * (rmask & lmask).long()
        ids = ids.topk(k=min_span, dim=-1)[1].sort(-1)[0].detach_().long()  # require_grad=False, [batch, t, k]

        new_feature = []
        for i in range(bs):  # batch
            t = x[i, ids[i], :]  # [t, k, 512]
            new_feature.append(t.unsqueeze(0))
        new_feature = torch.cat(new_feature, dim=0)
        relative_pos = ids - torch.arange(num_f).unsqueeze(0).unsqueeze(-1).to(x.device)  # [bs, t, k]

        att_span = self.window_size
        relative_pos = torch.clamp(relative_pos + att_span, 0, (att_span * 2) - 1) # [bs, t, k]

        pos_emb = self.rel_embedding(relative_pos)  # [bs, t, k, 512]
        new_feature = new_feature + pos_emb
        new_feature = new_feature.view(bs*num_f, min_span, self.feat_dim) # [bs*t, k, 512]

        out = self.tcn(new_feature.permute(0, 2, 1).contiguous())
        out = out.view(bs, num_f, self.feat_dim)
        return out
        

class TemporalConv(nn.Module):
    def __init__(self, feat_dim, kernel_sizes=[3, 3]):
        super(TemporalConv, self).__init__()
        """
        kernel_size: list, kernel size along temporal dimension
        num_heads: d/H in "pay less attention" paper
        conv_type: "lightweight" or "dynamic"
        conv_dim: the same as input dim
        use_glu:
        weight_dropout_rate:
        """
        self.conv1 = nn.Conv1d(feat_dim, feat_dim, kernel_size=kernel_sizes[0], padding=0, groups=feat_dim)
        self.ln1 = nn.LayerNorm(feat_dim)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(feat_dim, feat_dim, kernel_size=kernel_sizes[1], padding=0, groups=feat_dim)
        self.ln2 = nn.LayerNorm(feat_dim)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x):
        """
        x: [bs, hs, t]
        """
        x = self.conv1(x).permute(0, 2, 1).contiguous()
        x = self.ln1(x).permute(0, 2, 1).contiguous()
        x = self.relu(x)
        x = self.conv2(x).permute(0, 2, 1).contiguous()
        x = self.ln2(x).permute(0, 2, 1).contiguous()
        x = self.relu(x)
        x = self.max_pool(x)
        return x.squeeze(-1)


if __name__ == "__main__":
    m = AdaptiveTemporalConv(1024, 16).cuda()
    x = torch.randn(2, 100, 1024).cuda()
    out = m(x)
    print(out.shape)


