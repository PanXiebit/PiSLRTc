import torch
import torch.nn as nn
from torch.nn import Parameter

class RelativePosition(nn.Module):

    def __init__(self, opts):
        super().__init__()
        self.hidden_size = int (opts.hidden_size / opts.num_heads)
        self.max_relative_position = opts.max_relative_position
        self.embeddings_table = Parameter(torch.Tensor(opts.max_relative_position * 2 + 1, self.hidden_size))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        # print("distance_mat_clipped: \n", distance_mat)
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        # print("distance_mat_clipped: \n", distance_mat_clipped)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        # print("final_mat: \n", final_mat)
        embeddings = self.embeddings_table[final_mat].cuda()
        return embeddings


