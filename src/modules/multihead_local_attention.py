import math
import torch
import torch.nn as nn
from torch import Tensor
from configs.options import get_parser

config = get_parser()

WINDOW_SIZE = config.window_size


def mask_non_local_mask(size, local_ws=16):
    """

    :param size:
    :param local_ws:
    :return:
    tensor([[[1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
         [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
         [0, 1, 1, 1, 1, 1, 0, 0, 0, 0],
         [0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
         [0, 0, 0, 1, 1, 1, 1, 1, 0, 0],
         [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
         [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
         [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]])
    """
    tmp = torch.ones(size, size).long()
    mask = torch.triu(tmp, diagonal=int(local_ws/2)) | (1 - torch.triu(tmp, diagonal=-int(local_ws/2-1)))
    return (1 - mask).unsqueeze(0)  # [1, length, length]

# res = mask_non_local_mask(10, local_ws=6)
# print(res.shape, "\n", res)
# exit()

# pylint: disable=arguments-differ
class MultiHeadedLocalAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """

    def __init__(self, opts, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedLocalAttention, self).__init__()
        self.is_adaptive = opts.is_adaptive
        assert size % num_heads == 0

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size, num_heads * head_size)
        self.v_layer = nn.Linear(size, num_heads * head_size)
        self.q_layer = nn.Linear(size, num_heads * head_size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads

        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        k = k.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        v = v.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)
        q = q.view(batch_size, -1, num_heads, self.head_size).transpose(1, 2)

        # compute scores
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        scores = torch.matmul(q, k.transpose(2, 3))

        # print("scores: ", scores.shape, scores[0, 0, :10, :10])
        # print("mask: ", mask.shape, mask.unsqueeze(1).shape, mask[:, 0, -50:])
        # exit()
        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]

        nonlocal_mask = mask_non_local_mask(size=scores.size(2), local_ws=2 * WINDOW_SIZE).to(scores.device)

        if mask is not None:
            # print("scores: ", scores.shape)
            # print("mask: ", mask.unsqueeze(1).shape)
            scores = scores.masked_fill(nonlocal_mask.unsqueeze(1) == 0, -1e9)
            scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))  # [bs, head, q_len, k_len]
            if self.is_adaptive:
                topk_scores, topk_ids = scores.topk(k=WINDOW_SIZE, dim=-1, largest=True)  # [bs, head, q_len, k]
                threshod = topk_scores[:, :, :, -1].unsqueeze(-1).detach()  # [bs, head, q_len, 1]
                scores = scores.masked_fill(scores < threshod, -1e9)

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores) # [bs, head, q_len, k_len]
        # print("attention: ", attention[0, 0, :50, :50])
        attention = self.dropout(attention)

        # get context vector (select values with attention) and reshape
        # back to [B, M, D]
        context = torch.matmul(attention, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, num_heads * self.head_size)
        )

        output = self.output_layer(context)

        return output