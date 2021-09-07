
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils import freeze_params
from src.modules.position_encoding import PositionalEncoding
from src.models.adaptive_tcn import AdaptiveTemporalConv

import math
import torch
import torch.nn as nn
from torch import Tensor
from src.modules.multihead_local_attention import MultiHeadedLocalAttention
from src.modules.multihead_attention import MultiHeadedAttention
# TODO? using span_based
from .relative_local_span_deberta import DisentangledLocalSelfAttention
# from .relative_local_deberta import DisentangledLocalSelfAttention
from .relative_deberta import DisentangledSelfAttention

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forward layer.
        :param input_size: dimensionality of the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(
        self, opts, size: int = 0, ff_size: int = 0, num_heads: int = 0, dropout: float = 0.1,
        local_layer: bool = False, use_relative: bool = True,
    ):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()
        self.use_relative = use_relative

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.rel_embeddings = nn.Embedding(opts.max_relative_positions * 2, opts.hidden_size)
        if local_layer and use_relative:
            self.src_src_att = DisentangledLocalSelfAttention(opts)
        elif not local_layer and use_relative:
            self.src_src_att = DisentangledSelfAttention(opts)
        elif local_layer and not use_relative:
            self.src_src_att = MultiHeadedLocalAttention(opts, num_heads, size, dropout=dropout)
        else:
            self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(
            input_size=size, ff_size=ff_size, dropout=dropout
        )
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, q: Tensor, kv: Tensor, mask: Tensor, src_length: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.

        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        q_norm = self.layer_norm(q)
        kv_norm = self.layer_norm(kv)
        if self.use_relative:
            mask = self.get_attention_mask(mask)
            h = self.src_src_att(
                hidden_states=kv_norm,
                attention_mask=mask,
                return_att=False,
                query_states=q_norm,
                relative_pos=None,
                rel_embeddings=self.rel_embeddings.weight,
                src_length=src_length)
        else:
            h = self.src_src_att(kv_norm, kv_norm, q_norm, mask)
        h = self.dropout(h) + q_norm
        o = self.feed_forward(h)
        return o

    def get_attention_mask(self, attention_mask):
        if attention_mask.dim() <= 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            # print("extended_attention_mask: ", extended_attention_mask.shape)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
            # print("attention_mask: ", attention_mask.shape, attention_mask)
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)
        return attention_mask

class TransformerEncoder(nn.Module):
    """
    Transformer Encoder
    """

    # pylint: disable=unused-argument
    def __init__(self, opts, freeze: bool = False, **kwargs):
        """
        Initializes the Transformer.
        :param hidden_size: hidden size and size of embeddings
        :param ff_size: position-wise feed-forward layer size.
          (Typically this is 2*hidden_size.)
        :param num_layers: number of layers
        :param num_heads: number of heads for multi-headed attention
        :param dropout: dropout probability for Transformer layers
        :param emb_dropout: Is applied to the input (word embeddings).
        :param freeze: freeze the parameters of the encoder during training
        :param kwargs:
        """
        super(TransformerEncoder, self).__init__()
        self.opts = opts
        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    opts=opts,
                    size=opts.hidden_size,
                    ff_size=opts.ff_size,
                    num_heads=opts.num_heads,
                    dropout=opts.dropout,
                    local_layer=num < opts.local_num_layers,
                    use_relative=opts.use_relative
                )
                for num in range(opts.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(opts.hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(opts.hidden_size)
        self.emb_dropout = nn.Dropout(p=opts.emb_dropout)

        self._output_size = opts.hidden_size
        self.atcn = AdaptiveTemporalConv(feat_dim=512, window_size=opts.window_size)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ):
        """
        Pass the input (and mask) through each layer in turn.
        Applies a Transformer encoder to sequence of embeddings x.
        The input mini-batch x needs to be sorted by src length.
        x and mask should have the same dimensions [batch, time, dim].

        :param embed_src: embedded src inputs,
            shape (batch_size, src_len, embed_size)
        :param src_length: length of src inputs
            (counting tokens before padding), shape (batch_size)
        :param mask: indicates padding areas (zeros where padding), shape
            (batch_size, src_len, embed_size)
        :return:
            - output: hidden states with
                shape (batch_size, max_length, directions*hidden),
            - hidden_concat: last hidden state with
                shape (batch_size, directions*hidden)
        """
        
        
        if self.opts.use_adaptive_tcn:
            kv = self.atcn(embed_src)
            kv = self.pe(kv)  # add position encoding to word embeddings
            kv = self.emb_dropout(kv)  # [bs, length, embed_size]
            q = self.pe(embed_src)  # add position encoding to word embeddings
            q = self.emb_dropout(q)  # [bs, length, embed_size]
        else:
            q = self.pe(embed_src)  # add position encoding to word embeddings
            kv = q = self.emb_dropout(q)  # [bs, length, embed_size]

        for i, layer in enumerate(self.layers):
            if i == 0:
                x = layer(q, kv, mask, src_length)
            else:
                x = layer(x, x, mask, src_length)
        return self.layer_norm(x)


if __name__ == "__main__":
    class Config():
        hidden_size = 512
        ff_size = 2048
        num_heads = 8
        dropout = 0.1
        emb_dropout = 0.1
        num_layers = 6

    opts = Config()
    m = TransformerEncoder(opts)
    x = torch.randn(5, 100, 512)
    out = m(x, None, None)
    print(out.shape)