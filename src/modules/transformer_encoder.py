
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils import freeze_params
from src.modules.transformer_layers import TransformerEncoderLayer
from src.modules.position_encoding import PositionalEncoding
from src.mymodules.relative_pe import RelativePosition


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

        # build all (num_layers) layers
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    size=opts.hidden_size,
                    ff_size=opts.ff_size,
                    num_heads=opts.num_heads,
                    dropout=opts.dropout,
                )
                for _ in range(opts.num_layers)
            ]
        )

        self.layer_norm = nn.LayerNorm(opts.hidden_size, eps=1e-6)
        self.pe = PositionalEncoding(opts.hidden_size)
        self.emb_dropout = nn.Dropout(p=opts.emb_dropout)

        # TODO?
        self.relative_pe = RelativePosition(opts)

        self._output_size = opts.hidden_size

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(
        self, embed_src: Tensor, src_length: Tensor, mask: Tensor
    ) -> (Tensor, Tensor):
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
        x = self.pe(embed_src)  # add position encoding to word embeddings
        x = self.emb_dropout(x)  # [bs, length, embed_size]

        # TODO?
        rl_pe = self.relative_pe(x.size(1), x.size(1))

        for layer in self.layers:
            x = layer(x, mask, rl_pe)
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