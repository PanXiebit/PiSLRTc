import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.utils import freeze_params
from src.modules.transformer_layers import TransformerDecoderLayer
from src.modules.position_encoding import PositionalEncoding
import numpy as np



def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype("uint8")
    return torch.from_numpy(mask) == 0


class TransformerDecoder(nn.Module):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(
        self, opts, gls_text_vocab, freeze: bool = False,
        # num_layers: int = 4,
        # num_heads: int = 8,
        # hidden_size: int = 512,
        # ff_size: int = 2048,
        # dropout: float = 0.1,
        # emb_dropout: float = 0.1,

        **kwargs
    ):
        """
        Initialize a Transformer decoder.

        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self._hidden_size = opts.hidden_size
        self._output_size = gls_text_vocab.text_vocab_len()

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList(
            [
                TransformerDecoderLayer(
                    size=opts.hidden_size,
                    ff_size=opts.ff_size,
                    num_heads=opts.num_heads,
                    dropout=opts.dropout,
                )
                for _ in range(opts.num_layers)
            ]
        )

        self.pe = PositionalEncoding(opts.hidden_size)
        self.layer_norm = nn.LayerNorm(opts.hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=opts.emb_dropout)
        self.output_layer = nn.Linear(opts.hidden_size, self._output_size, bias=False)

        if freeze:
            freeze_params(self)

    def forward(
        self,
        trg_embed: Tensor = None,
        encoder_output: Tensor = None,
        src_mask: Tensor = None,
        trg_mask: Tensor = None,
        **kwargs
    ):
        """
        Transformer decoder forward pass.

        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param src_mask:
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kwargs:
        :return:
        """
        assert trg_mask is not None, "trg_mask required for Transformer"

        x = self.pe(trg_embed)  # add position encoding to word embedding
        x = self.emb_dropout(x)

        trg_mask = trg_mask & subsequent_mask(trg_embed.size(1)).type_as(trg_mask)


        for layer in self.layers:
            x = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

        x = self.layer_norm(x)
        output = self.output_layer(x)

        return output

if __name__ == "__main__":
    from src.data.vocabulary import GlsTextVocab
    class Config():
        data_path = "data_bin/PHOENIX2014T"

        embedding_dim = 512
        input_size = 1024
        freeze = False,
        norm_type = "batch"
        activation_type = "softsign"
        scale = False
        scale_factor = None
        fp16 = False

        hidden_size = 512
        ff_size = 2048
        num_heads = 8
        dropout = 0.1
        emb_dropout = 0.1
        num_layers = 6

    opts = Config()
    gls_text_vocab = GlsTextVocab(opts)
    m = TransformerDecoder(opts, gls_text_vocab)
