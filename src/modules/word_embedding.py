import math
import torch
from torch import nn, Tensor
from src.utils import freeze_params, get_activation, MaskedNorm

class WordEmbeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self, opts, gls_text_vocab, freeze: bool = False, **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(WordEmbeddings, self).__init__()

        self.embedding_dim = opts.embedding_dim
        self.vocab_size = gls_text_vocab.text_vocab_len()
        self.padding_idx = gls_text_vocab.token2idx("<pad>")
        self.embed = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=self.padding_idx)

        self.norm_type = opts.norm_type
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=opts.norm_type, num_groups=opts.num_heads, num_features=opts.embedding_dim
            )

        self.activation_type = opts.activation_type
        if self.activation_type:
            self.activation = get_activation(opts.activation_type)

        self.scale = opts.scale
        if self.scale:
            if opts.scale_factor:
                self.scale_factor = opts.scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor = None) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param mask: token masks
        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """

        x = self.embed(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x