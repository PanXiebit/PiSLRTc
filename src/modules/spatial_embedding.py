import math
import torch
from torch import nn, Tensor
from src.utils import freeze_params, get_activation, MaskedNorm


class SpatialEmbeddings(nn.Module):

    """
    Simple Linear Projection Layer
    (For encoder outputs to predict glosses)
    """

    # pylint: disable=unused-argument
    def __init__(self, opts, freeze=False):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param input_size:
        :param freeze: freeze the embeddings during training
        """
        super().__init__()

        self.opts = opts
        self.embedding_dim = opts.embedding_dim
        self.input_size = opts.input_size
        self.linear = nn.Linear(self.input_size, self.embedding_dim)

        self.norm_type = opts.norm_type
        self.num_heads = opts.num_heads
        if self.norm_type:
            self.norm = MaskedNorm(
                norm_type=self.norm_type, num_groups=self.num_heads, num_features=self.embedding_dim
            )

        self.activation_type = opts.activation_type
        if self.activation_type:
            self.activation = get_activation(self.activation_type)

        self.scale = opts.scale
        if self.scale:
            if opts.scale_factor:
                self.scale_factor = opts.scale_factor
            else:
                self.scale_factor = math.sqrt(self.embedding_dim)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        :param mask: frame masks
        :param x: input frame features
        :return: embedded representation for `x`
        """
        x = self.linear(x)

        if self.norm_type:
            x = self.norm(x, mask)

        if self.activation_type:
            x = self.activation(x)

        if self.scale:
            return x * self.scale_factor
        else:
            return x

    def __repr__(self):
        return "%s(embedding_dim=%d, input_size=%d)" % (
            self.__class__.__name__,
            self.embedding_dim,
            self.input_size,
        )



if __name__ == "__main__":
    class Config():
        embedding_dim = 512
        input_size = 1024
        num_heads = 3
        freeze = False,
        norm_type = "batch"
        activation_type = "softsign"
        scale = False
        scale_factor = None
        fp16 = False
    opts = Config()
    m = SpatialEmbeddings(opts)
    x = torch.randn(5, 100, 1024)
    x_len = torch.LongTensor([100, 98, 57, 39, 100])
    out = m(x, x_len)
    print(out.shape)
