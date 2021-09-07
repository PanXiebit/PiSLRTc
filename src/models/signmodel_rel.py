import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math
from src.modules.spatial_embedding import SpatialEmbeddings
from src.mymodules.transformer_rel_encoder import TransformerEncoder
from src.modules.word_embedding import WordEmbeddings
from src.modules.transformer_decoder import TransformerDecoder
from .adaptive_tcn import AdaptiveTemporalConv



class SignModel(nn.Module):
    def __init__(self, opts, gls_text_vocab, do_recognition=True, do_translation=True):
        super(SignModel, self).__init__()

        self.opts = opts
        self.sgn_embeding = SpatialEmbeddings(opts)

        self.encoder = TransformerEncoder(opts)
        self.do_recognition = do_recognition
        self.do_translation = do_translation

        self.text_pad = gls_text_vocab.token2idx("<pad>")

        if do_recognition:
            self.gloss_output_layer = nn.Linear(self.encoder._output_size, gls_text_vocab.gloss_vocab_len())
        else:
            self.gloss_output_layer = None

        if do_translation:
            self.text_embedding = WordEmbeddings(opts, gls_text_vocab)
            self.decoder = TransformerDecoder(opts, gls_text_vocab)

    def forward(self, sgn, sgn_len, text, text_len):

        sgn_mask = self._get_mask(sgn_len) # [bs, 1, sgn_len]
        sgn = self.sgn_embeding(sgn, sgn_mask)
        encoder_output = self.encoder(sgn, sgn_len, sgn_mask)

        if self.do_recognition:
            gloss_logits = self.gloss_output_layer(encoder_output)
        else:
            gloss_logits = None

        if self.do_translation:
            txt_mask = text.ne(self.text_pad).unsqueeze(1)  # [bs, 1, trg_len]
            txt_embed = self.text_embedding(text, txt_mask)
 

            decoder_outputs = self.decoder(
                trg_embed=txt_embed,
                encoder_output=encoder_output,
                src_mask=sgn_mask,
                trg_mask=txt_mask)
        else:
            decoder_outputs = None
        return gloss_logits, decoder_outputs

    def _get_mask(self, x_len):
        pos = torch.arange(0, max(x_len)).unsqueeze(0).repeat(x_len.size(0), 1).to(x_len.device)
        if self.opts.fp16:
            pos = pos.half()
        pos[pos >= x_len.unsqueeze(1)] = max(x_len) + 1
        mask = pos.ne(max(x_len) + 1)
        return mask.unsqueeze(1)

    def inference(self, sgn, sgn_len):
        sgn_mask = self._get_mask(sgn_len)
        sgn = self.sgn_embeding(sgn, sgn_len)  # [bs, 1, sgn_len]
        encoder_output = self.encoder(sgn, sgn_len, sgn_mask)

        if self.do_recognition:
            gloss_logits = self.gloss_output_layer(encoder_output)
        else:
            gloss_logits = None
        return gloss_logits, encoder_output, sgn_mask

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
    m = SignModel(opts, gls_text_vocab)

    sign = torch.randn(2, 100, 1024)
    sign_len = torch.LongTensor([100, 87])
    text = torch.LongTensor([[190, 92, 5, 89, 23], [56, 90, 10, 2, 2]])
    text_len = torch.LongTensor([5, 3])
    out = m(sign, sign_len, text, text_len)
    print(out[0].shape, out[1].shape)