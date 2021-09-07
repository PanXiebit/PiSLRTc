import torch
import torch.nn as nn
from src.criterion.ce_loss import XentLoss
import tensorflow as tf
import numpy as np
from itertools import groupby


class CtcCeLoss(nn.Module):
    def __init__(self, opts, gls_text_vocab, reg_loss_weight=1.0, tran_loss_weight=1.0, smoothing=0.0):
        super(CtcCeLoss, self).__init__()
        self.opts = opts
        self.reg_loss_weight = reg_loss_weight
        self.tran_loss_weight = tran_loss_weight
        self.gls_text_vocab = gls_text_vocab

        blank_id = gls_text_vocab.gloss2idx("<blank>")
        self.reg_ctc_loss = nn.CTCLoss(blank=blank_id, reduction="mean")
            # blank=blank_id, reduction="mean", zero_infinity=True)

        text_pad_idx = gls_text_vocab.token2idx("<pad>")
        self.ce_loss = XentLoss(text_pad_idx, smoothing=smoothing)

    def forward(self, sample, model):
        vname = sample["name"]
        sgn_feat = sample["sign_feature"]
        sgn_len = sample["sign_len"]
        gloss = sample["gloss"]
        gloss_len = sample["gloss_len"]
        text_inp = sample["text_inp"]
        text_trg = sample["text_trg"]
        text_len = sample["text_len"]

        # gloss_logits: [bs, sgn_len, gloss_vocab_size]
        # decoder_outputs: [bs, trg_len, text_vocab_size]
        gloss_logits, decoder_outputs = model(sgn_feat, sgn_len, text_inp, text_len)

        # recognition ctc loss
        gloss_probs = gloss_logits.log_softmax(-1)
        gloss_probs = gloss_probs.permute(1, 0, 2)
        reg_ctc_loss = self.reg_ctc_loss(
            gloss_probs.cpu(), gloss.cpu(), sgn_len.cpu(), gloss_len.cpu()).to(gloss.device) * self.reg_loss_weight
        # reg_ctc_loss /= gloss.size(0)
        if torch.isnan(reg_ctc_loss):
            print(sgn_len, gloss_len)
            # exit()
        # translation loss
        dec_probs = decoder_outputs.log_softmax(-1)
        tran_loss = self.ce_loss(dec_probs, text_trg) * self.tran_loss_weight
        num_tokens = text_trg.ne(self.gls_text_vocab.token2idx("<pad>")).sum()
        tran_loss /= (num_tokens + 1e-8)
        if torch.isnan(tran_loss):
            print("num_tokens: ", num_tokens)
            print("reg_ctc_loss: ", reg_ctc_loss)
            print("tran_loss: ", tran_loss)
            exit()

        return reg_ctc_loss, tran_loss

    # def inference(self, sample, model):
    #     vname = sample["name"]
    #     sgn_feat = sample["sign_feature"]
    #     sgn_len = sample["sign_len"]
    #     gloss = sample["gloss"]
    #     gloss_len = sample["gloss_len"]
    #     text_inp = sample["text_inp"]
    #     text_trg = sample["text_trg"]
    #     text_len = sample["text_len"]
    #
    #     gloss_logits, decoder_outputs = model(sgn_feat, sgn_len, text_inp, text_len)
    #
    #     return gloss_logits, decoder_outputs

    def inference_tf(self, sample, model):
        vname = sample["name"]
        sgn_feat = sample["sign_feature"]
        sgn_len = sample["sign_len"]
        gloss = sample["gloss"]
        gloss_len = sample["gloss_len"]
        text_inp = sample["text_inp"]
        text_trg = sample["text_trg"]
        text_len = sample["text_len"]

        gloss_logits, decoder_outputs = model(sgn_feat, sgn_len, text_inp, text_len)

        # recognition ctc loss
        gloss_probs = gloss_logits.log_softmax(-1)
        gloss_probs = gloss_probs.permute(1, 0, 2)

        gloss_probs = gloss_probs.cpu().detach().numpy()

        tf_gloss_probs = np.concatenate((gloss_probs[:, :, 1:], gloss_probs[:, :, 0, None]), axis=-1)

        ctc_decode, _ = tf.nn.ctc_beam_search_decoder(
            inputs=tf_gloss_probs,
            sequence_length=sgn_len.cpu().detach().numpy(),
            beam_width=self.opts.reg_beam_size,
            top_paths=1,
        )
        ctc_decode = ctc_decode[0]
        # Create a decoded gloss list for each sample
        tmp_gloss_sequences = [[] for i in range(gloss_logits.shape[0])]
        print("tmp_gloss_sequences: ", tmp_gloss_sequences)

        for (value_idx, dense_idx) in enumerate(ctc_decode.indices):
            tmp_gloss_sequences[dense_idx[0]].append(
                ctc_decode.values[value_idx].numpy() + 1
            )
        decoded_gloss_sequences = []
        for seq_idx in range(0, len(tmp_gloss_sequences)):
            decoded_gloss_sequences.append(
                [x[0] for x in groupby(tmp_gloss_sequences[seq_idx])]
            )

        print("decoded_gloss_sequences: ", decoded_gloss_sequences)

        dec_probs = decoder_outputs.log_softmax(-1)
        tran_loss = self.ce_loss(dec_probs, text_trg) * self.tran_loss_weight
        return decoded_gloss_sequences, tran_loss