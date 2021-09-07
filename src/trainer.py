import torch
from src.utils import move_to_cuda
import logging
import math
import tensorflow as tf
import numpy as np
from itertools import groupby
from phoenix_utils.phoenix_cleanup import clean_phoenix_2014, clean_phoenix_2014_trans
from torch.utils.data import DataLoader
from src.optimizer.build_optim import build_optimizer
from src.optimizer.lr_schedule import build_scheduler
from torch.optim import lr_scheduler
import ctcdecode
import torch.nn.functional as F
from metrics.wer import get_wer_delsubins
from src.decoder.left_to_right import LeftToRight


class Trainer(object):
    def __init__(self, opts, model, criterion, gls_text_vocab):
        self.opts = opts
        self.model = model
        self.criterion = criterion
        self.gls_text_vocab = gls_text_vocab
        self.max_output_length = opts.max_output_length
        self.text_beam_size = opts.text_beam_size
        self.alpha = opts.alpha

        self.cuda = torch.cuda.is_available()
        if self.cuda:
            self.criterion = self.criterion.cuda()
            self.model = self.model.cuda()
            if opts.fp16:
                self.model = self.model.half()

        # params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        # self.minimize_metric = True

        slr_params, slt_params = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad and "sgn_embedding" in n:
                print("slr parameters: ", n)
                slr_params.append(p)
            elif p.requires_grad and "encoder" in n:
                print("slr parameters: ", n)
                slr_params.append(p)
            elif p.requires_grad and "gloss_output_layer" in n:
                print("slr parameters: ", n)
                slr_params.append(p)
            else:
                print("slt parameters: ", n)
                slt_params.append(p)

        self.optimizer = torch.optim.Adam([
            {"name": "slr", "params": slr_params, "lr": opts.slr_learning_rate},
            {"name": "slt", "params": slt_params, "lr": opts.slt_learning_rate}],
            betas=(0.9, 0.999),
            eps=opts.eps,
            weight_decay=opts.weight_decay,
            amsgrad=opts.amsgrad,
        )
        for param_group in self.optimizer.param_groups:
            print(param_group["name"], "lr: ", param_group["lr"])

        self.decoder_vocab = [chr(x) for x in range(20000, 20000 + self.gls_text_vocab.gloss_vocab_len())]
        self.decoder = ctcdecode.CTCBeamDecoder(self.decoder_vocab, beam_width=self.opts.reg_beam_size,
                                                blank_id=gls_text_vocab.gloss2idx("<blank>"),
                                                num_processes=10)

        self.text_decoder = LeftToRight(opts)


    def train_step(self, samples):
        self._set_seed()
        self.model.train()
        self.criterion.train()
        self.optimizer.zero_grad()
        samples = self._prepare_sample(samples, self.opts)
        reg_loss, tran_loss = self.criterion(samples, self.model)
        loss = reg_loss + tran_loss

        loss.backward()
        self.slt_lr_schedule(self.get_num_updates())
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opts.clip)

        self.set_num_updates(self.get_num_updates() + 1)
        self.optimizer.step()
        # self.lr_schedule.step()
        return loss, reg_loss, tran_loss, self.get_num_updates()

    def valid_step(self, sample, decoded_dict, tran_outputs):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample, self.opts)

            vname = sample["name"]
            sgn_feat = sample["sign_feature"]
            sgn_len = sample["sign_len"]
            gloss = sample["gloss"]
            gloss_len = sample["gloss_len"]
            text_inp = sample["text_inp"]
            text_trg = sample["text_trg"]
            text_len = sample["text_len"]

            gloss_logits, encoder_out, sgn_mask = self.model.inference(sgn_feat, sgn_len)

            # TODO! recognition prediction and compute wer
            gloss_logits = F.softmax(gloss_logits, dim=-1)
            # print("gloss_logits: ", gloss_logits.shape)  # [bs, sgn_len, gloss_vocab_size]
            # print("sgn_len: ", sgn_len)
            pred_seq, _, _, out_seq_len = self.decoder.decode(gloss_logits, sgn_len)
            # print("pred_seq: ", pred_seq.shape)        # [bs, reg_beam_size, sgn_len]
            # print("out_seq_len: ", out_seq_len)  # [bs, reg_beam_size]
            err_delsubins = np.zeros([4])
            count = 0
            correct = 0
            for i, length in enumerate(gloss_len):
                ref = gloss[i][:length].tolist()
                hyp = [x[0] for x in groupby(pred_seq[i][0][:out_seq_len[i][0]].tolist())]
                # print("ref: ", ref)
                # print("hyp: ", hyp)
                decoded_dict[vname[i]] = (ref, hyp)
                correct += int(ref == hyp)
                err = get_wer_delsubins(ref, hyp)
                err_delsubins += np.array(err)
                count += 1

            # TODO! Translation prediction and compute bleu
            final_outputs, _ = self.text_decoder.generate(
                self.model, self.gls_text_vocab, encoder_out, sgn_mask, self.max_output_length,
                self.text_beam_size, self.alpha)
            for i in range(final_outputs.shape[0]):
                tran_outputs.append((final_outputs[i].tolist(), text_trg[i].cpu().numpy().tolist()))
        return err_delsubins, correct, count, tran_outputs


    def valid_tf_step(self, sample, all_gls_outputs):
        with torch.no_grad():
            self.model.eval()
            self.criterion.eval()

            sample = self._prepare_sample(sample, self.opts)
            decoded_gloss_sequences, tran_loss = self.criterion.inference_tf(sample, self.model)
            all_gls_outputs.append(decoded_gloss_sequences)

            if self.opts.dataset_version == "phoenix_2014_trans":
                gls_cln_fn = clean_phoenix_2014_trans
            elif self.opts.dataset_version == "phoenix_2014":
                gls_cln_fn = clean_phoenix_2014
            else:
                raise ValueError("Unknown Dataset Version: " + self.opts.dataset_version)


    def get_batch_iterator(self, datasets, batch_size, shuffle, num_workers=8, drop_last=True):
        return DataLoader(datasets,
                          batch_size=batch_size,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          collate_fn=datasets.collate_fn,
                          drop_last=drop_last,
                          pin_memory=False)


    def _prepare_sample(self, sample, opts):
        if sample is None or len(sample) == 0:
            return None
        if self.cuda:
            sample = move_to_cuda(sample, opts)
        return sample

    def _set_seed(self):
        # Set seed based on opts.seed and the update number so that we get
        # reproducible results when resuming from checkpoints
        seed = self.opts.seed + self.get_num_updates()
        torch.manual_seed(seed)
        if self.cuda:
            torch.cuda.manual_seed(seed)

    def get_num_updates(self):
        """Get the number of parameters updates."""
        return self._num_updates

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        self._num_updates = num_updates

    def save_checkpoint(self, filename, epoch, num_updates, loss):
        state_dict = {
            'epoch': epoch,
            'num_updates': num_updates,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            "loss": loss,
        }
        torch.save(state_dict, filename)

    def load_checkpoint(self, filename):
        state_dict = torch.load(filename)
        epoch = state_dict["epoch"]
        num_updates = state_dict["num_updates"]
        loss = state_dict["loss"]
        self.model.load_state_dict(state_dict["model_state_dict"])
        if not self.opts.reset_lr:
            self.optimizer.load_state_dict(state_dict["optimizer_state_dict"])
        else:
            old_lr = []
            for param_group in state_dict["optimizer_state_dict"]["param_groups"]:
                old_lr.append(param_group["lr"])
            print('==== Change lr from %s to %f ====' % (" ".join([str(lr) for lr in old_lr]),
                                                         self.opts.learning_rate))
        return epoch, num_updates, loss

    def get_lr(self):
        lrs = {}
        for param_group in self.optimizer.param_groups:
            lrs[param_group["name"]] = param_group['lr']
        return lrs

    def slt_lr_schedule(self, step, d_model=512):
        warmup_steps = self.opts.warmup_steps
        arg1 = 1 / math.sqrt(step + 1)
        arg2 = step * (warmup_steps ** -1.5)
        new_lr = 1 / math.sqrt(d_model) * min(arg1, arg2)

        for param_group in self.optimizer.param_groups:
            # if param_group["name"] == "slt":
            param_group['lr'] = new_lr
            # logging.info("name: {}, lr: {:.6f}".format(param_group["name"], param_group["lr"]))


    def slr_lr_schedule(self, epoch):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        if epoch >= 40 and (epoch-40) % 3 == 0:
            for param_group in self.optimizer.param_groups:
                if param_group["name"] == "slr":
                    param_group['lr'] = param_group['lr'] * 0.5
                    logging.info("name: {}, lr: {:.6f}".format(param_group["name"], param_group["lr"]))
