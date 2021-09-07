import torch
import numpy as np
import random
import os

from configs.options import get_parser
import logging
from src.utils import load_config, init_logging
from src.data.datasets import SignRegTranDataset
from src.models.signmodel_rel import SignModel
from src.data.vocabulary import GlsTextVocab
from src.criterion.ctc_and_ce_loss import CtcCeLoss
from src.trainer import Trainer
from tqdm import tqdm
from metrics.metrics import wer_single
from phoenix_utils.phoenix_cleanup import clean_phoenix_2014_trans, clean_phoenix_2014
from metrics.metrics import bleu, chrf, rouge
import torch.optim as optim
from src.utils import ModelManager, ModelManager_bleu
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    opts = get_parser()
    setup_seed(opts.seed)
    if not os.path.exists(opts.log_dir):
        os.makedirs(opts.log_dir)
    log_name = "_".join(time.asctime(time.localtime(time.time())).split(" ")[-2].split(":"))
    if opts.mode == "train" or opts.mode == "dev":
        init_logging(os.path.join(opts.log_dir, "train_{}.log".format(log_name)))
    else:
        init_logging(os.path.join(opts.log_dir, "test_{}.log".format(log_name)))

    if torch.cuda.is_available():
        # torch.cuda.set_device(opts.gpu)
        logging.info("Using GPU!")
        device = "cuda"
    else:
        logging.info("Using CPU!")
        device = "cpu"

    logging.info(opts)

    gls_text_vocab = GlsTextVocab(opts)
    if opts.mode == "train":
        train_datasets = SignRegTranDataset(opts, gls_text_vocab, phrase="train", DEBUG=opts.DEBUG)
        valid_datasets = SignRegTranDataset(opts, gls_text_vocab, phrase="dev", DEBUG=opts.DEBUG, sample=False)
        test_datasets = SignRegTranDataset(opts, gls_text_vocab, phrase="test", DEBUG=opts.DEBUG, sample=False)
    else:
        train_datasets = SignRegTranDataset(opts, gls_text_vocab, phrase="dev", DEBUG=opts.DEBUG, sample=False)
        valid_datasets = SignRegTranDataset(opts, gls_text_vocab, phrase="dev", DEBUG=opts.DEBUG, sample=False)
        test_datasets = SignRegTranDataset(opts, gls_text_vocab, phrase="test", DEBUG=opts.DEBUG, sample=False)

    model = SignModel(opts, gls_text_vocab, do_recognition=True, do_translation=True)
    criterion = CtcCeLoss(opts, gls_text_vocab, reg_loss_weight=opts.reg_loss_weight,
                          tran_loss_weight=opts.tran_loss_weight, smoothing=opts.label_smoothing)

    logging.info(model)
    # exit()

    trainer = Trainer(opts, model, criterion, gls_text_vocab)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     trainer.optimizer, factor=opts.decrease_factor, patience=opts.patience)


    if os.path.exists(opts.check_point):
        logging.info("Loading checkpoint file from {}".format(opts.check_point))
        epoch, num_updates, loss = trainer.load_checkpoint(opts.check_point)
    else:
        logging.info("No checkpoint file in found in {}".format(opts.check_point))
        epoch, num_updates, loss = 0, 0, 0.0

    logging.info('| num. module params: {} (num. trained: {})'.format(
        sum(p.numel() for p in model.parameters()),
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    ))

    trainer.set_num_updates(num_updates)
    model_manager = ModelManager(max_num_models=5)
    model_manager_bleu = ModelManager_bleu(max_num_models=5)
    best_wer = 100.0
    best_bleu = 0.0

    while epoch < opts.max_epoch and trainer.get_num_updates() < opts.max_updates:
        epoch += 1
        # trainer.slr_lr_schedule(epoch)
        if opts.mode == "train" or opts.mode == "dev":
            loss = train(opts, train_datasets, trainer, epoch)
            reg_res, tran_res = eval(opts, valid_datasets, trainer, epoch, gls_text_vocab)
            eval(opts, test_datasets, trainer, epoch, gls_text_vocab)
            if reg_res["wer"] < best_wer:
                best_wer = reg_res["wer"]
            if tran_res["bleu"]["bleu4"] > best_bleu:
                best_bleu = tran_res["bleu"]["bleu4"]
            logging.info("Epoch: {}, best_wer: {:.4f} and best_bleu-4: {:.4f}".format(
                epoch, best_wer, best_bleu
            ))
            save_ckpt = os.path.join(opts.log_dir, 'ep{:d}_wer_{:.4f}_bleu_{:.4f}.pkl'.
                                     format(epoch, reg_res["wer"], tran_res["bleu"]["bleu4"]))
            trainer.save_checkpoint(save_ckpt, epoch, num_updates, loss)
            # TODO? lr schedule.
            model_manager.update(save_ckpt, reg_res["wer"], epoch)
            # model_manager_bleu.update(save_ckpt, tran_res["bleu"]["bleu4"], epoch)

        if opts.mode == "test":
            # print("HERE!!!")
            eval(opts, valid_datasets, trainer, epoch, gls_text_vocab)
            eval(opts, test_datasets, trainer, epoch, gls_text_vocab)
            exit()


def train(opts, train_datasets, trainer, epoch):
    train_iter = trainer.get_batch_iterator(train_datasets, batch_size=opts.batch_size, shuffle=True,
                                            num_workers=opts.num_workers)
    epoch_loss, epoch_reg_loss, epoch_tran_loss = [], [], []
    for samples in train_iter:
        loss, reg_loss, tran_loss, num_updates = trainer.train_step(samples)
        epoch_loss.append(loss.item())
        epoch_reg_loss.append(reg_loss.item())
        epoch_tran_loss.append(tran_loss.item())

        lrs = trainer.get_lr()

        if (num_updates % opts.print_step) == 0:
            logging.info('Epoch: {:d}, num_updates: {:d}, loss: {:.3f}, reg_loss: {:.3f}, tran_loss: {:.3f}, '
                         'slr_lr: {:.6f}, slt_lr: {:.6f}'.
                         format(epoch, num_updates, loss, reg_loss, tran_loss, lrs["slr"], lrs["slt"]))
    logging.info('Epoch: {:d}, loss: {:.3f}, reg_loss: {:.3f}, tran_loss: {:.3f}'.
                 format(epoch, np.mean(epoch_loss), np.mean(epoch_reg_loss), np.mean(epoch_tran_loss)))
    return np.mean(epoch_loss)


def eval(opts, valid_datasets, trainer, epoch, gls_text_vocab):
    eval_iter = trainer.get_batch_iterator(valid_datasets, batch_size=opts.batch_size, shuffle=False,
                                           num_workers=opts.num_workers)

    decoded_dict = {}
    tran_outputs = []
    val_err, val_correct, val_count = np.zeros([4]), 0, 0

    for samples in tqdm(eval_iter):
        err, correct, count, tran_outputs = trainer.valid_step(samples, decoded_dict, tran_outputs)
        val_err += err
        val_correct += correct
        val_count += count

    logging.info('-' * 50)
    logging.info('Epoch: {:d}, DEV ACC: {:.5f}, {:d}/{:d}'.format(epoch, val_correct / val_count, val_correct, val_count))
    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'.format(epoch,
        val_err[0] / val_count, val_err[1] / val_count, val_err[2] / val_count, val_err[3] / val_count))

    if opts.dataset_version == "phoenix_2014_trans":
        gls_cln_fn = clean_phoenix_2014_trans
    elif opts.dataset_version == "phoenix_2014":
        gls_cln_fn = clean_phoenix_2014
    else:
        raise ValueError("Unknown Dataset Version: " + opts.dataset_version)

    total_error = total_del = total_ins = total_sub = total_ref_len = 0
    for vname, (ref, hyp) in decoded_dict.items():
        ref_sent = gls_cln_fn(gls_text_vocab.gloss_lis_to_sentences(ref))
        hyp_sent = gls_cln_fn(gls_text_vocab.gloss_lis_to_sentences(hyp))

        res = wer_single(ref_sent, hyp_sent)
        total_error += res["num_err"]
        total_del += res["num_del"]
        total_ins += res["num_ins"]
        total_sub += res["num_sub"]
        total_ref_len += res["num_ref"]

    wer = (total_error / total_ref_len) * 100
    del_rate = (total_del / total_ref_len) * 100
    ins_rate = (total_ins / total_ref_len) * 100
    sub_rate = (total_sub / total_ref_len) * 100

    logging.info('Epoch: {:d}, DEV WER: {:.5f}, SUB: {:.5f}, INS: {:.5f}, DEL: {:.5f}'
                 .format(epoch, wer, sub_rate, ins_rate, del_rate))

    # translation
    txt_refs, txt_hyps = [], []
    for i in range(len(tran_outputs)):
        txt_hyp = gls_text_vocab.text_list_to_sentences(tran_outputs[i][0])
        txt_ref = gls_text_vocab.text_list_to_sentences(tran_outputs[i][1])
        # print("txt_hyp: ", txt_hyp)
        # print("txt_ref: ", txt_ref)
        txt_refs.append(txt_ref)
        txt_hyps.append(txt_hyp)

    txt_bleu = bleu(references=txt_refs, hypotheses=txt_hyps)
    txt_chrf = chrf(references=txt_refs, hypotheses=txt_hyps)
    txt_rouge = rouge(references=txt_refs, hypotheses=txt_hyps)
    logging.info('Epoch: {:d}, BLEU-1: {:.5f}, BLEU-2: {:.5f}, BLEU-3: {:.4f}, BLEU-4: {:.4f}'.
                 format(epoch, txt_bleu["bleu1"], txt_bleu["bleu2"], txt_bleu["bleu3"], txt_bleu["bleu4"]))
    logging.info('Epoch: {:d}, CHRF: {:.5f}, ROUGE: {:.5f}'
                 .format(epoch, txt_chrf, txt_rouge))

    reg_res = {"wer": wer, "del_rate": del_rate, "ins_rate": ins_rate, "sub_rate": sub_rate,}
    tran_res = {"bleu": txt_bleu, "chrf": txt_chrf, "rouge": txt_rouge}
    return reg_res, tran_res


if __name__ == "__main__":
    main()
