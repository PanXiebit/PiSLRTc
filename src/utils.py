import yaml
import torch
import numpy as np
import random
import torch.nn as nn
from torch import nn, Tensor
import logging
import os

def init_logging(log_file):
    """Init for logging
    """
    logging.basicConfig(level = logging.INFO,
                        format = '%(asctime)s: %(message)s',
                        datefmt = '%m-%d %H:%M:%S',
                        filename = log_file,
                        filemode = 'w')
    # define a Handler which writes INFO message or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter(fmt='%(asctime)s: %(message)s', datefmt='%m-%d %H:%M:%S')
    # tell the handler to use this format
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def set_seed(seed: int):
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def load_config(path="configs/sign.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, "r", encoding="utf-8") as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg

def freeze_params(module: nn.Module):
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False

def get_activation(activation_type):
    if activation_type == "relu":
        return nn.ReLU()
    elif activation_type == "relu6":
        return nn.ReLU6()
    elif activation_type == "prelu":
        return nn.PReLU()
    elif activation_type == "selu":
        return nn.SELU()
    elif activation_type == "celu":
        return nn.CELU()
    elif activation_type == "gelu":
        return nn.GELU()
    elif activation_type == "sigmoid":
        return nn.Sigmoid()
    elif activation_type == "softplus":
        return nn.Softplus()
    elif activation_type == "softshrink":
        return nn.Softshrink()
    elif activation_type == "softsign":
        return nn.Softsign()
    elif activation_type == "tanh":
        return nn.Tanh()
    elif activation_type == "tanhshrink":
        return nn.Tanhshrink()
    else:
        raise ValueError("Unknown activation type {}".format(activation_type))

class MaskedNorm(nn.Module):
    """
        Original Code from:
        https://discuss.pytorch.org/t/batchnorm-for-different-sized-samples-in-batch/44251/8
    """

    def __init__(self, norm_type, num_groups, num_features):
        super().__init__()
        self.norm_type = norm_type
        if self.norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features=num_features)
        elif self.norm_type == "group":
            self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=num_features)
        elif self.norm_type == "layer":
            self.norm = nn.LayerNorm(normalized_shape=num_features)
        else:
            raise ValueError("Unsupported Normalization Layer")

        self.num_features = num_features

    def forward(self, x: Tensor, mask: Tensor):
        if self.training:
            reshaped = x.reshape([-1, self.num_features])
            # print("reshaped: ", reshaped.shape)
            reshaped_mask = mask.reshape([-1, 1]) > 0
            # print("reshaped_mask: ", reshaped_mask.shape)
            # print("mask: ", mask.shape)
            # exit()
            selected = torch.masked_select(reshaped, reshaped_mask).reshape(
                [-1, self.num_features]
            )

            batch_normed = self.norm(selected)
            # print("batch_normed: ", batch_normed.shape)
            scattered = reshaped.masked_scatter(reshaped_mask, batch_normed)
            return scattered.reshape([x.shape[0], -1, self.num_features])
        else:
            reshaped = x.reshape([-1, self.num_features])
            batched_normed = self.norm(reshaped)
            return batched_normed.reshape([x.shape[0], -1, self.num_features])



def apply_to_sample(f, sample):
    if len(sample) == 0:
        return {}

    def _apply(x):
        if torch.is_tensor(x):
            return f(x)
        elif isinstance(x, dict):
            return {
                key: _apply(value)
                for key, value in x.items()
            }
        elif isinstance(x, list):
            return [_apply(x) for x in x]
        else:
            return x

    return _apply(sample)


def move_to_cuda(sample, opts):

    def _move_to_cuda(tensor):
        if opts.fp16:
            return tensor.cuda(non_blocking=False).half()
        return tensor.cuda(non_blocking=False)

    return apply_to_sample(_move_to_cuda, sample)


class ModelManager(object):
    def __init__(self, max_num_models=5):
        self.max_num_models = max_num_models
        self.best_epoch = 0
        self.best_err = 1000
        self.model_file_list = []

    def update(self, model_file, err, epoch):
        self.model_file_list.append((model_file, err))
        self.update_best_err(err, epoch)
        self.sort_model_list()
        if len(self.model_file_list) > self.max_num_models:
            worst_model_file = self.model_file_list.pop(-1)[0]
            if os.path.exists(worst_model_file):
                os.remove(worst_model_file)
        logging.info('CURRENT BEST PERFORMANCE (epoch: {:d}): WER: {:.5f}'.
            format(self.best_epoch, self.best_err))

    def update_best_err(self, err, epoch):
        if err < self.best_err:
            self.best_err = err
            self.best_epoch = epoch

    def sort_model_list(self):
        self.model_file_list.sort(key=lambda x: x[1])


class ModelManager_bleu(object):
    def __init__(self, max_num_models=5):
        self.max_num_models = max_num_models
        self.best_epoch = 0
        self.best_bleu = 0.0
        self.model_file_list = []

    def update(self, model_file, bleu, epoch):
        self.model_file_list.append((model_file, bleu))
        self.update_best_bleu(bleu, epoch)
        self.sort_model_list()
        if len(self.model_file_list) > self.max_num_models:
            worst_model_file = self.model_file_list.pop(-1)[0]
            if os.path.exists(worst_model_file):
                os.remove(worst_model_file)
        logging.info('CURRENT BEST PERFORMANCE (epoch: {:d}): WER: {:.5f}'.
            format(self.best_epoch, self.best_bleu))

    def update_best_bleu(self, bleu, epoch):
        if bleu > self.best_bleu:
            self.best_bleu = bleu
            self.best_epoch = epoch

    def sort_model_list(self):
        self.model_file_list.sort(key=lambda x: x[1])


