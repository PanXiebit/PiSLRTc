
import argparse
import os


def get_parser():
    ap = argparse.ArgumentParser("Joey NMT")
    # if "--" is absoluted, the parameters must be passed by the command line.

    # mymodule parameters
    ap.add_argument("--max_relative_positions", default=8, type=int, help="max relative distance")
    ap.add_argument("--relative_attention", default=True, type=bool, help="whether using relative attention")
    ap.add_argument("--pos_att_type", default="c2p|p2c", type=str, help="position attention type")
    ap.add_argument("--window_size", default=16, type=int, help="window size")
    ap.add_argument("--local_num_layers", default=0, type=int, help="local_num_layers")
    ap.add_argument("--use_relative", default=True, type=bool, help="use_relative")
    ap.add_argument("--is_adaptive", default=True, type=bool, help="is_adaptive")
    ap.add_argument("--use_adaptive_tcn", default=True, type=bool, help="use_adaptive_tcn")

    # model parameters
    ap.add_argument("--embedding_dim", default=512, type=int, help="embedding dim")
    ap.add_argument("--input_size", default=1024, type=int, help="input feature dim")
    ap.add_argument("--hidden_size", default=512, type=int, help="hidden dim")
    ap.add_argument("--ff_size", default=2048, type=int, help="dim of feedforward")
    ap.add_argument("--num_heads", default=8, type=int, help="number of heads")
    ap.add_argument("--num_layers", default=6, type=int, help="number of layers")
    ap.add_argument("--dropout", default=0.1, type=float, help="attention dropout")
    ap.add_argument("--emb_dropout", default=0.1, type=float, help="embedding dropout")

    ap.add_argument("--freeze", default=False, type=bool, help="weather freeze the parameters")
    ap.add_argument("--norm_type", default="batch", type=str, help="normalization for spatial/word embedding")
    ap.add_argument("--activation_type", default="softsign", type=str, help="activation function for spatial/word embedding")
    ap.add_argument("--scale", default=False, type=bool, help="weather scale the embedding feature")
    ap.add_argument("--scale_factor", default=None, type=float, help="scale factor for the embedding feature")
    ap.add_argument("--fp16", default=False, type=bool, help="fp16")

    # criterion parameters
    ap.add_argument("--reg_loss_weight", default=1.0, type=float, help="reg_loss_weight")
    ap.add_argument("--tran_loss_weight", default=1.0, type=float, help="tran_loss_weight")
    ap.add_argument("--label_smoothing", default=0.0, type=float, help="label_smoothing")

    # optimizer parameters
    ap.add_argument("--optimizer", default="adam", type=str, help="optimizer")
    ap.add_argument("--clip", default=5.0, type=float, help="gradient clip")
    ap.add_argument("--slr_learning_rate", default=0.001, type=float, help="learning rate")
    ap.add_argument("--slt_learning_rate", default=0.001, type=float, help="learning rate")
    ap.add_argument("--weight_decay", default=0.0001, type=float, help="weight decay")
    ap.add_argument("--eps", default=1e-8, type=float, help="epsion of optimizer")
    ap.add_argument("--amsgrad", default=False, type=bool, help="amsgrad of optimizer")
    ap.add_argument("--decrease_factor", default=0.7, type=float, help="decrease_factor of lr schedule")
    ap.add_argument("--patience", default=8, type=int, help="patience of lr schedule")
    ap.add_argument("--warmup_steps", default=2000, type=int, help="warmup_steps")

    # inference parameters
    ap.add_argument("--reg_beam_size", default=5, type=int, help="recognition beam size")
    ap.add_argument("--max_output_length", default=500, type=int, help="max_output_length")
    ap.add_argument("--text_beam_size", default=5, type=int, help="text beam size")
    ap.add_argument("--alpha", default=-1, type=int, help="alpha")

    ap.add_argument("--reset_lr", default=False, type=bool, help="reset lr")


    # data
    ap.add_argument("--batch_size", default=5, type=int, help="batch size")
    ap.add_argument("--num_workers", default=8, type=int, help="num workers")
    ap.add_argument("--data_path", default="data_bin/PHOENIX2014T", type=str, help="data path")
    ap.add_argument("--corpus_dir", default="corpus_annotations", type=str, help="corpus path")
    ap.add_argument("--data_path_scratch",
                    default="/home/panxie/xp_workspace/Data/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/features/fullFrame-210x260px",
                    type=str, help="video path")
    ap.add_argument("--dataset_version", default="phoenix_2014_trans", type=str, help="dataset version")
    ap.add_argument("--check_point", default="log/ep53.pkl", type=str, help="checkpoint dir")
    ap.add_argument("--max_epoch", type=int, default=100, help="max_epoch")
    ap.add_argument("--max_updates", type=int, default=30000000, help="max updates")
    ap.add_argument("--print_step", type=int, default=100, help="print_step")
    ap.add_argument("--DEBUG", type=bool, default=True, help="debug")

    # fixed parameters
    ap.add_argument("--seed", default=8, type=int, help="random seed")
    ap.add_argument("--log_dir", default="log", type=str, help="log dir")
    # ap.add_argument("--log_file", default="v1", type=str, help="log file")
    ap.add_argument("--mode", type=str, default="train", help="train/dev/testz")

    args = ap.parse_args()
    return args