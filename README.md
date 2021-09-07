# PiSLRTc

[PiSLTRc: Position-informed Sign Language Transformer with Content-aware Convolution](https://ieeexplore.ieee.org/document/9528010) in IEEE Transactions on Multimedia (TMM).

## Installation
- Pytorch (1.0)
- ctcdecode==0.4 [parlance/ctcdecode]
- sclite [kaldi-asr/kaldi], install kaldi tool to get sclite for evaluation

## Data Preparation
1. Download the feature of RWTH-PHOENIX-Weather 2014 Dataset
```
#!/usr/bin/env bash
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.train"
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.dev"
wget "http://cihancamgoz.com/files/cvpr2020/phoenix14t.pami0.test"
```
2. put the feature in "data-bin/PHOENIX2014T/"
```
└── PHOENIX2014T
    ├── gloss_vocab.txt
    ├── phoenix14t.pami0.dev
    ├── phoenix14t.pami0.test
    ├── phoenix14t.pami0.train
    └── text_vocab.txt 
```

## Training
BASE_PATH=log/piSLRTc
python -m train --mode "train" --log_dir $BASE_PATH --data_path "../data-bin/PHOENIX2014T/" --input_size 1024 --embedding_dim 512 --hidden_size 512 --num_heads 8 --num_layers 6 --max_relative_positions 16 --norm_type "batch" --activation_type "softsign" --reg_loss_weight 1.0 --tran_loss_weight 1.0 --label_smoothing 0.0001 --optimizer "adam" --slr_learning_rate 0.001 --slt_learning_rate 0.2 --warmup_steps 4000 --weight_decay 0.0001 --decrease_factor 0.5 --patience 5 --reg_beam_size 5 --max_output_length 300 --text_beam_size 5 --alpha 2 --batch_size 5 --check_point $BASE_PATH/ep.pkl --max_epoch 60 --print_step 100

## inference
1. averaging checkpoints

```
ckpt_pth=log/piSLRTc
avg_path=log/piSLRTc/average
mkdir $avg_path
python -m average_ckpt \
    --inputs $ckpt_pth \
    --output $avg_path/average_ckpt.pt \
    --num-epoch-checkpoints 5 \
    --checkpoint-upper-bound 42
```

2. testing

python -m train --mode "test" --log_dir $BASE_PATH --data_path "../data-bin/PHOENIX2014T/" --input_size 1024 --embedding_dim 512 --hidden_size 512 --num_heads 8 --num_layers 6 --max_relative_positions 16 --norm_type "batch" --activation_type "softsign" --reg_loss_weight 1.0 --tran_loss_weight 1.0 --label_smoothing 0.0001 --optimizer "adam" --slr_learning_rate 0.001 --slt_learning_rate 0.2 --warmup_steps 4000 --weight_decay 0.0001 --decrease_factor 0.5 --patience 5 --reg_beam_size 5 --max_output_length 300 --text_beam_size 5 --alpha 2 --batch_size 5 --check_point $avg_path/average_ckpt.pt --max_epoch 100 --print_step 100

## Citation
If you find this repo useful in your research works, please consider citing:
```
@ARTICLE{9528010,
  author={Xie, Pan and Zhao, Mengyi and Hu, Xiaohui},
  journal={IEEE Transactions on Multimedia}, 
  title={PiSLTRc: Position-informed Sign Language Transformer with Content-aware Convolution}, 
  year={2021},
  volume={},
  number={},
  pages={1-1},
  doi={10.1109/TMM.2021.3109665}}
```

