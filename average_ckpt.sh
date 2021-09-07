LOCAL_NUM=2
ckpt_pth=log/ablation/diff_only_relative$LOCAL_NUM/
avg_path=log/ablation/diff_only_relative$LOCAL_NUM/average
mkdir $avg_path
python -m average_ckpt \
    --inputs $ckpt_pth \
    --output $avg_path/average_ckpt.pt \
    --num-epoch-checkpoints 5 \
    --checkpoint-upper-bound 42

python -m train \
    --mode "test" \
    --log_dir $ckpt_pth \
    --data_path "data_bin/PHOENIX2014T" \
    --embedding_dim 512 \
    --hidden_size 512 \
    --num_heads 8 \
    --num_layers 6 \
    --local_num_layers $LOCAL_NUM \
    --max_relative_positions 16 \
    --norm_type "batch" \
    --activation_type "softsign" \
    --reg_loss_weight 1.0 \
    --tran_loss_weight 1.0 \
    --label_smoothing 0.0 \
    --optimizer "adam" \
    --slr_learning_rate 0.001 \
    --slt_learning_rate 0.001 \
    --weight_decay 0.0001\
    --decrease_factor 0.5 \
    --patience 3 \
    --reg_beam_size 5 \
    --max_output_length 500 \
    --text_beam_size 5 \
    --alpha 2 \
    --batch_size 5 \
    --check_point $avg_path/average_ckpt.pt \
    --max_epoch 100 \
    --print_step 100
