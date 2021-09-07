#shellcheck disable=SC2034

CUDA_VISIBLE_DEVICES=1


BASE_PATH=log/piSLRTc
python -m train \
  --mode "train" \
  --log_dir $BASE_PATH \
  --data_path "../data-bin/PHOENIX2014T/" \
  --input_size 1024 \
  --embedding_dim 512 \
  --hidden_size 512 \
  --num_heads 8 \
  --num_layers 6 \
  --max_relative_positions 16 \
  --norm_type "batch" \
  --activation_type "softsign" \
  --reg_loss_weight 1.0 \
  --tran_loss_weight 1.0 \
  --label_smoothing 0.0001 \
  --optimizer "adam" \
  --slr_learning_rate 0.001 \
  --slt_learning_rate 0.2 \
  --warmup_steps 4000 \
  --weight_decay 0.0001\
  --decrease_factor 0.5 \
  --patience 5 \
  --reg_beam_size 5 \
  --max_output_length 300 \
  --text_beam_size 5 \
  --alpha 2 \
  --batch_size 5 \
  --check_point $BASE_PATH/ep.pkl \
  --max_epoch 60 \
  --print_step 100 \
