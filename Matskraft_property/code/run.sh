#!/bin/sh

seed=$1
# variant="gnn_property_segregation_lr_5e-5_1e-5_30_bs_4_retrain_v100_${seed}"
# variant="gnn_property_segregation_lr_5e-5_1e-5_30_bs_1_retrain_a100_${seed}"
# variant="gnn_property_segregation_lr_5e-5_1e-5_30_bs_4_retrain_a100_${seed}"
variant="augmented_gnn_ps_lr_5e-5_1e-5_30_bs_2_heavier_2_captions_${seed}_160"

model_save_file="../saved_models_captions/model_${variant}.bin"
res_file="res_${variant}.pkl"
out_file="out_${variant}"
err_file="err_${variant}"

# python -u train_gnn.py --seed $seed --hidden_layer_sizes 256 128 64 --num_heads 4 4 4 --num_epochs 15 --use_regex_feat --use_max_freq_feat --lr 3e-4 --lm_lr 1e-5 --add_constraint --c_loss_lambda 50.0 --gid_loss_lambda 1.0 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
python -u train_gnn.py --seed $seed --hidden_layer_sizes 2048 1024 512 --num_heads 4 4 4 --num_epochs 30 --lr 5e-5 --lm_lr 1e-5  --add_constraint --c_loss_lambda 50.0 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file

# lm_lr 1e-4 gives high accuracy