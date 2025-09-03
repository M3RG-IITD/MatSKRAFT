#!/bin/sh

#seed=$1
# variant="gnn_property_segregation_lr_5e-5_1e-5_30_bs_4_retrain_v100_${seed}"
# variant="gnn_property_segregation_lr_5e-5_1e-5_30_bs_1_retrain_a100_${seed}"
# variant="gnn_property_segregation_lr_5e-5_1e-5_30_bs_4_retrain_a100_${seed}"
# variant="gnn_ps_lr_5e-5_1e-5_30_bs_2_${seed}"
model_variant="best_model_2"

#model_save_file="../../saved_models/model_${variant}.bin"
res_file="res_${model_variant}_inf_discomat_res.pkl"
out_file="out_${model_variant}_inf_discomat"
err_file="err_${model_variant}_inf_discomat"

# python -u train_gnn.py --seed $seed --hidden_layer_sizes 256 128 64 --num_heads 4 4 4 --num_epochs 15 --use_regex_feat --use_max_freq_feat --lr 3e-4 --lm_lr 1e-5 --add_constraint --c_loss_lambda 50.0 --gid_loss_lambda 1.0 --model_save_file $model_save_file --res_file $res_file >> $out_file 2>> $err_file
time python -u test_th_gnn_separate_split_for_inferencing.py --seeds 2 --alphas 0.75 --hidden_layer_sizes 2048 1024 --num_heads 4 4 --lr 5e-5 --lm_lr 1e-5  --add_constraint --use_caption --c_loss_lambda 50.0 --model_variant $model_variant --res_file $res_file >> $out_file 2>> $err_file

# lm_lr 1e-4 gives high accuracy