#!/bin/sh

python get_regex_from_texts.py


python -u gnn1.py --data_file ../../data/all_disco_data.pkl --model_save_file ../../models/model_gnn1_discomat_1.bin --hidden_layer_sizes 256 128 64 --num_heads 4 4 4 --use_regex_feat --use_max_freq_feat --res_file res_gnn1.pkl

python -u gnn2.py --data_file ../../data/all_disco_data.pkl --model_save_file ../../models/model_gnn2_discomat_0.bin --hidden_layer_sizes 128 128 64 --num_heads 6 4 4 --use_max_freq_feat --max_freq_emb_size 128 --use_caption --res_file res_gnn2.pkl

python -u generate_final_res.py --gnn1_res_file res_gnn1.pkl --gnn2_res_file res_gnn2.pkl --res_file res_discomat.pkl
