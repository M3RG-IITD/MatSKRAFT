from argparse import ArgumentParser
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

from utils import get_tuples_metrics, get_composition_metrics, violation_funcs, get_composition_metrics_without_ids


parser = ArgumentParser()
parser.add_argument('--gnn1_variant', required=True, type=str)
parser.add_argument('--gnn2_variant', required=True, type=str)
parser.add_argument('--split', choices=['val', 'test'], default='test', type=str)
args = parser.parse_args()

table_dir = '../../data'

val_test_data = pickle.load(open(os.path.join(table_dir, 'val_test_data_verified_2.pkl'), 'rb'))
split_dict = pickle.load(open(os.path.join(table_dir, 'train_val_test_split.pkl'), 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in val_test_data}
split = args.split


def get_gold_tuples(pii, t_idx):
    '''
    Function for obtaining gold tuples using paper PII and table index
    '''
    c = comp_data_dict[(pii, t_idx)]
    tuples = []
    for i in range(c['num_rows']):
        for j in range(c['num_cols']):
            if c['full_comp'][i][j] is None: continue
            for k in range(len(c['full_comp'][i][j])):
                prefix = f'{pii}_{t_idx}_{i}_{j}_{k}'
                for x in c['full_comp'][i][j][k]:
                    if x[2] == 0: continue
                    gid = prefix if x[0] is None else prefix + '_' + x[0]
                    tuples.append((gid, x[1], round(float(x[2]), 5), x[3]))
    return tuples


def cnt_violations(d: dict, scc_table=False):
    if scc_table:
        cnt_3 = (d['row'] + d['col']).count(1)
        return cnt_3 * (cnt_3 - 1) // 2
    return sum(f(d)[0] for f in violation_funcs.values())


def process_mids(l):
    return [1 if x == 3 else 0 for x in l]


def get_split_res(scc_res_file, non_scc_res_file, fold):
#     folder_name = ['dir_res_abl_anno_algo', 'dir_res_abl_caption', 'dir_res_abl_const_learn', 'dir_res_abl_threshold', 'dir_res_no_abl']
#     print('path scc_res_file:',os.path.join(table_dir, fold, scc_res_file))
    scc_res = pickle.load(open(os.path.join(table_dir, fold, scc_res_file), 'rb'))[split]
    
#     print('path non_scc_res_file:',os.path.join(table_dir, fold, non_scc_res_file))
    non_scc_res = pickle.load(open(os.path.join(table_dir, fold, non_scc_res_file), 'rb'))[split]

    all_gold_tuples, all_pred_tuples = [], []
    all_gold_mids, all_pred_mids = [], []
    all_gold_table_type, all_pred_table_type = [], []
    violations = 0

    for i, pii_t_idx in enumerate(split_dict[split]):
        c = comp_data_dict[pii_t_idx]
        all_gold_tuples += get_gold_tuples(*pii_t_idx)
        all_gold_mids += c['gid_row_label'] + c['gid_col_label']
        if c['regex_table'] == 1:
            all_gold_table_type.append(0)
        elif c['sum_less_100'] == 0 and c['comp_table']:
            all_gold_table_type.append(1)
        elif c['sum_less_100'] == 1:
            all_gold_table_type.append(2)
        else:
            all_gold_table_type.append(3)

        scc_idx = scc_res['identifier'].index(pii_t_idx)
        non_scc_idx = non_scc_res['identifier'].index(pii_t_idx)
        assert scc_idx == non_scc_idx
        if scc_res['scc_pred'][scc_idx] == 1:
#             print(scc_res['tuples_pred'][scc_idx])
#             print()
            all_pred_tuples += scc_res['tuples_pred'][scc_idx]
            all_pred_mids += scc_res['gid_pred_orig'][scc_idx]['row'] + scc_res['gid_pred_orig'][scc_idx]['col']
            all_pred_table_type.append(0)
            violations += cnt_violations(scc_res['gid_pred_orig'][scc_idx], scc_table=True)
        else:
            all_pred_tuples += non_scc_res['tuples_pred'][non_scc_idx]
            all_pred_mids += process_mids(non_scc_res['comp_gid_pred_orig'][non_scc_idx]['row'] + non_scc_res['comp_gid_pred_orig'][non_scc_idx]['col'])
            all_pred_table_type.append(non_scc_res['type_labels'][non_scc_idx])
            violations += cnt_violations(non_scc_res['comp_gid_pred_orig'][non_scc_idx], scc_table=False)

    table_type_acc = accuracy_score(all_gold_table_type, all_pred_table_type)
    table_type_fscore = f1_score(all_gold_table_type, all_pred_table_type, average='weighted')
    table_type_precision = precision_score(all_gold_table_type, all_pred_table_type, average='weighted')
    table_type_recall = recall_score(all_gold_table_type, all_pred_table_type, average='weighted')

    mid_fscore = f1_score(all_gold_mids, all_pred_mids)
    mid_precision = precision_score(all_gold_mids, all_pred_mids)
    mid_recall = recall_score(all_gold_mids, all_pred_mids)
    mid_accuracy = accuracy_score(all_gold_mids, all_pred_mids)

    tuple_metrics = get_tuples_metrics(all_gold_tuples, all_pred_tuples)
    mat_metrics = get_composition_metrics(all_gold_tuples, all_pred_tuples)
    mat_metrics_without_id = get_composition_metrics_without_ids(all_gold_tuples, all_pred_tuples)
    
#     print(f'For all Table types  ==> Len of pred tuples = {len(all_pred_tuples)}, Len of gold tuples = {len(all_gold_tuples)}')
    
    pickle.dump(all_pred_tuples, open(f'all_pred_tuples_{split}.pkl', 'wb'))
    pickle.dump(all_gold_tuples, open(f'all_gold_tuples_{split}.pkl', 'wb'))

    return (table_type_acc, table_type_precision, table_type_recall, table_type_fscore), (mid_fscore, mid_precision, mid_recall, mid_accuracy), tuple_metrics, mat_metrics, violations, mat_metrics_without_id


def compute_metrics(scc_variant, non_scc_variant):
    (all_table_type_acc, all_table_type_fscore, all_table_type_precision, all_table_type_recall) , (all_mid_fscore, all_mid_precision, all_mid_recall, all_mid_accuracy), all_tuple_metrics, all_mat_metrics, all_violations, all_seeds, all_mat_metrics_final = ([], [], [], []), ([], [], [], []), [], [], [], [], []
    
    folder_name = ['dir_res_abl_anno_algo', 'dir_res_abl_caption', 'dir_res_abl_const_learn', 'dir_res_abl_threshold', 'dir_res_no_abl']
    for fold in folder_name:
        for seed_1 in [1]:
            for seed_2 in [0]:
                (table_type_acc, table_type_precision, table_type_recall, table_type_fscore), (mid_fscore, mid_precision, mid_recall, mid_accuracy), tuple_metrics, mat_metrics, violations, mat_metrics_without_ids = \
                get_split_res(f'res_{scc_variant}_{seed_1}_0.7.pkl', f'res_{non_scc_variant}_{seed_2}_0.7.pkl', fold)

                all_table_type_acc.append(table_type_acc)
                all_table_type_fscore.append(table_type_fscore)
                all_table_type_precision.append(table_type_precision)
                all_table_type_recall.append(table_type_recall)
                all_mid_fscore.append(mid_fscore)
                all_mid_precision.append(mid_precision)
                all_mid_recall.append(mid_recall)
                all_mid_accuracy.append(mid_accuracy)
                all_tuple_metrics.append(tuple_metrics)
                all_mat_metrics.append(mat_metrics)
                all_violations.append(violations)
                all_seeds.append([seed_1, seed_2])
                all_mat_metrics_final.append(mat_metrics_without_ids)

    for ind, fold in enumerate(folder_name):
#         for seed_pair, metrics in zip(all_seeds, all_mat_metrics_without_ids):
        print(f'For {fold} study, material metrics obtained is {all_mat_metrics_final[ind]}')
        #print(all_mat_metrics)
        print()

compute_metrics(args.gnn1_variant, args.gnn2_variant)
