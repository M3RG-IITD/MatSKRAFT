from argparse import ArgumentParser
import os
import pickle

from gnn2_utils import violation_funcs


parser = ArgumentParser()
parser.add_argument('--gnn1_res_file', required=True, type=str)
parser.add_argument('--gnn2_res_file', required=True, type=str)
parser.add_argument('--res_file', required=True, type=str)
args = parser.parse_args()

gnn1_res = pickle.load(open(args.gnn1_res_file, 'rb'))
gnn2_res = pickle.load(open(args.gnn2_res_file, 'rb'))

assert gnn1_res['identifier'] == gnn2_res['identifier']

res_discomat = {'identifier': [], 'pred_table_type_labels': [], 'pred_row_col_labels': [], 'pred_edges': [], 'pred_tuples': []}

for idx, scc_label in enumerate(gnn1_res['scc_pred']):
    res_discomat['identifier'].append(gnn1_res['identifier'][idx])
    if scc_label == 1:
        res_discomat['pred_table_type_labels'].append(0)
        res_discomat['pred_row_col_labels'].append({k: [3 if l == 1 else l for l in gnn1_res['gid_pred'][idx][k]] for k in ['row', 'col']})
        res_discomat['pred_edges'].append([])
        res_discomat['pred_tuples'].append(gnn1_res['tuples_pred'][idx])
    else:
        assert gnn2_res['type_labels'][idx] in [1, 2, 3] # 0 -> SCC, 1 -> MCC-CI, 2 -> MCC-PI, 3 -> NC
        res_discomat['pred_table_type_labels'].append(gnn2_res['type_labels'][idx])
        res_discomat['pred_row_col_labels'].append(gnn2_res['comp_gid_pred'][idx])
        res_discomat['pred_edges'].append(gnn2_res['edge_pred'][idx])
        res_discomat['pred_tuples'].append(gnn2_res['tuples_pred'][idx])

for n, f in violation_funcs.items():
    violations, total = 0, 0
    for table in res_discomat['pred_row_col_labels']:
        v, t = f(table)
        violations += v
        total += t
    print(f'{n}: {violations}/{total}')
    res_discomat[n] = violations

pickle.dump(res_discomat, open(args.res_file, 'wb'))
