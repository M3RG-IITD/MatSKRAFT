from argparse import ArgumentParser
import multiprocessing as mp
import os
import pickle
import sys
sys.path.append('..')

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from gnn1_model import GNN_1_Model as Model
from regex_lib import parse_composition
from gnn1_utils import *

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# device = torch.device('cpu')
print('using device:', device)


parser = ArgumentParser()
parser.add_argument('--data_file', required=True, type=str)
parser.add_argument('--model_save_file', required=True, type=str)
parser.add_argument('--hidden_layer_sizes', nargs='+', required=True, type=int)
parser.add_argument('--num_heads', nargs='+', required=True, type=int)
parser.add_argument('--use_regex_feat', action='store_true')
parser.add_argument('--use_max_freq_feat', action='store_true')
parser.add_argument('--regex_emb_size', required=False, default=256, type=int)
parser.add_argument('--max_freq_emb_size', required=False, default=256, type=int)
parser.add_argument('--res_file', required=True, type=str)
parser.add_argument('--add_constraint', action='store_true')
parser.add_argument('--c_loss_lambda', required=False, default=50.0, type=float)

args = parser.parse_args()
print(args)

alpha = 0.7

comp_data = pickle.load(open(args.data_file, 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}

lm_name = 'm3rg-iitd/matscibert'
table_dir = '../../data'
cache_dir = os.path.join(table_dir, '.cache')
os.makedirs(os.path.dirname(os.path.abspath(args.model_save_file)), exist_ok=True)

if args.use_regex_feat:
    for c in tqdm(comp_data):
        try:
            c['regex_feats'] = get_regex_feats(c['act_table'])
        except:
            c['regex_feats'] = []
            

if args.use_max_freq_feat:
    for c in comp_data:
        c['max_freq_feat'] = get_max_freq_feat(c['act_table'])

torch.set_deterministic(True)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = False

dataset = TableDataset(comp_data)

batch_size = 8
num_workers = mp.cpu_count()
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=lambda x: x)

model_args = {
    'hidden_layer_sizes': args.hidden_layer_sizes,
    'num_heads': args.num_heads,
    'lm_name': lm_name,
    'cache_dir': cache_dir,
    'use_regex_feat': args.use_regex_feat,
    'use_max_freq_feat': args.use_max_freq_feat,
    'regex_emb_size': args.regex_emb_size,
    'max_freq_emb_size': args.max_freq_emb_size,
}
model = Model(model_args).to(device)


def get_pred_tuples(pii_t_idx: tuple, regex_table: list, orient: str, gid):
    gid_list = []
    c = comp_data_dict[pii_t_idx]
    table = c['act_table']
    pii, t_idx = pii_t_idx
    if orient == 'row':
        for i in range(len(table)):
            if gid is not None and table[i][gid]:
                gid_list.append('_' + table[i][gid])
            else:
                gid_list.append('')
    else:
        for j in range(len(table[0])):
            if gid is not None and table[gid][j]:
                gid_list.append('_' + table[gid][j])
            else:
                gid_list.append('')
    tuples = []
    for i in range(len(table)):
        for j in range(len(table[0])):
#             if (regex_table[i][j] is None) or (i>2 and j>2): continue
            if regex_table[i][j] is None: continue
            prefix = f'{pii}_{t_idx}_{i}_{j}_0'
            for x in regex_table[i][j]:
                if x[1] == 0: continue
                gid = gid_list[i] if orient == 'row' else gid_list[j]
                tuples.append((prefix + gid, x[0], x[1], pred_cell_mol_wt(c, i, j, pii)))
    return tuples


def get_regex_table_and_orient(table):
    regex_table = []
    regex_label = 0
    for ind_r, r in enumerate(table):
        res_r = []
        for ind_c, cell in enumerate(r):
            if ind_r>2 and ind_c>2:
                res_r.append(None)
                continue
            comp = parse_composition(cell)
            if len(comp) == 0 or len(comp[0][0]) == 1:
                res_r.append(None)
                continue
            l = comp[0][0]
            new_l = []
            for x in l:
                if type(x[1]) == float:
                    x = (x[0], round(x[1], 5))
                elif type(x[1]) == int:
                    x = (x[0], float(x[1]))
                new_l.append(x)
            if all(type(x[1]) == float for x in new_l):
                regex_label = 1
                res_r.append(new_l)
            else:
                res_r.append(None)
        regex_table.append(res_r)
    if regex_label == 0:
        return None, None
    row_max = 0
    for r in range(len(table)):
        curr = 0
        for comp in regex_table[r]:
            if type(comp) == list:
                curr += 1
        row_max = max(row_max, curr)
    col_max = 0
    for c in range(len(table[0])):
        curr = 0
        for r in range(len(table)):
            if type(regex_table[r][c]) == list:
                curr += 1
        col_max = max(col_max, curr)
    if row_max <= col_max:
        return regex_table, 'row'
    return regex_table, 'col'


def get_gid_labels_and_tuples(gid_logits, scc_label: int, pii_t_idx: tuple, num_rows: int, num_cols: int):
    row_gid_labels, col_gid_labels = [0] * num_rows, [0] * num_cols
    if scc_label == 0:
        return row_gid_labels + col_gid_labels, []
    regex_table, orient = get_regex_table_and_orient(comp_data_dict[pii_t_idx]['act_table'])
    if orient is None:
        return row_gid_labels + col_gid_labels, []
    gid = None
    if orient == 'row':
        gid_col_probs = F.softmax(gid_logits[num_rows:], dim=1)
        gid_idx = gid_col_probs[:, 1].argmax()
        if gid_col_probs[gid_idx, 1] > 0.5:
            col_gid_labels[gid_idx] = 1
            gid = gid_idx
    else:
        gid_row_probs = F.softmax(gid_logits[:num_rows], dim=1)
        gid_idx = gid_row_probs[:, 1].argmax()
        if gid_row_probs[gid_idx, 1] > 0.5:
            row_gid_labels[gid_idx] = 1
            gid = gid_idx
    return row_gid_labels + col_gid_labels, get_pred_tuples(pii_t_idx, regex_table, orient, gid)


def get_batch_gid_labels_and_tuples(gid_logits, scc_labels: list, pii_t_idxs: list, num_rows: list, num_cols: list):
    base_gid = 0
    pred_gid_labels, pred_tuples = [], []
    for pii_t_idx, regex_label, r, c in zip(pii_t_idxs, scc_labels, num_rows, num_cols):
        num_gid_logits = r + c
        gids_labels, tuples = get_gid_labels_and_tuples(
            gid_logits[base_gid:base_gid+num_gid_logits], regex_label, pii_t_idx, r, c)
        pred_gid_labels += gids_labels
        pred_tuples.append(tuples)
        base_gid += num_gid_logits
    return pred_gid_labels, pred_tuples


def infer_model():
    model.eval()
    identifier = []
    y_scc_pred = []
    y_gids_pred, ret_gids_pred = [], []
    ret_tuples_pred = []

    with torch.no_grad():
        tepoch = tqdm(data_loader, unit='batch')
        print(f'alpha = {alpha}')
        for batch_data in tepoch:
            tepoch.set_description('infer mode')
            scc_logits, gid_logits = model(batch_data)
            
            scc_logits_prob = F.softmax(scc_logits, dim=1)
#             print(scc_logits_prob.shape) #torch.Size([8, 2])
#             print(scc_logits_prob)
            for ind, elem in enumerate(scc_logits_prob.argmax(dim=1)):
                if elem == 1 and scc_logits_prob[ind][elem]<alpha:
                    leng = len(scc_logits_prob[ind])
                    scc_logits[ind] =  torch.tensor([0.0] * leng)
                    scc_logits[ind][0] = torch.tensor(1.0)
            
            pred_regex_labels = scc_logits.argmax(1).cpu().detach().tolist()
            y_scc_pred += pred_regex_labels
            pred_gid_labels = gid_logits.argmax(1).cpu().detach().tolist()
            base = 0
            for p, x in zip(pred_regex_labels, batch_data):
                if p == 1:
                    y_gids_pred += pred_gid_labels[base:base+x['num_rows']+x['num_cols']]
                else:
                    y_gids_pred += [0] * (x['num_rows'] + x['num_cols'])
                base += x['num_rows'] + x['num_cols']

            num_rows, num_cols = [x['num_rows'] for x in batch_data], [x['num_cols'] for x in batch_data]
            pii_t_idxs = [(x['pii'], x['t_idx']) for x in batch_data]
            identifier += pii_t_idxs
            pred_gid_labels, pred_tuples = get_batch_gid_labels_and_tuples(
                gid_logits.cpu().detach(), pred_regex_labels, pii_t_idxs, num_rows, num_cols)
            ret_tuples_pred += pred_tuples

            base_gid = 0
            for x in batch_data:
                gid_dict = dict()
                gid_dict['row'] = pred_gid_labels[base_gid:base_gid+x['num_rows']]
                base_gid += x['num_rows']
                gid_dict['col'] = pred_gid_labels[base_gid:base_gid+x['num_cols']]
                base_gid += x['num_cols']
                ret_gids_pred.append(gid_dict)

    return identifier, y_scc_pred, (y_gids_pred, ret_gids_pred), ret_tuples_pred


#model.load_state_dict(torch.load(args.model_save_file, map_location=torch.device('cpu')))
model.load_state_dict(torch.load(args.model_save_file, map_location=device))
model = model.to(device)

res = dict()
print()

res['identifier'], res['scc_pred'], (res['gid_pred_orig'], res['gid_pred']), res['tuples_pred'] = infer_model()
k = 'gid_pred_orig'
gid_pred_orig = []
base = 0
for pii_t_idx in res['identifier']:
    c = comp_data_dict[pii_t_idx]
    d = dict()
    d['row'] = res[k][base:base+c['num_rows']]
    base += c['num_rows']
    d['col'] = res[k][base:base+c['num_cols']]
    base += c['num_cols']
    gid_pred_orig.append(d)
res[k] = gid_pred_orig

violations, total = 0, 0
for table in res[k]:
    v, t = cnt_3_3_violations(table)
    violations += v
    total += t
print(f'3_3_violations: {violations}/{total}')
res['3_3_violations'] = violations

pickle.dump(res, open(args.res_file, 'wb'))
