import multiprocessing as mp
import os
import pickle

from argparse import ArgumentParser
import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import warnings

from gnn2_model import GNN_2_Model as Model
from gnn2_utils import *


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')
print('using device:', device)

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

warnings.filterwarnings('ignore')

parser = ArgumentParser()
parser.add_argument('--data_file', required=True, type=str)
# parser.add_argument('--threshold', required=True,  type=float)
parser.add_argument('--model_save_file', required=True, type=str)
parser.add_argument('--hidden_layer_sizes', nargs='+', required=True, type=int)
parser.add_argument('--num_heads', nargs='+', required=True, type=int)
parser.add_argument('--use_max_freq_feat', action='store_true')
parser.add_argument('--max_freq_emb_size', required=False, default=128, type=int)
parser.add_argument('--use_caption', action='store_true')
parser.add_argument('--res_file', required=True, type=str)
args = parser.parse_args()
print(args)

comp_data = pickle.load(open(args.data_file, 'rb'))
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}
# print(f'Length of comp_data == {len(comp_data)}') #5883
# if ('S0022309300001034', 0) in comp_data_dict.keys():
#     print('Foundddd')

lm_name = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')
os.makedirs(os.path.dirname(os.path.abspath(args.model_save_file)), exist_ok=True)

th = 0.7

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
    'use_max_freq_feat': args.use_max_freq_feat,
    'max_freq_emb_size': args.max_freq_emb_size,
    'use_caption': args.use_caption,
}
model = Model(model_args).to(device)

def get_thresholded_labels(labels, softmax_logits, th=0.9):
    
    new_labels = labels
    for idxx, x in enumerate(labels):
        if x==2:
            if softmax_logits[idxx][2] < th: 
                new_labels[idxx] = 0
            else:
                new_labels[idxx] = x
        else:
            new_labels[idxx] = x
    return new_labels


def get_edge_gid_pred_labels(comp_gid_labels, comp_gid_logits, edge_logits, num_rows: int, num_cols: int):
    row_col_labels = [0 if c == 3 else c for c in comp_gid_labels]
    row_labels, col_labels = rectify_comp_labels(row_col_labels[:num_rows], row_col_labels[-num_cols:])
    assert row_labels.count(1) * col_labels.count(1) + row_labels.count(2) * col_labels.count(2) == 0
    r, c = Tensor(row_labels).unsqueeze(1), Tensor(col_labels).unsqueeze(0)
    comp_cells = np.where((r * c).view(-1) == 2)[0]

    if sum(row_labels + col_labels) > 0:
        if 1 in row_labels:
            assert 2 in col_labels
            if 3 in comp_gid_labels[-num_cols:]:
                col_gid_probs = F.softmax(comp_gid_logits[-num_cols:], dim=1)
                idx = np.where(col_gid_probs[:, 3] == col_gid_probs[col_gid_probs.argmax(1) == 3, 3].max())[0][0]
                col_labels[idx] = 3
        else:
            assert 1 in col_labels and 2 in row_labels
            if 3 in comp_gid_labels[:num_rows]:
                row_gid_probs = F.softmax(comp_gid_logits[:num_rows], dim=1)
                idx = np.where(row_gid_probs[:, 3] == row_gid_probs[row_gid_probs.argmax(1) == 3, 3].max())[0][0]
                row_labels[idx] = 3

    df = pd.DataFrame(Model.get_edges(num_rows, num_cols).tolist())
    df.columns = ['src', 'dst']
    df['wt'] = edge_logits
    idx = (df.groupby('src')['wt'].transform(max) == df['wt']) & df['src'].isin(comp_cells)
    df.drop('wt', inplace=True, axis=1)
    edges = df[idx].applymap(lambda x: (x // num_cols, x % num_cols)).values.tolist()
    return row_labels + col_labels, list(idx.astype(int).values), edges


def get_batch_edge_gid_pred_labels(comp_gid_logits, edge_logits, num_rows: list, num_cols: list):
    base_comp_gid, base_edge = 0, 0
    pred_edge_labels, pred_edges, pred_row_col_gid_labels = [], [], []
    comp_gid_labels = comp_gid_logits.argmax(1).tolist()
    
    softmax_logits = F.softmax(comp_gid_logits, dim=1).cpu().detach().tolist()
    labels = comp_gid_logits.argmax(1).cpu().detach().tolist()
#     global th
#     print(f'th = {th}')
    new_labels = get_thresholded_labels(labels, softmax_logits, th)
    
    comp_gid_labels = new_labels
    
    for r, c in zip(num_rows, num_cols):
        num_comp_labels, num_edge_logits = r + c, r * c * (r + c - 1)
        row_col_gid_labels, edge_labels, edges = get_edge_gid_pred_labels(
            comp_gid_labels[base_comp_gid:base_comp_gid+num_comp_labels], 
            comp_gid_logits[base_comp_gid:base_comp_gid+num_comp_labels], 
            edge_logits[base_edge:base_edge+num_edge_logits], r, c)
        pred_edge_labels += edge_labels
        pred_edges.append(edges)
        pred_row_col_gid_labels += row_col_gid_labels
        base_comp_gid += num_comp_labels
        base_edge += num_edge_logits
    return pred_row_col_gid_labels, pred_edge_labels, pred_edges


def infer_model():
    model.eval()
    identifier = []
    y_comp_pred, ret_comp_pred = [], []
    y_edge_pred, ret_edge_pred = [], []
    print(f'th = {th}')

    with torch.no_grad():

        tepoch = tqdm(data_loader, unit='batch')
        for batch_data in tepoch:
            tepoch.set_description('infer mode')
            comp_gid_logits, edge_logits = model(batch_data)
            
            softmax_logits = F.softmax(comp_gid_logits, dim=1).cpu().detach().tolist()
            labels = comp_gid_logits.argmax(1).cpu().detach().tolist()
#             global th
            new_labels = get_thresholded_labels(labels, softmax_logits, th)
            
#             y_comp_pred += comp_gid_logits.argmax(1).cpu().detach().tolist()
            y_comp_pred += new_labels
            
#             y_comp_pred += comp_gid_logits.argmax(1).cpu().detach().tolist()
            num_rows, num_cols = [x['num_rows'] for x in batch_data], [x['num_cols'] for x in batch_data]
            pred_comp_gid_labels, pred_edge_labels, batch_pred_edges = get_batch_edge_gid_pred_labels(
                comp_gid_logits.cpu().detach(), edge_logits.cpu().detach(), num_rows, num_cols)
            y_edge_pred += pred_edge_labels
            ret_edge_pred += batch_pred_edges

            base_comp_gid = 0
            for x in batch_data:
                identifier.append((x['pii'], x['t_idx']))
                comp_dict = dict()
                comp_dict['row'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_rows']]
                base_comp_gid += x['num_rows']
                comp_dict['col'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_cols']]
                base_comp_gid += x['num_cols']
                ret_comp_pred.append(comp_dict)


    ret_tuples_pred, type_labels = [], []
    for (pii, t_idx), edges, comp_gid_pred in zip(identifier, ret_edge_pred, ret_comp_pred):
        pred_tuples, label = get_pred_tuples(comp_data_dict[(pii, t_idx)], edges, comp_gid_pred)
        ret_tuples_pred.append(pred_tuples)
        type_labels.append(label)
        
#     pickle.dump(pii_tables, open('pii_pred_tables.pkl', 'wb'))

    return identifier, (y_comp_pred, ret_comp_pred), ret_edge_pred, ret_tuples_pred, type_labels


model.load_state_dict(torch.load(args.model_save_file, map_location=device))
model = model.to(device)

res = dict()
res['identifier'], (res['comp_gid_pred_orig'], res['comp_gid_pred']), res['edge_pred'], res['tuples_pred'], res['type_labels'] = infer_model()
k = 'comp_gid_pred_orig'
comp_gid_pred_orig = []
base = 0
for pii_t_idx in res['identifier']:
    c = comp_data_dict[pii_t_idx]
    d = dict()
    d['row'] = res[k][base:base+c['num_rows']]
    base += c['num_rows']
    d['col'] = res[k][base:base+c['num_cols']]
    base += c['num_cols']
    comp_gid_pred_orig.append(d)
res[k] = comp_gid_pred_orig

for n, f in violation_funcs.items():
    violations, total = 0, 0
    for table in res[k]:
        v, t = f(table)
        violations += v
        total += t
    print(f'{n}: {violations}/{total}')
    res[n] = violations

pickle.dump(res, open(args.res_file, 'wb'))
