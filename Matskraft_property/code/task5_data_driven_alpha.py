"""
Task 5: Data-Driven Per-Property Confidence Threshold Selection
================================================================

All alpha decisions are made exclusively on the VALIDATION split.
The test split is used only for final held-out reporting and is NEVER
consulted during any threshold selection step.

Pipeline
--------
  Step 1  Compute per-property row/column metrics at alpha=0 (no threshold)
          on the validation split.  Properties where
              Recall - Precision > 3 percentage points  (gap > 0.03)
          are identified as candidates for confidence thresholding.
          High recall with low precision indicates the model is
          over-predicting that class; a confidence threshold can reduce
          false positives without proportionally hurting true positives.

  Step 2  For each candidate property, sweep alpha INDEPENDENTLY over
              {0.00, 0.05, 0.10, ..., 0.95, 1.00}   (step = 0.05, 21 values)
          Only that property's predictions are filtered at each alpha;
          all other properties remain unthresholded (alpha=0).
          The alpha that maximises the property-specific validation F1
          is the best raw alpha for that property.

  Step 3  Map each best raw alpha to a coarser representative bucket
          to reduce overfitting to the validation split:
              [0,   20]%  ->  0.10
              (20,  40]%  ->  0.30
              (40,  60]%  ->  0.50
              (60,  80]%  ->  0.70
              (80, 100]%  ->  0.90
          The '(' notation means strictly greater than the lower bound.

  Step 4  Apply the bucketed alpha to each candidate and re-evaluate
          validation F1.  Accept the threshold if and only if:
              F1(bucketed alpha) - F1(alpha=0) > 0.03   (> 3 pp)
          Properties that do not pass this check revert to alpha=0.
          This guards against small or noise-driven gains, especially
          for properties with few validation instances.

  Step 5  Apply all accepted per-property thresholds SIMULTANEOUSLY
          and report final metrics on:
            (a) Row/column classification level  (direct GNN output)
            (b) Tuple extraction level           (end-to-end pipeline)
          Both validation (decision basis) and test (held-out) results
          are reported.

Usage:
    python task5_data_driven_alpha.py \\
        --seeds 2 \\
        --hidden_layer_sizes 2048 1024 \\
        --num_heads 4 4 \\
        --add_constraint \\
        --use_caption \\
        --model_variant best_model_2 \\
        --out_dir ./analysis_outputs/task5
"""

# ── Imports ───────────────────────────────────────────────────────────────────

from argparse import ArgumentParser
import math
import multiprocessing as mp
import os
import pickle
import sys
sys.path.append('..')

from sklearn.metrics import classification_report
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import Patch


# ── Nature journal figure settings ───────────────────────────────────────────
matplotlib.rcParams.update({
    'font.family':       'sans-serif',
    'font.sans-serif':   ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size':          6,
    'axes.labelsize':     7,
    'axes.titlesize':     7,
    'xtick.labelsize':    6,
    'ytick.labelsize':    6,
    'legend.fontsize':    5.5,
    'legend.framealpha':  0.85,
    'legend.edgecolor':   '0.8',
    'lines.linewidth':    1.0,
    'lines.markersize':   4,
    'axes.linewidth':     0.5,
    'xtick.major.width':  0.5,
    'ytick.major.width':  0.5,
    'xtick.major.size':   2.5,
    'ytick.major.size':   2.5,
    'grid.linewidth':     0.4,
    'grid.alpha':         0.3,
    'savefig.dpi':        600,
    'savefig.bbox':       'tight',
    'figure.dpi':         150,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'pdf.fonttype':       42,
    'ps.fonttype':        42,
})

# Wong (2011) colorblind-safe palette
C = {
    'blue':       '#0072B2',
    'orange':     '#E69F00',
    'green':      '#009E73',
    'sky':        '#56B4E9',
    'vermillion': '#D55E00',
    'purple':     '#CC79A7',
    'yellow':     '#F0E442',
    'gray':       '#999999',
    'lightgray':  '#CCCCCC',
    'black':      '#000000',
}

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from gnn_model import GNN_Model as Model
from utils import *
# Use comp_data = pickle.load(open(os.path.join(table_dir, 'final_val_test_data_with_id_with_tuple_revised_updated.pkl'), 'rb')) in utils
from units import *
from post_processing import *
from post_processing_2 import *
import re
import math 

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
pd.set_option('display.max_columns', None)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('Using device:', device)


# ── Argument parser ───────────────────────────────────────────────────────────

parser = ArgumentParser()
parser.add_argument('--seeds',              nargs='+', required=True,  type=int)
parser.add_argument('--hidden_layer_sizes', nargs='+', required=True,  type=int)
parser.add_argument('--num_heads',          nargs='+', required=True,  type=int)
parser.add_argument('--num_epochs',         required=False, default=15,  type=int)
parser.add_argument('--lr',                 required=False, default=1e-3, type=float)
parser.add_argument('--lm_lr',              required=False, default=1e-5, type=float)
parser.add_argument('--add_constraint',     action='store_true')
parser.add_argument('--c_loss_lambda',      required=False, default=50.0, type=float)
parser.add_argument('--use_caption',        action='store_true')
parser.add_argument('--res_file',           required=False, type=str)
parser.add_argument('--model_variant',      required=True,  type=str)
parser.add_argument('--out_dir',            required=False,
                    default='./analysis_outputs/task5', type=str)
args = parser.parse_args()
print(args)
splits = ['val', 'test']   # ← add this

seed      = args.seeds[0]
lm_name   = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')
os.makedirs(args.out_dir, exist_ok=True)


# ── Dataset / loaders ────────────────────────────────────────────────────────

datasets = dict()
for split in splits:
    datasets[split] = TableDataset(
        [comp_data_dict[pii_t_idx] for pii_t_idx in train_val_test_split[split]])

batch_size  = 8
num_workers = mp.cpu_count()
loaders     = dict()
for split in splits:
    loaders[split] = DataLoader(
        datasets[split], batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        collate_fn=lambda x: x)


# ── Property names and class index mapping ───────────────────────────────────
# class 0  = background (no property)
# class 1  = glass identifier
# class 2..20 = materials properties  (prop_names[i] <-> class i+2)

prop_names = [
    'Density',                          # class  2
    'Glass transition temperature',     # class  3
    'Refractive index',                 # class  4
    'Abbe value',                       # class  5
    "Young's modulus",                  # class  6
    'Shear modulus',                    # class  7
    'Vickers hardness',                 # class  8
    'Poisson ratio',                    # class  9
    'Fracture toughness',               # class 10
    'Crystallization temp',             # class 11
    'Melting temp',                     # class 12
    'Electric conductivity',            # class 13
    'Dielectric constant',              # class 14  -- always excluded
    'Softening Point (Temperature)',    # class 15
    'Annealing Point (Temperature)',    # class 16
    'Thermal expansion coefficient',    # class 17
    'Liquidus temperature',             # class 18
    'Bulk modulus',                     # class 19
    'Activation energy',                # class 20
]

DIELECTRIC_CLS = prop_names.index('Dielectric constant') + 2   # 14


# ── Gold tuples ───────────────────────────────────────────────────────────────

conf_tables     = []
gold_tuples     = dict()
ret_tuples_gold = dict()
for split in splits:
    gold_tuples[split]     = []
    ret_tuples_gold[split] = []
    for pii, t_idx in train_val_test_split[split]:
        gold_tuples[split]         += get_gold_tuples(pii, t_idx)
        ret_tuples_gold[split].append(get_gold_tuples(pii, t_idx))


# ── Helper: parse column / row heading names ──────────────────────────────────

def clean_names(name):
    match = re.search(r'(\[.*\]|\(.*\)|\{.*\}|\<.*\>)([^a-zA-Z]*)', name)
    if match:
        unit_name    = match.group(1)[1:-1]
        cleaned_name = re.sub(
            r'(\[.*\]|\(.*\)|\{.*\}|\<.*\>)([^a-zA-Z]*)', '', name).strip()
    else:
        cleaned_name = name
        unit_name    = None
    return cleaned_name, unit_name


# ── get_pred_tuples (unchanged from original task1) ──────────────────────────

def get_pred_tuples(pii, t_idx, comp_gid_pred_dict, split, modify_tables=True):
    t_name  = pii + '_' + str(t_idx)
    c       = comp_data_dict[(pii, t_idx)]
    caption = c['caption']
    footer  = c['footer']
    table   = c['act_table']
    nr, nc  = c['num_rows'], c['num_cols']
    pr_rl, pr_cl = (comp_gid_pred_dict['pred_row_label'],
                    comp_gid_pred_dict['pred_col_label'])
    comp_gid_pred = (comp_gid_pred_dict['pred_row_label']
                     + comp_gid_pred_dict['pred_col_label'])
    assert len(comp_gid_pred) == nr + nc

    comp_gid_pred_alt = [2 if i in range(2, 21) else i for i in comp_gid_pred]
    row_ratio = comp_gid_pred_alt[:nr].count(2) / nr
    col_ratio = comp_gid_pred_alt[nr:].count(2) / nc

    if row_ratio <= col_ratio:
        for j in range(nc):
            table_np   = np.array(table)
            r_table    = list(table_np[:, j])
            pred_label = pr_cl[j]
            heading    = r_table[0]
            num_val    = r_table[1:]

            head_flag = check_heading(pii, t_idx, t_name, r_table, caption, pred_label)
            if not head_flag:
                comp_gid_pred_dict['pred_col_label'][j] = 0

            if pred_label == 0:
                mod_pred_label = direct_matching(heading)
                if mod_pred_label != 0:
                    comp_gid_pred_dict['pred_col_label'][j] = mod_pred_label

            tm_liq_list = ['t m liq', 'tm liq', 'tmliq']
            if any(e in heading.lower() for e in tm_liq_list) and pred_label != 18:
                comp_gid_pred_dict['pred_col_label'][j] = 18

            ea_list = ['E0', 'Ae', 'Ea', 'E a', 'Ec', 'E c', 'E A',
                       'E s', 'E dc', 'Edc', 'Es']
            if any(e in heading for e in ea_list) and pred_label == 0:
                clean_heading, unit = clean_names(heading)
                if unit is None:
                    foundd = False
                else:
                    arb = ['/mol', '/ mol', 'mol-1', 'mol -1',
                           '/at', '/ at', 'at-1', 'at -1', 'eV']
                    foundd = any(e in unit for e in arb)
                if foundd or 'activation ene' in caption.lower():
                    comp_gid_pred_dict['pred_col_label'][j] = 20

            if heading in ['Y (GPa)', 'Y(GPa)', 'y (GPa)', 'y(GPa)',
                           'Y (Gpa)', 'Y(Gpa)'] and pred_label != 6:
                comp_gid_pred_dict['pred_col_label'][j] = 6
            elif heading in ['m (GPa)', 'm(GPa)', 'M (GPa)', 'M(GPa)',
                             'M (Gpa)', 'M(Gpa)'] and pred_label != 7:
                comp_gid_pred_dict['pred_col_label'][j] = 7
            elif heading in ['H (GPa)', 'H(GPa)', 'h (GPa)', 'h(GPa)',
                             'H (Gpa)', 'H(Gpa)'] and pred_label != 8:
                comp_gid_pred_dict['pred_col_label'][j] = 8

            if 'exo' in heading and ('t' in heading.lower()
                                     or 'Temp' in caption):
                comp_gid_pred_dict['pred_col_label'][j] = 11
            elif 'endo' in heading and ('t' in heading.lower()
                                        or 'Temp' in caption):
                comp_gid_pred_dict['pred_col_label'][j] = 12

            power_regex = re.compile(r'10\s*[\*\^]?\*?\s*[-−–]\s*(\d+)')
            match = power_regex.search(heading)
            if match and pred_label in [13, 14, 17]:
                x      = int(match.group(1))
                values = [find_num(v) for v in num_val if find_num(v) is not None]
                numeric_values = [float(v) for v in values]
                median = np.median(numeric_values)
                if not math.isnan(median) and median > 0.1:
                    assert len(num_val) == nr - 1
                    for ind, val in enumerate(num_val):
                        new_val = find_num(
                            comp_data_dict[(pii, t_idx)]['act_table'][ind + 1][j])
                        if new_val is not None and modify_tables:
                            comp_data_dict[(pii, t_idx)]['act_table'][ind + 1][j] = \
                                str(new_val * 10 ** -x)
    else:
        for i in range(nr):
            table_np   = np.array(table)
            r_table    = list(table_np[i, :])
            pred_label = pr_rl[i]
            heading    = r_table[0]
            num_val    = r_table[1:]

            head_flag = check_heading(pii, t_idx, t_name, r_table, caption, pred_label)
            if not head_flag:
                comp_gid_pred_dict['pred_row_label'][i] = 0

            if pred_label == 0:
                mod_pred_label = direct_matching(heading)
                if mod_pred_label != 0:
                    comp_gid_pred_dict['pred_row_label'][i] = mod_pred_label

            tm_liq_list = ['t m liq', 'tm liq', 'tmliq']
            if any(e in heading.lower() for e in tm_liq_list) and pred_label != 18:
                comp_gid_pred_dict['pred_row_label'][i] = 18

            ea_list = ['E0', 'Ae', 'Ea', 'E a', 'Ec', 'E c', 'E A',
                       'E s', 'E dc', 'Edc', 'Es']
            if any(e in heading for e in ea_list) and pred_label == 0:
                clean_heading, unit = clean_names(heading)
                if unit is None:
                    foundd = False
                else:
                    arb = ['/mol', '/ mol', 'mol-1', 'mol -1',
                           '/at', '/ at', 'at-1', 'at -1', 'eV']
                    foundd = any(e in unit for e in arb)
                if foundd or 'activation ene' in caption.lower():
                    comp_gid_pred_dict['pred_row_label'][i] = 20

            if heading in ['Y (GPa)', 'Y(GPa)', 'y (GPa)', 'y(GPa)',
                           'Y (Gpa)', 'Y(Gpa)'] and pred_label != 6:
                comp_gid_pred_dict['pred_row_label'][i] = 6
            elif heading in ['m (GPa)', 'm(GPa)', 'M (GPa)', 'M(GPa)',
                             'M (Gpa)', 'M(Gpa)'] and pred_label != 7:
                comp_gid_pred_dict['pred_row_label'][i] = 7
            elif heading in ['H (GPa)', 'H(GPa)', 'h (GPa)', 'h(GPa)',
                             'H (Gpa)', 'H(Gpa)'] and pred_label != 8:
                comp_gid_pred_dict['pred_row_label'][i] = 8

            if 'exo' in heading and ('t' in heading.lower()
                                     or 'Temp' in caption):
                comp_gid_pred_dict['pred_row_label'][i] = 11
            elif 'endo' in heading and ('t' in heading.lower()
                                        or 'Temp' in caption):
                comp_gid_pred_dict['pred_row_label'][i] = 12

            power_regex = re.compile(r'10\s*[\*\^]?\*?\s*[-−–]\s*(\d+)')
            match = power_regex.search(heading)
            if match and pred_label in [13, 14, 17]:
                x      = int(match.group(1))
                values = [find_num(v) for v in num_val if find_num(v) is not None]
                numeric_values = [float(v) for v in values]
                median = np.median(numeric_values)
                if not math.isnan(median) and median > 0.1:
                    assert len(num_val) == nc - 1
                    for ind, val in enumerate(num_val):
                        new_val = find_num(
                            comp_data_dict[(pii, t_idx)]['act_table'][i][ind + 1])
                        if new_val is not None and modify_tables:
                            comp_data_dict[(pii, t_idx)]['act_table'][i][ind + 1] = \
                                str(new_val * 10 ** -x)

    pr_rl, pr_cl = (comp_gid_pred_dict['pred_row_label'],
                    comp_gid_pred_dict['pred_col_label'])
    comp_gid_pred = (comp_gid_pred_dict['pred_row_label']
                     + comp_gid_pred_dict['pred_col_label'])
    assert len(comp_gid_pred) == nr + nc
    table = comp_data_dict[(pii, t_idx)]['act_table']

    tuples = []

    if 2 in comp_gid_pred_alt:
        if row_ratio <= col_ratio:
            if 2 in comp_gid_pred_alt[nr:]:
                gid_index = None
                if 1 in comp_gid_pred[nr:]:
                    gid_index = comp_gid_pred[nr:].index(1)
                for prop_col_annotation in range(2, 21):
                    for j in range(nc):
                        if comp_gid_pred[nr:][j] == prop_col_annotation:
                            table_np  = np.array(table)
                            r_table   = list(table_np[:, j])
                            unit      = set_units(r_table, prop_col_annotation,
                                                  pii, t_idx)
                            prop_name = prop_names[prop_col_annotation - 2]
                            heading   = r_table[0]
                            heading   = heading.replace('−', '-').replace('–', '-').replace(' ', '').strip()  # Normalize heading
                            if unit == '':
                                unit = check_heading_for_unit(heading, prop_name)
                                if unit != '':
                                    unit = norm_unit(unit, prop_name)
                            check_unit = check_non_controv_unit(prop_name, unit)
                            if not check_unit:
                                new_u = check_heading_for_unit(heading, prop_name)
                                if new_u != '':
                                    unit = norm_unit(new_u, prop_name)
                            prop_flag = check_whether_in_limit(
                                t_name, r_table, prop_col_annotation)
                            if not prop_flag:
                                comp_gid_pred_dict['pred_col_label'][j] = 0
                                continue
                            for i in range(nr):
                                gid = (f'{pii}_{t_idx}_{i}_{j}'
                                       if gid_index is None
                                       else f'{pii}_{t_idx}_{i}_{j}'
                                            f'_{table[i][gid_index]}')
                                num = find_num(table[i][j])
                                if num is not None and num != 0:
                                    tuples.append((gid,
                                                   prop_names[prop_col_annotation - 2],
                                                   num, unit))
        else:
            if 2 in comp_gid_pred_alt[:nr]:
                gid_index = None
                if 1 in comp_gid_pred[:nr]:
                    gid_index = comp_gid_pred[:nr].index(1)
                for prop_row_annotation in range(2, 21):
                    for i in range(nr):
                        if comp_gid_pred[:nr][i] == prop_row_annotation:
                            table_np  = np.array(table)
                            r_table   = list(table_np[i, :])
                            unit      = set_units(r_table, prop_row_annotation,
                                                  pii, t_idx)
                            prop_name = prop_names[prop_row_annotation - 2]
                            heading   = r_table[0]
                            heading   = heading.replace('−', '-').replace('–', '-').replace(' ', '').strip()  # Normalize heading
                            if unit == '':
                                unit = check_heading_for_unit(heading, prop_name)
                                if unit != '':
                                    unit = norm_unit(unit, prop_name)
                            check_unit = check_non_controv_unit(prop_name, unit)
                            if not check_unit:
                                new_u = check_heading_for_unit(heading, prop_name)
                                if new_u != '':
                                    unit = norm_unit(new_u, prop_name)
                            prop_flag = check_whether_in_limit(
                                t_name, r_table, prop_row_annotation)
                            if not prop_flag:
                                comp_gid_pred_dict['pred_row_label'][i] = 0
                                continue
                            for j in range(nc):
                                gid = (f'{pii}_{t_idx}_{i}_{j}'
                                       if gid_index is None
                                       else f'{pii}_{t_idx}_{i}_{j}'
                                            f'_{table[gid_index][j]}')
                                num = find_num(table[i][j])
                                if num is not None and num != 0:
                                    tuples.append((gid,
                                                   prop_names[prop_row_annotation - 2],
                                                   num, unit))

    return tuples, comp_gid_pred_dict


# ── eval_model: collects raw softmax confidences (unchanged from original) ───

def eval_model(split, debug=False):
    model.eval()
    identifier      = []
    y_comp_true     = []
    y_comp_pred     = []
    ret_comp_pred   = []
    y_comp_raw_pred = []   # argmax before post-processing
    y_comp_conf     = []   # max softmax probability

    with torch.no_grad():
        tepoch = tqdm(loaders[split], unit='batch')
        for batch_data in tepoch:
            tepoch.set_description(f'{split} mode')
            comp_gid_logits, comp_gid_labels = model(batch_data)
            y_comp_true += comp_gid_labels.cpu().detach().tolist()

            comp_gid_probs          = F.softmax(comp_gid_logits, dim=1).cpu().detach()
            y_comp_raw_pred        += comp_gid_probs.argmax(1).tolist()
            comp_conf_list_batch    = comp_gid_probs.max(1).values.tolist()
            y_comp_conf            += comp_conf_list_batch

            pred_comp_gid_labels = comp_gid_logits.argmax(1).cpu().detach().tolist()
            del comp_gid_logits, comp_gid_labels

            base_comp_gid = 0
            total_len     = 0
            for x in batch_data:
                identifier.append((x['pii'], x['t_idx']))
                comp_dict = dict()
                comp_dict['pred_row_label'] = pred_comp_gid_labels[
                    base_comp_gid:base_comp_gid + x['num_rows']]
                comp_dict['raw_pred_row']   = list(comp_dict['pred_row_label'])
                comp_dict['conf_row']       = comp_conf_list_batch[
                    base_comp_gid:base_comp_gid + x['num_rows']]
                base_comp_gid += x['num_rows']
                comp_dict['pred_col_label'] = pred_comp_gid_labels[
                    base_comp_gid:base_comp_gid + x['num_cols']]
                comp_dict['raw_pred_col']   = list(comp_dict['pred_col_label'])
                comp_dict['conf_col']       = comp_conf_list_batch[
                    base_comp_gid:base_comp_gid + x['num_cols']]
                base_comp_gid += x['num_cols']
                total_len += x['num_rows'] + x['num_cols']
                comp_dict['act_table']   = x['act_table']
                comp_dict['prop_orient'] = x.get('prop_orient', None)
                ret_comp_pred.append(comp_dict)
            assert len(pred_comp_gid_labels) == total_len

    ret_tuples_pred, all_pred_tuples = [], []
    ret_tuples_true, all_true_tuples = [], []
    for (pii, t_idx), comp_gid_pred in zip(identifier, ret_comp_pred):
        pred_tuples, mod_gid_pred_dict = get_pred_tuples(
            pii, t_idx, comp_gid_pred, split)
        ret_tuples_pred.append(pred_tuples)
        all_pred_tuples += pred_tuples
        true_tuples = get_gold_tuples(pii, t_idx)
        ret_tuples_true.append(true_tuples)
        all_true_tuples += true_tuples
        y_comp_pred += (mod_gid_pred_dict['pred_row_label']
                        + mod_gid_pred_dict['pred_col_label'])

    row_col_pred_dict = {id_: item for id_, item in zip(identifier, ret_comp_pred)}

    all_pred_tuples_upd = [
        (a, b, eval_expr(c) if isinstance(c, str) else c, d)
        for a, b, c, d in all_pred_tuples]
    m_pred_tuples  = temp_cut_off(all_pred_tuples_upd)
    mod_pred_tuples = remove_tuples_on_units(m_pred_tuples)

    prop_gid_metrics = pd.DataFrame(
        classification_report(y_comp_true, y_comp_pred,
                              labels=list(range(21)), output_dict=True)
    ).round(3)
    print(f'{split} prop_gid_metrics =\n{prop_gid_metrics}\n')

    new_pred_tup = [tup for tup in mod_pred_tuples
                    if tup[1] != 'Dielectric constant']
    new_gold_tup = [tup for tup in all_true_tuples
                    if tup[1] != 'Dielectric constant']
    new_pred_tup = remove_tuples_on_units(new_pred_tup)

    for ind, tuples in enumerate(new_pred_tup):
        pii_parts = tuples[0].split('_')
        pii_str   = pii_parts[0]
        t_idx_str = int(pii_parts[1])
        prop_name = tuples[1]
        unit      = tuples[-1]
        check_unit = check_non_controv_unit(prop_name, unit)
        if not check_unit:
            new_unit = find_unit_in_paper(pii_str, t_idx_str, prop_name)
            new_unit = new_unit.strip()
            if new_unit != '':
                new_pred_tup[ind] = (*tuples[:-1],
                                     norm_unit(new_unit, prop_name))
            if new_unit == '' and prop_name == 'Fracture toughness':
                if ((tuples[2] > 0.3 and tuples[2] < 10)
                        or tuples[-1].lower().startswith('mp')):
                    new_pred_tup[ind] = (*tuples[:-1],
                                         norm_unit('MPam1/2', prop_name))

    ulti_gold_tuples = []
    for tupp in new_gold_tup:
        n_unit = norm_unit(tupp[-1], tupp[1])
        ulti_gold_tuples.append((tupp[0], tupp[1], tupp[2], n_unit))

    new_pred_tup     = remove_tuples_on_units(new_pred_tup)
    new_pred_tup     = final_checker_on_units(new_pred_tup)
    ulti_gold_tuples = remove_spaces_from_units(ulti_gold_tuples)
    new_pred_tup     = temp_cut_off(new_pred_tup)
    new_pred_tup     = remove_spaces_from_units(new_pred_tup)

    tuple_metrics      = get_tuples_metrics(ulti_gold_tuples, new_pred_tup, split)
    tuple_prop_metrics = get_property_metrics(
        ulti_gold_tuples, new_pred_tup, prop_names, split)
    print(f'{split} tuple metrics =\n{tuple_metrics}\n')
    print(f'{split} tuple_prop_metrics =')
    for key, value in tuple_prop_metrics.items():
        print(f'  {key} :: {value}')
    print()

    pickle.dump(new_pred_tup,
                open(os.path.join(os.getcwd(), split + '_pred_tuples_no_checker.pkl'), 'wb'))
    pickle.dump(ulti_gold_tuples,
                open(os.path.join(os.getcwd(), split + '_gold_tuples.pkl'), 'wb'))

    return (identifier, y_comp_pred, ret_comp_pred, ret_tuples_pred,
            new_pred_tup, ulti_gold_tuples,
            y_comp_true, y_comp_raw_pred, y_comp_conf)


# ── Load model and run inference on val and test ─────────────────────────────

model_args = {
    'hidden_layer_sizes': args.hidden_layer_sizes,
    'num_heads':          args.num_heads,
    'lm_name':            lm_name,
    'cache_dir':          cache_dir,
    'add_constraint':     args.add_constraint,
    'use_caption':        args.use_caption,
}
model = Model(model_args).to(device)

model_path = f"{args.model_variant}.bin"
_ckpt = torch.load(model_path, map_location=device)
_ckpt = {('gat_layers.1.res_fc.bias' if k == 'gat_layers.1.bias' else k): v
         for k, v in _ckpt.items()}
model.load_state_dict(_ckpt)
model = model.to(device)

res = {'val': dict(), 'test': dict()}

print('\nRunning model inference on validation and test splits...\n')
for split in ['val', 'test']:
    out = eval_model(split, debug=True)
    res[split]['identifier']      = out[0]
    res[split]['y_comp_pred']     = out[1]
    res[split]['ret_comp_pred']   = out[2]
    res[split]['ret_tuples_pred'] = out[3]
    res[split]['pred_tuples']     = out[4]
    res[split]['gold_tuples']     = out[5]
    res[split]['y_comp_true']     = out[6]
    res[split]['y_comp_raw_pred'] = out[7]
    res[split]['y_comp_conf']     = out[8]

# Convenience aliases — VALIDATION is used for all decisions
y_true_val     = np.array(res['val']['y_comp_true'])
y_raw_pred_val = np.array(res['val']['y_comp_raw_pred'])
y_conf_val     = np.array(res['val']['y_comp_conf'])

# Test is kept strictly for final held-out reporting
y_true_test     = np.array(res['test']['y_comp_true'])
y_raw_pred_test = np.array(res['test']['y_comp_raw_pred'])
y_conf_test     = np.array(res['test']['y_comp_conf'])

print(f'\nValidation split: {len(y_true_val):,} row/col nodes  '
      f'({(y_true_val != 0).sum():,} non-zero true labels)')
print(f'Test split:       {len(y_true_test):,} row/col nodes  '
      f'({(y_true_test != 0).sum():,} non-zero true labels)')


# ═══════════════════════════════════════════════════════════════════════════════
# Alpha sweep grid: percentages 0, 5, 10, ..., 95, 100  ->  decimals 0.00..1.00
# ═══════════════════════════════════════════════════════════════════════════════

SWEEP_ALPHAS = np.round(np.arange(0.0, 1.01, 0.05), 3)
# [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
#  0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00]
# 21 values total, matching "interval of 5" from 0 to 100

CANDIDATE_THRESHOLD = 0.03   # Recall - Precision must exceed this to be a candidate
F1_GAIN_THRESHOLD   = 0.03   # Bucketed alpha must gain more than this in F1 to be accepted


# ═══════════════════════════════════════════════════════════════════════════════
# Helper functions
# ═══════════════════════════════════════════════════════════════════════════════

def compute_single_prop_rc(y_true, y_pred, y_conf, cls_idx, alpha):
    """
    Compute precision, recall, F1 for a SINGLE property class (cls_idx),
    applying the confidence threshold alpha ONLY to predictions of that class.
    All other classes are kept unchanged.

    Filtering rule: a prediction p == cls_idx is kept if confidence >= alpha,
    otherwise suppressed to 0 (background).

    Returns
    -------
    precision, recall, f1 : float
    n_pred : int   number of predictions of cls_idx kept at this alpha
    n_true : int   number of true labels of cls_idx in the split
    """
    tp = fp = fn = 0
    for t, p, c in zip(y_true, y_pred, y_conf):
        # Apply threshold only to this property's predictions
        p_eff = (p if c >= alpha else 0) if p == cls_idx else p

        if p_eff == cls_idx:
            if p_eff == t: tp += 1
            else:          fp += 1
        if t == cls_idx and p_eff != t:
            fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1, tp + fp, tp + fn


def compute_overall_rc_alpha_dict(y_true, y_pred, y_conf, alpha_dict):
    """
    Apply per-property alpha thresholding simultaneously across all properties.
    alpha_dict : {cls_idx: alpha}  — classes absent from dict use alpha=0.0

    Returns
    -------
    overall_prec, overall_rec, overall_f1 : float
    per_prop : dict  {prop_name: {'precision','recall','f1','n_pred','n_true'}}
    """
    per_class_tp = [0] * 21
    per_class_fp = [0] * 21
    per_class_fn = [0] * 21

    for t, p, c in zip(y_true, y_pred, y_conf):
        if p == 0:
            p_eff = 0
        else:
            alpha = alpha_dict.get(p, 0.0)
            p_eff = p if c >= alpha else 0

        if p_eff != 0:
            if p_eff == t: per_class_tp[p_eff] += 1
            else:          per_class_fp[p_eff] += 1
        if t != 0 and p_eff != t:
            per_class_fn[t] += 1

    # Overall: exclude class 0 (background) and class 14 (Dielectric constant)
    active = [c for c in range(1, 21) if c != DIELECTRIC_CLS]
    tp_tot = sum(per_class_tp[c] for c in active)
    fp_tot = sum(per_class_fp[c] for c in active)
    fn_tot = sum(per_class_fn[c] for c in active)
    overall_prec = tp_tot / (tp_tot + fp_tot) if (tp_tot + fp_tot) > 0 else 1.0
    overall_rec  = tp_tot / (tp_tot + fn_tot) if (tp_tot + fn_tot) > 0 else 0.0
    overall_f1   = (2 * overall_prec * overall_rec
                    / (overall_prec + overall_rec)
                    if (overall_prec + overall_rec) > 0 else 0.0)

    per_prop = {}
    for name in prop_names:
        if name == 'Dielectric constant':
            continue
        cls = prop_names.index(name) + 2
        tp  = per_class_tp[cls]
        fp  = per_class_fp[cls]
        fn  = per_class_fn[cls]
        p   = tp / (tp + fp) if (tp + fp) > 0 else 1.0
        r   = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_prop[name] = {
            'precision': p, 'recall': r, 'f1': f1,
            'n_pred': tp + fp, 'n_true': tp + fn,
        }
    return overall_prec, overall_rec, overall_f1, per_prop


def bucket_alpha(alpha):
    """
    Map a per-property best alpha to a coarser representative bucket value.
    Input alpha is in [0.0, 1.0]; comparison uses percentage values.

    Bucket rules:
        [0,   20]%  ->  0.10     (includes alpha == 0.00)
        (20,  40]%  ->  0.30
        (40,  60]%  ->  0.50
        (60,  80]%  ->  0.70
        (80, 100]%  ->  0.90
    """
    pct = alpha * 100.0
    if   pct <= 20.0: return 0.10
    elif pct <= 40.0: return 0.30
    elif pct <= 60.0: return 0.50
    elif pct <= 80.0: return 0.70
    else:             return 0.90


def post_process_pred_tuples(all_pred_tuples, all_true_tuples):
    """
    Replicate the full post-processing pipeline from eval_model.
    Used for tuple-level evaluation after applying custom alpha thresholds.
    """
    all_pred_tuples_upd = [
        (a, b, eval_expr(c) if isinstance(c, str) else c, d)
        for a, b, c, d in all_pred_tuples]
    m_pred_tuples   = temp_cut_off(all_pred_tuples_upd)
    mod_pred_tuples = remove_tuples_on_units(m_pred_tuples)

    new_pred_tup = [t for t in mod_pred_tuples if t[1] != 'Dielectric constant']
    new_gold_tup = [t for t in all_true_tuples  if t[1] != 'Dielectric constant']
    new_pred_tup = remove_tuples_on_units(new_pred_tup)

    for ind, tuples in enumerate(new_pred_tup):
        pii_parts = tuples[0].split('_')
        pii_str   = pii_parts[0]
        t_idx_str = int(pii_parts[1])
        prop_name = tuples[1]
        unit      = tuples[-1]
        check_unit = check_non_controv_unit(prop_name, unit)
        if not check_unit:
            new_unit = find_unit_in_paper(pii_str, t_idx_str, prop_name)
            new_unit = new_unit.strip()
            if new_unit != '':
                new_pred_tup[ind] = (*tuples[:-1],
                                     norm_unit(new_unit, prop_name))
            if new_unit == '' and prop_name == 'Fracture toughness':
                if ((tuples[2] > 0.3 and tuples[2] < 10)
                        or tuples[-1].lower().startswith('mp')):
                    new_pred_tup[ind] = (*tuples[:-1],
                                         norm_unit('MPam1/2', prop_name))

    ulti_gold = []
    for tupp in new_gold_tup:
        ulti_gold.append((tupp[0], tupp[1], tupp[2],
                          norm_unit(tupp[-1], tupp[1])))

    new_pred_tup = remove_tuples_on_units(new_pred_tup)
    new_pred_tup = final_checker_on_units(new_pred_tup)
    ulti_gold    = remove_spaces_from_units(ulti_gold)
    new_pred_tup = temp_cut_off(new_pred_tup)
    new_pred_tup = remove_spaces_from_units(new_pred_tup)
    return new_pred_tup, ulti_gold


def compute_tuple_metrics_alpha_dict(alpha_dict, identifier, ret_comp_base,
                                     all_true_tuples, split='val'):
    """
    Apply per-property alpha_dict to raw GNN predictions, run the full
    tuple extraction + post-processing pipeline, return metrics.

    alpha_dict : {cls_idx: alpha}  — classes absent use alpha=0.0 (no filter)
    """
    all_pred_tuples = []
    for (pii, t_idx), base in zip(identifier, ret_comp_base):
        thresholded_row, thresholded_col = [], []
        for lst, out in [
            (zip(base['raw_pred_row'], base['conf_row']), thresholded_row),
            (zip(base['raw_pred_col'], base['conf_col']), thresholded_col),
        ]:
            for p, c in lst:
                alpha = alpha_dict.get(p, 0.0)
                out.append(p if c >= alpha else 0)

        pred_dict = {
            'pred_row_label': thresholded_row,
            'pred_col_label': thresholded_col,
            'act_table':      base['act_table'],
            'prop_orient':    base.get('prop_orient', None),
        }
        pred_tuples, _ = get_pred_tuples(
            pii, t_idx, pred_dict, split, modify_tables=False)
        all_pred_tuples += pred_tuples

    new_pred_tup, ulti_gold = post_process_pred_tuples(
        all_pred_tuples, all_true_tuples)
    overall  = get_tuples_metrics(ulti_gold, new_pred_tup, split)
    per_prop = get_property_metrics(ulti_gold, new_pred_tup, prop_names, split)
    return overall, per_prop


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: Baseline per-property metrics at alpha=0 (validation, row/col level)
# ═══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 80)
print('STEP 1 — Baseline Per-Property Metrics at alpha=0')
print('         Split: VALIDATION | Level: Row/Column Classification')
print('=' * 80)
print("""
  No confidence threshold is applied (alpha=0 means all predictions are kept).
  For each property class we compute Precision, Recall, and F1, along with the
  gap  (Recall - Precision).  A positive gap means the model is over-predicting
  that property (high recall, low precision), i.e., it produces false positives
  that confidence thresholding may be able to suppress.

  Candidate criterion: Recall - Precision > 3 pp  (gap > 0.03)
  Properties not meeting this criterion receive alpha=0 (no thresholding).
""")

baseline = {}   # name -> dict
for name in prop_names:
    if name == 'Dielectric constant':
        continue
    cls = prop_names.index(name) + 2
    p, r, f1, n_pred, n_true = compute_single_prop_rc(
        y_true_val, y_raw_pred_val, y_conf_val, cls, alpha=0.0)
    baseline[name] = {
        'cls': cls, 'precision': p, 'recall': r,
        'f1': f1, 'gap': r - p, 'n_pred': n_pred, 'n_true': n_true,
    }

print(f'  {"Property":<42}  {"Prec":>7}  {"Rec":>7}  {"F1":>7}  '
      f'{"R-P":>7}  {"n_pred":>7}  {"n_true":>7}  {"Candidate?":>11}')
print(f'  {"─"*42}  {"─"*7}  {"─"*7}  {"─"*7}  '
      f'{"─"*7}  {"─"*7}  {"─"*7}  {"─"*11}')

candidates = []
for name, v in baseline.items():
    is_cand = v['gap'] > CANDIDATE_THRESHOLD
    tag     = '  YES ✓' if is_cand else '  no'
    print(f'  {name:<42}  {v["precision"]:>7.4f}  {v["recall"]:>7.4f}  '
          f'{v["f1"]:>7.4f}  {v["gap"]:>+7.4f}  {v["n_pred"]:>7}  '
          f'{v["n_true"]:>7}  {tag}')
    if is_cand:
        candidates.append(name)

print(f'\n  {len(candidates)} candidate propert{"y" if len(candidates)==1 else "ies"} '
      f'(Recall − Precision > {CANDIDATE_THRESHOLD*100:.0f} pp):')
for nm in candidates:
    print(f'    • {nm:<42}  '
          f'R={baseline[nm]["recall"]:.4f}  '
          f'P={baseline[nm]["precision"]:.4f}  '
          f'gap={baseline[nm]["gap"]:+.4f}  '
          f'n_pred={baseline[nm]["n_pred"]}')

if not candidates:
    print('\n  No candidates found. Alpha=0 is optimal for all properties.')
    print('  Exiting.')
    sys.exit(0)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: Per-property alpha sweep on validation (row/col level, isolated)
# ═══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 80)
print('STEP 2 — Per-Property Alpha Sweep')
print('         Split: VALIDATION | Level: Row/Column Classification')
print('=' * 80)
print(f"""
  For each candidate property, alpha is swept independently over the grid:
      {list(np.round(SWEEP_ALPHAS, 2))}
  ({len(SWEEP_ALPHAS)} values from 0.00 to 1.00 in steps of 0.05)

  ISOLATION: Only that property's predictions are filtered at each alpha step.
  All other property predictions remain unchanged (alpha=0 for all others).
  This ensures each property's optimal alpha is identified without cross-property
  confounding effects.

  The best alpha is the value that maximises the property-specific validation F1.
""")

sweep_results        = {}   # name -> list of (alpha, prec, rec, f1, n_pred)
best_raw_alpha       = {}   # name -> best alpha (raw, before bucketing)

for name in candidates:
    cls         = baseline[name]['cls']
    f1_base     = baseline[name]['f1']
    p_base      = baseline[name]['precision']
    r_base      = baseline[name]['recall']
    n_true_base = baseline[name]['n_true']

    rows = []
    for alpha in SWEEP_ALPHAS:
        p, r, f1, n_pred, _ = compute_single_prop_rc(
            y_true_val, y_raw_pred_val, y_conf_val, cls, alpha)
        rows.append((alpha, p, r, f1, n_pred))
    sweep_results[name] = rows

    best_idx          = int(np.argmax([x[3] for x in rows]))
    best_raw_alpha[name] = rows[best_idx][0]

    print(f'\n  ── {name}  (class index {cls}) ──')
    print(f'  Baseline (alpha=0.00):  '
          f'P={p_base:.4f}  R={r_base:.4f}  F1={f1_base:.4f}  '
          f'n_pred={baseline[name]["n_pred"]}  n_true={n_true_base}')
    print(f'  {"alpha":>7}  {"Precision":>10}  {"Recall":>8}  '
          f'{"F1":>8}  {"n_pred":>7}  {"Note"}')
    print(f'  {"─"*7}  {"─"*10}  {"─"*8}  {"─"*8}  {"─"*7}')
    for alpha, p, r, f1, n_pred in rows:
        note = '  <- best alpha' if alpha == best_raw_alpha[name] else ''
        print(f'  {alpha:>7.2f}  {p:>10.4f}  {r:>8.4f}  '
              f'{f1:>8.4f}  {n_pred:>7}{note}')
    print(f'  => Best raw alpha = {best_raw_alpha[name]:.2f}  '
          f'(F1 = {rows[best_idx][3]:.4f}, '
          f'ΔF1 vs baseline = {rows[best_idx][3] - f1_base:+.4f})')


# ── Plot 1: Per-candidate alpha sweep curves (F1, Precision, Recall vs alpha) ─

_PROP_COLORS = [
    '#0072B2', '#E69F00', '#009E73', '#56B4E9', '#D55E00',
    '#CC79A7', '#4E9A06', '#8B4513', '#1A9E77', '#7570B3',
]

n_cand = len(candidates)
ncols  = min(n_cand, 3)
nrows  = math.ceil(n_cand / ncols)
fig, axes = plt.subplots(nrows, ncols,
                          figsize=(3.504 * ncols, 2.8 * nrows),
                          squeeze=False)

for idx, name in enumerate(candidates):
    ax    = axes[idx // ncols][idx % ncols]
    rows  = sweep_results[name]
    alphas_plot = [x[0] for x in rows]
    f1s_plot    = [x[3] for x in rows]
    precs_plot  = [x[1] for x in rows]
    recs_plot   = [x[2] for x in rows]
    best_a      = best_raw_alpha[name]
    best_f1_v   = rows[int(np.argmax(f1s_plot))][3]

    ax.plot(alphas_plot, f1s_plot,    '-o', color=C['blue'],
            ms=2.5, lw=1.0, label='F1')
    ax.plot(alphas_plot, precs_plot,  '-s', color=C['green'],
            ms=2.5, lw=1.0, label='Precision')
    ax.plot(alphas_plot, recs_plot,   '-^', color=C['vermillion'],
            ms=2.5, lw=1.0, label='Recall')
    ax.axvline(best_a, color=C['gray'], ls='--', lw=0.8,
               label=f'Best α={best_a:.2f}')
    ax.set_title(name, fontsize=6)
    ax.set_xlabel('alpha')
    ax.set_ylabel('Score')
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(-0.02, 1.10)
    ax.grid(True)
    if idx == 0:
        ax.legend(loc='lower left', fontsize=5)

# Hide unused subplots
for idx in range(n_cand, nrows * ncols):
    axes[idx // ncols][idx % ncols].set_visible(False)

fig.suptitle('Step 2: Per-Property Alpha Sweep (Validation Split)',
             fontsize=7, fontweight='bold')
fig.tight_layout(pad=0.5)
fig.savefig(os.path.join(args.out_dir, '1_task5_alpha_sweep_per_property.png'))
print(f'\nSaved: 1_task5_alpha_sweep_per_property.png')
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: Bucketing — map best raw alpha to coarser representative value
# ═══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 80)
print('STEP 3 — Alpha Bucketing')
print('=' * 80)
print("""
  The best raw alpha for each candidate is mapped to a coarser bucket to
  reduce overfitting to the validation split.  The bucket boundaries and
  representative values are (percentages):

      [0,   20]%   ->  0.10       (includes alpha == 0.00)
      (20,  40]%   ->  0.30
      (40,  60]%   ->  0.50
      (60,  80]%   ->  0.70
      (80, 100]%   ->  0.90

  This prevents the threshold from being tuned too precisely to validation
  noise, particularly important for properties with small n_pred.
""")

bucketed = {}   # name -> bucketed alpha

print(f'  {"Property":<42}  {"Best α (raw)":>12}  {"Pct":>7}  '
      f'{"Bucket range":>14}  {"Bucketed α":>10}')
print(f'  {"─"*42}  {"─"*12}  {"─"*7}  {"─"*14}  {"─"*10}')

for name in candidates:
    raw  = best_raw_alpha[name]
    bkt  = bucket_alpha(raw)
    pct  = raw * 100.0
    bucketed[name] = bkt

    if pct <= 20.0:   brng = '[0, 20]%'
    elif pct <= 40.0: brng = '(20, 40]%'
    elif pct <= 60.0: brng = '(40, 60]%'
    elif pct <= 80.0: brng = '(60, 80]%'
    else:             brng = '(80, 100]%'

    print(f'  {name:<42}  {raw:>12.2f}  {pct:>7.1f}  '
          f'{brng:>14}  {bkt:>10.2f}')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: F1 gain check — accept bucketed alpha only if gain > 3 pp
# ═══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 80)
print('STEP 4 — F1 Gain Check')
print('         Split: VALIDATION | Level: Row/Column Classification')
print('=' * 80)
print(f"""
  The bucketed alpha is now applied to each candidate property (in isolation,
  same as in the sweep) and the resulting validation F1 is compared to the
  baseline F1 at alpha=0.

  ACCEPTANCE CRITERION:
      F1(bucketed alpha) - F1(alpha=0)  >  {F1_GAIN_THRESHOLD:.2f}  (> {F1_GAIN_THRESHOLD*100:.0f} percentage points)

  Properties that do not meet this criterion are discarded (alpha reverts to 0).
  This final gate prevents accepting thresholds whose F1 gain is too small to
  be considered meaningful — a common outcome when:
    (a) the bucketing mapped a useful raw alpha to a stricter bucket, or
    (b) the validation n_pred is so small that the sweep overfit.
""")

final_alpha_dict = {}    # {cls_idx: alpha}  only accepted properties
accepted         = []    # list of tuples for reporting
discarded        = []

print(f'  {"Property":<42}  {"cls":>4}  {"α=0.00 F1":>10}  '
      f'{"Bkt α":>6}  {"Bkt F1":>8}  {"ΔF1":>8}  {"Decision":>12}')
print(f'  {"─"*42}  {"─"*4}  {"─"*10}  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*12}')

for name in candidates:
    cls    = baseline[name]['cls']
    f1_0   = baseline[name]['f1']
    bkt    = bucketed[name]

    p_b, r_b, f1_b, n_pred_b, _ = compute_single_prop_rc(
        y_true_val, y_raw_pred_val, y_conf_val, cls, bkt)
    delta = f1_b - f1_0

    if delta > F1_GAIN_THRESHOLD:
        decision = 'ACCEPT ✓'
        final_alpha_dict[cls] = bkt
        accepted.append({
            'name': name, 'cls': cls, 'alpha': bkt,
            'f1_0': f1_0, 'f1_alpha': f1_b, 'delta': delta,
            'prec_0': baseline[name]['precision'],
            'prec_alpha': p_b,
            'rec_0': baseline[name]['recall'],
            'rec_alpha': r_b,
            'n_pred': n_pred_b,
        })
    else:
        decision = 'DISCARD ✗'
        discarded.append({
            'name': name, 'cls': cls, 'candidate_alpha': bkt,
            'f1_0': f1_0, 'f1_alpha': f1_b, 'delta': delta,
        })

    print(f'  {name:<42}  {cls:>4}  {f1_0:>10.4f}  '
          f'{bkt:>6.2f}  {f1_b:>8.4f}  {delta:>+8.4f}  {decision:>12}')

print(f'\n  ── Summary ─────────────────────────────────────────────────────────────')
print(f'  ACCEPTED  ({len(accepted)} propert{"y" if len(accepted)==1 else "ies"}):')
for v in accepted:
    print(f'    cls {v["cls"]:>2}  {v["name"]:<42}  alpha = {v["alpha"]:.2f}  '
          f'F1: {v["f1_0"]:.4f} -> {v["f1_alpha"]:.4f}  '
          f'(Δ = {v["delta"]:+.4f})')

print(f'\n  DISCARDED ({len(discarded)} propert{"y" if len(discarded)==1 else "ies"}) '
      f'— reverted to alpha=0:')
for v in discarded:
    print(f'    cls {v["cls"]:>2}  {v["name"]:<42}  '
          f'candidate alpha={v["candidate_alpha"]:.2f}  '
          f'F1: {v["f1_0"]:.4f} -> {v["f1_alpha"]:.4f}  '
          f'(Δ = {v["delta"]:+.4f}, below threshold)')

print(f'\n  FINAL PER-PROPERTY ALPHA ASSIGNMENTS')
print(f'  {"Property":<42}  {"Class":>6}  {"Alpha":>6}')
print(f'  {"─"*42}  {"─"*6}  {"─"*6}')
for name in prop_names:
    if name == 'Dielectric constant':
        continue
    cls   = prop_names.index(name) + 2
    alpha = final_alpha_dict.get(cls, 0.0)
    tag   = '' if alpha == 0.0 else '  <- thresholded'
    print(f'  {name:<42}  {cls:>6}  {alpha:>6.2f}{tag}')


# ── Plot 2: Before/after F1 comparison for accepted properties ────────────────

if accepted:
    names_acc  = [v['name']       for v in accepted]
    f1_before  = [v['f1_0']       for v in accepted]
    f1_after   = [v['f1_alpha']   for v in accepted]
    prec_before = [v['prec_0']    for v in accepted]
    prec_after  = [v['prec_alpha'] for v in accepted]
    deltas     = [v['delta']      for v in accepted]

    x  = np.arange(len(names_acc))
    bw = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(7.205, 3.3))

    for panel_idx, (ax, b_vals, a_vals, ylabel) in enumerate([
        (axes[0], f1_before,   f1_after,   'F1 Score'),
        (axes[1], prec_before, prec_after, 'Precision'),
    ]):
        ax.bar(x - bw / 2, b_vals, width=bw,
               color=C['lightgray'], edgecolor='#888888',
               linewidth=0.4, zorder=2, label='Before (α=0)')
        ax.bar(x + bw / 2, a_vals, width=bw,
               color=C['blue'], alpha=0.85,
               edgecolor='none', zorder=2, label='After (final α)')
        for i, (b, a) in enumerate(zip(b_vals, a_vals)):
            delta  = a - b
            ypos   = max(b, a) + 0.012
            colour = C['green'] if delta > 0 else C['vermillion']
            ax.text(x[i] + bw / 2, ypos, f'{delta:+.3f}',
                    ha='center', va='bottom',
                    fontsize=4.5, color=colour, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names_acc, rotation=40, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1.22)
        ax.grid(True, axis='y')
        ax.legend(loc='lower right')
        ax.text(-0.10, 1.04, 'ab'[panel_idx],
                transform=ax.transAxes,
                fontsize=8, fontweight='bold', va='top')

    fig.suptitle('Step 4: Accepted Thresholds — Before vs After (Validation)',
                 fontsize=7, fontweight='bold')
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(args.out_dir, '2_task5_before_after_accepted.png'))
    print(f'\nSaved: 2_task5_before_after_accepted.png')
    plt.close(fig)
else:
    print('\n  No properties accepted — skipping before/after plot.')


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: Final Scores
# ═══════════════════════════════════════════════════════════════════════════════

print()
print('=' * 80)
print('STEP 5 — Final Extraction Scores')
print('=' * 80)
print(f"""
  All accepted per-property alpha thresholds are now applied SIMULTANEOUSLY.
  (The combined effect may differ slightly from individual sweep results since
   all thresholds interact within the same pipeline pass.)

  Metrics reported:
    (a) Row/column classification  — direct output of the GNN node classifier
    (b) Tuple extraction           — end-to-end pipeline with post-processing

  VALIDATION split — used for all threshold decisions (reference only).
  TEST split       — strictly held-out; NEVER consulted during Steps 1–4.
                     These results reflect true generalisation performance.
""")

# ─── (a) Row/column level ─────────────────────────────────────────────────────

print('── (a) Row/Column Classification Metrics ────────────────────────────────')

# Baseline alpha=0
prec_0v, rec_0v, f1_0v, pp_0v = compute_overall_rc_alpha_dict(
    y_true_val, y_raw_pred_val, y_conf_val, alpha_dict={})
prec_0t, rec_0t, f1_0t, pp_0t = compute_overall_rc_alpha_dict(
    y_true_test, y_raw_pred_test, y_conf_test, alpha_dict={})

# Final alpha dict
prec_fv, rec_fv, f1_fv, pp_fv = compute_overall_rc_alpha_dict(
    y_true_val, y_raw_pred_val, y_conf_val, final_alpha_dict)
prec_ft, rec_ft, f1_ft, pp_ft = compute_overall_rc_alpha_dict(
    y_true_test, y_raw_pred_test, y_conf_test, final_alpha_dict)

print(f'\n  {"Split":<5}  {"Setting":<22}  {"Precision":>10}  '
      f'{"Recall":>8}  {"F1":>8}  {"ΔF1 vs baseline":>16}')
print(f'  {"─"*5}  {"─"*22}  {"─"*10}  {"─"*8}  {"─"*8}  {"─"*16}')
print(f'  {"val":<5}  {"alpha=0 (baseline)":<22}  '
      f'{prec_0v:>10.4f}  {rec_0v:>8.4f}  {f1_0v:>8.4f}  {"—":>16}')
print(f'  {"val":<5}  {"final alpha (data-drv)":<22}  '
      f'{prec_fv:>10.4f}  {rec_fv:>8.4f}  {f1_fv:>8.4f}  '
      f'{f1_fv - f1_0v:>+16.4f}')
print(f'  {"test":<5}  {"alpha=0 (baseline)":<22}  '
      f'{prec_0t:>10.4f}  {rec_0t:>8.4f}  {f1_0t:>8.4f}  {"—":>16}')
print(f'  {"test":<5}  {"final alpha (data-drv)":<22}  '
      f'{prec_ft:>10.4f}  {rec_ft:>8.4f}  {f1_ft:>8.4f}  '
      f'{f1_ft - f1_0t:>+16.4f}')

print(f'\n  Per-Property Row/Column Metrics — VALIDATION split')
print(f'  {"Property":<42}  {"alpha":>6}  '
      f'{"F1@0":>8}  {"F1@α":>8}  {"ΔF1":>8}  '
      f'{"P@0":>7}  {"P@α":>7}  {"R@0":>7}  {"R@α":>7}  '
      f'{"n_pred@0":>9}  {"n_pred@α":>9}')
print(f'  {"─"*42}  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*8}  '
      f'{"─"*7}  {"─"*7}  {"─"*7}  {"─"*7}  {"─"*9}  {"─"*9}')

for name in prop_names:
    if name == 'Dielectric constant':
        continue
    cls   = prop_names.index(name) + 2
    alpha = final_alpha_dict.get(cls, 0.0)
    v0    = pp_0v[name]
    vf    = pp_fv[name]
    print(f'  {name:<42}  {alpha:>6.2f}  '
          f'{v0["f1"]:>8.4f}  {vf["f1"]:>8.4f}  '
          f'{vf["f1"] - v0["f1"]:>+8.4f}  '
          f'{v0["precision"]:>7.4f}  {vf["precision"]:>7.4f}  '
          f'{v0["recall"]:>7.4f}  {vf["recall"]:>7.4f}  '
          f'{v0["n_pred"]:>9}  {vf["n_pred"]:>9}')

print(f'\n  Per-Property Row/Column Metrics — TEST split (held-out)')
print(f'  {"Property":<42}  {"alpha":>6}  '
      f'{"F1@0":>8}  {"F1@α":>8}  {"ΔF1":>8}  '
      f'{"P@0":>7}  {"P@α":>7}  {"R@0":>7}  {"R@α":>7}')
print(f'  {"─"*42}  {"─"*6}  {"─"*8}  {"─"*8}  {"─"*8}  '
      f'{"─"*7}  {"─"*7}  {"─"*7}  {"─"*7}')

for name in prop_names:
    if name == 'Dielectric constant':
        continue
    cls   = prop_names.index(name) + 2
    alpha = final_alpha_dict.get(cls, 0.0)
    v0    = pp_0t[name]
    vf    = pp_ft[name]
    print(f'  {name:<42}  {alpha:>6.2f}  '
          f'{v0["f1"]:>8.4f}  {vf["f1"]:>8.4f}  '
          f'{vf["f1"] - v0["f1"]:>+8.4f}  '
          f'{v0["precision"]:>7.4f}  {vf["precision"]:>7.4f}  '
          f'{v0["recall"]:>7.4f}  {vf["recall"]:>7.4f}')


# ─── (b) Tuple extraction level ───────────────────────────────────────────────

print()
print('── (b) Tuple Extraction Metrics ─────────────────────────────────────────')
print('  Running full extraction pipeline with final alpha dict...')
print('  (This may take several minutes.)')

# Baseline tuple metrics: reuse tuples already computed by eval_model
tup_0v = get_tuples_metrics(
    res['val']['gold_tuples'], res['val']['pred_tuples'], 'val')
tup_0t = get_tuples_metrics(
    res['test']['gold_tuples'], res['test']['pred_tuples'], 'test')
tup_0v_pp = get_property_metrics(
    res['val']['gold_tuples'], res['val']['pred_tuples'], prop_names, 'val')
tup_0t_pp = get_property_metrics(
    res['test']['gold_tuples'], res['test']['pred_tuples'], prop_names, 'test')

# Final alpha tuple metrics
tup_fv, tup_fv_pp = compute_tuple_metrics_alpha_dict(
    final_alpha_dict,
    res['val']['identifier'],  res['val']['ret_comp_pred'],
    res['val']['gold_tuples'], split='val')

tup_ft, tup_ft_pp = compute_tuple_metrics_alpha_dict(
    final_alpha_dict,
    res['test']['identifier'], res['test']['ret_comp_pred'],
    res['test']['gold_tuples'], split='test')

# Overall summary
print(f'\n  {"Split":<5}  {"Setting":<22}  {"Precision":>10}  '
      f'{"Recall":>8}  {"F1":>8}  {"Support":>8}  {"ΔF1 vs baseline":>16}')
print(f'  {"─"*5}  {"─"*22}  {"─"*10}  {"─"*8}  {"─"*8}  '
      f'{"─"*8}  {"─"*16}')
print(f'  {"val":<5}  {"alpha=0 (baseline)":<22}  '
      f'{tup_0v["precision"]:>10.4f}  {tup_0v["recall"]:>8.4f}  '
      f'{tup_0v["fscore"]:>8.4f}  {tup_0v["support"]:>8}  {"—":>16}')
print(f'  {"val":<5}  {"final alpha (data-drv)":<22}  '
      f'{tup_fv["precision"]:>10.4f}  {tup_fv["recall"]:>8.4f}  '
      f'{tup_fv["fscore"]:>8.4f}  {tup_fv["support"]:>8}  '
      f'{tup_fv["fscore"] - tup_0v["fscore"]:>+16.4f}')
print(f'  {"test":<5}  {"alpha=0 (baseline)":<22}  '
      f'{tup_0t["precision"]:>10.4f}  {tup_0t["recall"]:>8.4f}  '
      f'{tup_0t["fscore"]:>8.4f}  {tup_0t["support"]:>8}  {"—":>16}')
print(f'  {"test":<5}  {"final alpha (data-drv)":<22}  '
      f'{tup_ft["precision"]:>10.4f}  {tup_ft["recall"]:>8.4f}  '
      f'{tup_ft["fscore"]:>8.4f}  {tup_ft["support"]:>8}  '
      f'{tup_ft["fscore"] - tup_0t["fscore"]:>+16.4f}')

# Per-property tuple metrics (val)
print(f'\n  Per-Property Tuple Metrics — VALIDATION split')
print(f'  {"Property":<42}  {"alpha":>6}  '
      f'{"Tup F1@0":>9}  {"Tup F1@α":>9}  {"ΔTup F1":>8}  '
      f'{"Prec@0":>8}  {"Prec@α":>8}  {"Rec@0":>7}  {"Rec@α":>7}')
print(f'  {"─"*42}  {"─"*6}  {"─"*9}  {"─"*9}  {"─"*8}  '
      f'{"─"*8}  {"─"*8}  {"─"*7}  {"─"*7}')

for name in prop_names:
    if name == 'Dielectric constant':
        continue
    cls   = prop_names.index(name) + 2
    alpha = final_alpha_dict.get(cls, 0.0)
    m0    = tup_0v_pp.get(name, {'fscore': 0.0, 'precision': 0.0,
                                   'recall': 0.0, 'support': 0})
    mf    = tup_fv_pp.get(name, {'fscore': 0.0, 'precision': 0.0,
                                   'recall': 0.0, 'support': 0})
    print(f'  {name:<42}  {alpha:>6.2f}  '
          f'{m0["fscore"]:>9.4f}  {mf["fscore"]:>9.4f}  '
          f'{mf["fscore"] - m0["fscore"]:>+8.4f}  '
          f'{m0["precision"]:>8.4f}  {mf["precision"]:>8.4f}  '
          f'{m0["recall"]:>7.4f}  {mf["recall"]:>7.4f}')

# Per-property tuple metrics (test — held-out)
print(f'\n  Per-Property Tuple Metrics — TEST split (HELD-OUT)')
print(f'  NOTE: Alpha values were fixed on the validation split.')
print(f'        These test results were NOT used to select any threshold.')
print(f'  {"Property":<42}  {"alpha":>6}  '
      f'{"Tup Prec":>9}  {"Tup Rec":>8}  {"Tup F1":>7}  {"Support":>8}')
print(f'  {"─"*42}  {"─"*6}  {"─"*9}  {"─"*8}  {"─"*7}  {"─"*8}')

for name in prop_names:
    if name == 'Dielectric constant':
        continue
    cls   = prop_names.index(name) + 2
    alpha = final_alpha_dict.get(cls, 0.0)
    mf    = tup_ft_pp.get(name, {'fscore': 0.0, 'precision': 0.0,
                                   'recall': 0.0, 'support': 0})
    print(f'  {name:<42}  {alpha:>6.2f}  '
          f'{mf["precision"]:>9.4f}  {mf["recall"]:>8.4f}  '
          f'{mf["fscore"]:>7.4f}  {mf.get("support", 0):>8}')


# ── Plot 3: Full property F1 impact (val + test, before/after) ────────────────

if accepted:
    # Show all 18 properties; thresholded ones highlighted
    all_names = [n for n in prop_names if n != 'Dielectric constant']
    x = np.arange(len(all_names))
    bw = 0.22

    fig, axes = plt.subplots(2, 2, figsize=(7.205, 6.0))
    panels = [
        (axes[0][0], pp_0v, pp_fv,   'Row/Column F1 (Validation)'),
        (axes[0][1], pp_0t, pp_ft,   'Row/Column F1 (Test — held-out)'),
        (axes[1][0], tup_0v_pp, tup_fv_pp, 'Tuple F1 (Validation)'),
        (axes[1][1], tup_0t_pp, tup_ft_pp, 'Tuple F1 (Test — held-out)'),
    ]

    accepted_names = {v['name'] for v in accepted}

    for ax, pp_before, pp_after, title in panels:
        b_vals, a_vals, bar_colours = [], [], []
        for nm in all_names:
            b_f1 = (pp_before[nm]['f1']
                    if 'f1' in pp_before.get(nm, {})
                    else pp_before.get(nm, {}).get('fscore', 0.0))
            a_f1 = (pp_after[nm]['f1']
                    if 'f1' in pp_after.get(nm, {})
                    else pp_after.get(nm, {}).get('fscore', 0.0))
            b_vals.append(b_f1)
            a_vals.append(a_f1)
            bar_colours.append(C['vermillion'] if nm in accepted_names
                                else C['blue'])

        ax.bar(x - bw / 2, b_vals, width=bw, color=C['lightgray'],
               edgecolor='#888888', linewidth=0.4, zorder=2)
        ax.bar(x + bw / 2, a_vals, width=bw, color=bar_colours,
               alpha=0.85, edgecolor='none', zorder=2)

        ax.set_xticks(x)
        ax.set_xticklabels(all_names, rotation=45, ha='right', fontsize=4.5)
        ax.set_ylabel('F1')
        ax.set_ylim(0, 1.15)
        ax.set_title(title, fontsize=6)
        ax.grid(True, axis='y')
        ax.legend(handles=[
            Patch(facecolor=C['lightgray'], edgecolor='#888888',
                  lw=0.4, label='Before (α=0)'),
            Patch(facecolor=C['blue'],      edgecolor='none',
                  label='After (α=0, unchanged)'),
            Patch(facecolor=C['vermillion'], edgecolor='none',
                  label='After (accepted threshold)'),
        ], loc='lower right', fontsize=4.5)

    fig.suptitle('Step 5: Full Per-Property F1 — Before vs After (All Properties)',
                 fontsize=7, fontweight='bold')
    fig.tight_layout(pad=0.5)
    fig.savefig(os.path.join(args.out_dir, '3_task5_full_property_impact.png'))
    print(f'\nSaved: 3_task5_full_property_impact.png')
    plt.close(fig)


# ── Save full results ─────────────────────────────────────────────────────────

task5_results = {
    # Alpha decisions (derived from validation only)
    'final_alpha_dict':         final_alpha_dict,
    'accepted_props': [
        {'name': v['name'], 'cls': v['cls'], 'alpha': v['alpha'],
         'f1_0_val': v['f1_0'], 'f1_alpha_val': v['f1_alpha'],
         'delta_val': v['delta']}
        for v in accepted
    ],
    'discarded_props': [
        {'name': v['name'], 'cls': v['cls'],
         'candidate_alpha': v['candidate_alpha'],
         'f1_0_val': v['f1_0'], 'f1_alpha_val': v['f1_alpha'],
         'delta_val': v['delta']}
        for v in discarded
    ],
    # Baseline (alpha=0) row/col
    'val_rc_baseline':          (prec_0v, rec_0v, f1_0v),
    'test_rc_baseline':         (prec_0t, rec_0t, f1_0t),
    # Final (accepted alphas) row/col
    'val_rc_final':             (prec_fv, rec_fv, f1_fv),
    'test_rc_final':            (prec_ft, rec_ft, f1_ft),
    # Baseline (alpha=0) tuple
    'val_tup_baseline':         tup_0v,
    'test_tup_baseline':        tup_0t,
    # Final tuple
    'val_tup_final':            tup_fv,
    'test_tup_final':           tup_ft,
    # Per-property tuple breakdown
    'val_tup_per_prop_final':   tup_fv_pp,
    'test_tup_per_prop_final':  tup_ft_pp,
    # Raw node-level data (for calibration analysis)
    'y_true_val':               list(y_true_val),
    'y_raw_pred_val':           list(y_raw_pred_val),
    'y_conf_val':               list(y_conf_val),
    'y_true_test':              list(y_true_test),
    'y_raw_pred_test':          list(y_raw_pred_test),
    'y_conf_test':              list(y_conf_test),
}

pkl_path = os.path.join(args.out_dir, 'task5_results.pkl')
pickle.dump(task5_results, open(pkl_path, 'wb'))
print(f'\nSaved: task5_results.pkl')
print(f'\nAll outputs in: {args.out_dir}')
print()
print('=' * 80)
print('TASK 5 COMPLETE')
print('=' * 80)
print(f'  Properties thresholded : {len(accepted)}')
for v in accepted:
    print(f'    {v["name"]:<42}  alpha = {v["alpha"]:.2f}')
print(f'  Properties at alpha=0  : {len([n for n in prop_names if n != "Dielectric constant"]) - len(accepted)}')
print(f'  Val  RC  F1:  {f1_0v:.4f} -> {f1_fv:.4f}  ({f1_fv-f1_0v:+.4f})')
print(f'  Test RC  F1:  {f1_0t:.4f} -> {f1_ft:.4f}  ({f1_ft-f1_0t:+.4f})')
print(f'  Val  Tup F1:  {tup_0v["fscore"]:.4f} -> {tup_fv["fscore"]:.4f}  '
      f'({tup_fv["fscore"]-tup_0v["fscore"]:+.4f})')
print(f'  Test Tup F1:  {tup_0t["fscore"]:.4f} -> {tup_ft["fscore"]:.4f}  '
      f'({tup_ft["fscore"]-tup_0t["fscore"]:+.4f})')
