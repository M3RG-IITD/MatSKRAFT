from argparse import ArgumentParser
import math
import multiprocessing as mp
import os
import pickle
import sys
sys.path.append('..')

from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed, get_linear_schedule_with_warmup
from matplotlib import pyplot as plt

from gnn_model import GNN_Model as Model
from utils import *
from units import *
from post_processing import *
from post_processing_2 import *
import re
import math

os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
pd.set_option('display.max_columns', None)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('using device:', device)


parser = ArgumentParser()
parser.add_argument('--seed', required=True, type=int)
parser.add_argument('--hidden_layer_sizes', nargs='+', required=True, type=int)
parser.add_argument('--num_heads', nargs='+', required=True, type=int)
parser.add_argument('--num_epochs', required=False, default=15, type=int)
parser.add_argument('--lr', required=False, default=1e-3, type=float)
parser.add_argument('--lm_lr', required=False, default=1e-5, type=float)
# parser.add_argument('--use_regex_feat', action='store_true')
# parser.add_argument('--use_max_freq_feat', action='store_true')
parser.add_argument('--add_constraint', action='store_true')
# parser.add_argument('--regex_emb_size', required=False, default=256, type=int)
# parser.add_argument('--max_freq_emb_size', required=False, default=256, type=int)
parser.add_argument('--c_loss_lambda', required=False, default=50.0, type=float)
# parser.add_argument('--row_col_loss_lambda', required=False, default=1.0, type=float)
parser.add_argument('--use_caption', action='store_true')
parser.add_argument('--model_save_file', required=True, type=str)
parser.add_argument('--res_file', required=False, type=str)
args = parser.parse_args()
print(args)

lm_name = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')
os.makedirs(os.path.dirname(os.path.abspath(args.model_save_file)), exist_ok=True)


# torch.use_deterministic_algorithms(True)
torch.set_deterministic(True)
torch.backends.cudnn.benchmark = False

datasets = dict()
for split in splits:
    datasets[split] = TableDataset([comp_data_dict[pii_t_idx] for pii_t_idx in train_val_test_split[split]])

set_seed(args.seed)
batch_size = 4
num_workers = mp.cpu_count()
loaders = dict()
for split in splits:
    loaders[split] = DataLoader(datasets[split], batch_size=batch_size, shuffle=(split == 'train'), \
    num_workers=num_workers, collate_fn=lambda x: x)

# all_train_regex_labels = [x['regex_table'] for x in datasets['train']]
all_train_row_col_labels = []
for x in datasets['train']:
    all_train_row_col_labels += x['prop_row_label'] + x['prop_col_label']

num_epochs = args.num_epochs
n_batches = math.ceil(len(datasets['train']) / batch_size)
n_steps = n_batches * num_epochs
warmup_steps = n_steps // 10

model_args = {
    'hidden_layer_sizes': args.hidden_layer_sizes,
    'num_heads': args.num_heads,
    'lm_name': lm_name,
    'cache_dir': cache_dir,
    'add_constraint': args.add_constraint,
    'use_caption': args.use_caption,
}
model = Model(model_args).to(device)

optim_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if 'encoder' not in n], 'lr': args.lr},
    {'params': [p for n, p in model.named_parameters() if 'encoder' in n], 'lr': args.lm_lr},
]
optim = torch.optim.AdamW(optim_grouped_parameters)


row_col_class_weights = torch.Tensor(compute_class_weight('balanced', classes=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], y=all_train_row_col_labels)).to(device) # 0 - irrelevant, 1 - id
row_col_loss_fn = nn.CrossEntropyLoss(weight=row_col_class_weights)

scheduler = get_linear_schedule_with_warmup(optim, num_warmup_steps=warmup_steps, num_training_steps=n_steps)

#prop_ids = [510, 1140, 2010, 2051, 540, 50, 180, 60, 160, 1014, 1015, 3012, 3174, 1116, 1119, 1020, 1011, 1118, 70, 1306]
prop_names = ['Density', 'Glass transition temperature', 'Refractive index', 'Abbe value', "Young's modulus", 'Shear modulus', 'Vickers hardness', 'Poisson ratio', 'Fracture toughness', 'Crystallization temp', 'Melting temp', 'Electric conductivity', 'Dielectric constant', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Thermal expansion coefficient', 'Liquidus temperature', 'Bulk modulus', 'Activation energy']

# gold_tuples = dict()
# ret_tuples_gold = dict()
# gold_tuples = dict()
# for split in splits:
#     gold_tuples[split] = []
#     ret_tuples_gold[split] = []
#     for pii, t_idx in train_val_test_split[split]:
#         gold_tuples[split] += get_gold_tuples(pii, t_idx)
#         ret_tuples_gold[split].append(get_gold_tuples(pii, t_idx))

# TODO: define get pred unit
# def get_pred_unit

# get_pred_tuples(pii, t_idx, comp_gid_pred, split)

def clean_names(name):
    match = re.search(r'(\[.*\]|\(.*\)|\{.*\}|\<.*\>)([^a-zA-Z]*)', name)
    if match:
        unit_name = match.group(1)[1:-1]
        cleaned_name = re.sub(r'(\[.*\]|\(.*\)|\{.*\}|\<.*\>)([^a-zA-Z]*)', '', name).strip()
    else:
        cleaned_name = name
        unit_name = None  # If no match, you can choose how to handle this case
    return cleaned_name, unit_name


def get_pred_tuples(pii, t_idx, comp_gid_pred_dict, split):
    # gid_list = []
    t_name = pii + '_' + str(t_idx)
    c = comp_data_dict[(pii, t_idx)]
    caption = c['caption']
    table = c['act_table']
    nr, nc = c['num_rows'], c['num_cols']
    pr_rl, pr_cl = comp_gid_pred_dict['pred_row_label'], comp_gid_pred_dict['pred_col_label']
    comp_gid_pred = comp_gid_pred_dict['pred_row_label'] + comp_gid_pred_dict['pred_col_label']
    # print(comp_gid_pred)
    assert len(comp_gid_pred) == nr+nc
    
    comp_gid_pred_alt = [2 if i in range(2,21) else i for i in comp_gid_pred]
    row_ratio = comp_gid_pred_alt[:nr].count(2)/nr
    col_ratio = comp_gid_pred_alt[nr:].count(2)/nc
    
    
    if row_ratio <= col_ratio:
        #column orientation - also default orientation
        for j in range(nc):
            table_np = np.array(table)
            r_table = list(table_np[:,j])
            pred_label = pr_cl[j]
            heading = r_table[0]
            num_val = r_table[1:]

            # check heading
            head_flag = check_heading(t_name,r_table, caption, pred_label)
            if not head_flag:
                comp_gid_pred_dict['pred_col_label'][j] = 0

            # direct matching
            if pred_label == 0:
                mod_pred_label = direct_matching(heading)
                if mod_pred_label!=0:
                    comp_gid_pred_dict['pred_col_label'][j] = mod_pred_label

            #for tm_liq
            tm_liq_list = ['t m liq', 'tm liq', 'tmliq']
            tm_liq_found = any([element in heading.lower() for element in tm_liq_list])
            if tm_liq_found and pred_label!= 18:
                comp_gid_pred_dict['pred_col_label'][j] = 18

            #for improving recall of activation energy
            ea_list = ['E0', 'Ae', 'Ea', 'E a', 'Ec', 'E c', 'E A', 'E s', 'E dc', 'Edc', 'Es']
            ea_found = any([element in heading for element in ea_list])
            if ea_found and pred_label == 0:
                clean_heading, unit = clean_names(heading)
                if unit == None: #same as unit is None
                    foundd = False
                else:    
                    arb = ['/mol', '/ mol', 'mol-1', 'mol -1', '/at', '/ at', 'at-1', 'at -1', 'eV']
                    foundd = any([element in unit for element in arb])
                #cap_ch = 'energy band gap'
                cap_ch2 = 'activation ene'
                cap_check = False
                if cap_ch2.lower() in caption.lower():
                    cap_check = True
                if foundd or cap_check:
                    comp_gid_pred_dict['pred_col_label'][j] = 20

            #for diff props
            if heading in ['Y (GPa)', 'Y(GPa)', 'y (GPa)', 'y(GPa)', 'Y (Gpa)', 'Y(Gpa)'] and pred_label!=6:
                comp_gid_pred_dict['pred_col_label'][j] = 6
            elif heading in ['m (GPa)', 'm(GPa)', 'M (GPa)', 'M(GPa)', 'M (Gpa)', 'M(Gpa)'] and pred_label!=7:
                comp_gid_pred_dict['pred_col_label'][j] = 7
            elif heading in ['H (GPa)', 'H(GPa)', 'h (GPa)', 'h(GPa)', 'H (Gpa)', 'H(Gpa)'] and pred_label!=8:
                comp_gid_pred_dict['pred_col_label'][j] = 8

            # for labelling exo as cryst and endo as melting
            if 'exo' in heading and ('t' in heading.lower() or 'Temp' in caption.lower()) :
                comp_gid_pred_dict['pred_col_label'][j] = 11
            elif 'endo' in heading and ('t' in heading.lower() or 'Temp' in caption.lower()):
                comp_gid_pred_dict['pred_col_label'][j] = 12
                
            #for 10-6 in EC, DiecC, ExpC
            power_regex = re.compile(r'10[-−](\d+)')
            match = power_regex.search(heading)
            if match and pred_label in [13, 14, 17]:
                x = int(match.group(1))
                values = [find_num(v) for v in num_val if find_num(v)!=None]
                numeric_values = [float(value) for value in values]
                median = np.median(numeric_values)
                
                if not math.isnan(median):
#                     print(f'The controversial case of 10-{x} = {(pii, t_idx)}')
                    assert len(num_val) == nr-1
                    if median>0.1:
                        for ind, val in enumerate(num_val):
                            new_val = find_num(comp_data_dict[(pii, t_idx)]['act_table'][ind+1][j]) #same as val, just for confidence
                            if new_val!=None:
                                comp_data_dict[(pii, t_idx)]['act_table'][ind+1][j] = str(new_val * 10**-x)
        
    else:
        #row orientation
        for i in range(nr):
            table_np = np.array(table)
            r_table = list(table_np[i,:])
            pred_label = pr_rl[i]
            heading = r_table[0]
            num_val = r_table[1:]

            #check heading
            head_flag = check_heading(t_name,r_table, caption, pred_label)
            if not head_flag:
                comp_gid_pred_dict['pred_row_label'][i] = 0


            #direct matching
            if pred_label == 0:
                mod_pred_label = direct_matching(heading)
                if mod_pred_label!=0:
                    comp_gid_pred_dict['pred_row_label'][i] = mod_pred_label

            #for tm_liq
            tm_liq_list = ['t m liq', 'tm liq', 'tmliq']
            tm_liq_found = any([element in heading.lower() for element in tm_liq_list])
            if tm_liq_found and pred_label!= 18:
                comp_gid_pred_dict['pred_row_label'][i] = 18

            #for improving recall of activation energy
            ea_list = ['E0', 'Ae', 'Ea', 'E a', 'Ec', 'E c', 'E A', 'E s', 'E dc', 'Edc', 'Es']
            ea_found = any([element in heading for element in ea_list])
            if ea_found and pred_label == 0:
                clean_heading, unit = clean_names(heading)
                if unit == None: #same as unit is None
                    foundd = False
                else:    
                    arb = ['/mol', '/ mol', 'mol-1', 'mol -1', '/at', '/ at', 'at-1', 'at -1', 'eV']
                    foundd = any([element in unit for element in arb])
                #cap_ch = 'energy band gap'
                cap_ch2 = 'activation ene'
                cap_check = False
                if cap_ch2.lower() in caption.lower():
                    cap_check = True
                if foundd or cap_check:
                    comp_gid_pred_dict['pred_row_label'][i] = 20

            #for diff props
            if heading in ['Y (GPa)', 'Y(GPa)', 'y (GPa)', 'y(GPa)', 'Y (Gpa)', 'Y(Gpa)'] and pred_label!=6:
                comp_gid_pred_dict['pred_row_label'][i] = 6
            elif heading in ['m (GPa)', 'm(GPa)', 'M (GPa)', 'M(GPa)', 'M (Gpa)', 'M(Gpa)'] and pred_label!=7:
                comp_gid_pred_dict['pred_row_label'][i] = 7
            elif heading in ['H (GPa)', 'H(GPa)', 'h (GPa)', 'h(GPa)', 'H (Gpa)', 'H(Gpa)'] and pred_label!=8:
                comp_gid_pred_dict['pred_row_label'][i] = 8

            # for labelling exo as cryst and endo as melting
            if 'exo' in heading and ('t' in heading.lower() or 'Temp' in caption.lower()) :
                comp_gid_pred_dict['pred_row_label'][i] = 11
            elif 'endo' in heading and ('t' in heading.lower() or 'Temp' in caption.lower()):
                comp_gid_pred_dict['pred_row_label'][i] = 12
                
            #for 10-6 in EC, DiecC, ExpC
            power_regex = re.compile(r'10[-−](\d+)')
            match = power_regex.search(heading)
            if match and pred_label in [13, 14, 17]:
                x = int(match.group(1))
                values = [find_num(v) for v in num_val if find_num(v)!=None]
                numeric_values = [float(value) for value in values]
                median = np.median(numeric_values)
                
                if not math.isnan(median):
                    if median>0.1:
#                     print(f'The controversial case of 10-{x} = {(pii, t_idx)}')
                        assert len(num_val) == nc-1
                        for ind, val in enumerate(num_val):
                            new_val = find_num(comp_data_dict[(pii, t_idx)]['act_table'][i][ind+1])
                            if new_val!=None:
                                comp_data_dict[(pii, t_idx)]['act_table'][i][ind+1] = str(new_val * 10**-x)
        
   
    # updating the values
    pr_rl, pr_cl = comp_gid_pred_dict['pred_row_label'], comp_gid_pred_dict['pred_col_label']
    comp_gid_pred = comp_gid_pred_dict['pred_row_label'] + comp_gid_pred_dict['pred_col_label']
    # print(comp_gid_pred)
    assert len(comp_gid_pred) == nr+nc
    table = comp_data_dict[(pii, t_idx)]['act_table']
    
    
    tuples = []
    
    
    if 2 in comp_gid_pred_alt:
        if row_ratio <= col_ratio:
            # orient is col
            if 2 in comp_gid_pred_alt[nr:]:
                gid_index = None
                if 1 in comp_gid_pred[nr:]:
                    gid_index = comp_gid_pred[nr:].index(1)
                # prop_col_annotation = 2
                for prop_col_annotation in range(2,21):
                    for j in range(nc):
                        if comp_gid_pred[nr:][j] == prop_col_annotation:
                            table_np = np.array(table)
                            r_table = list(table_np[:,j])
                            unit = set_units(r_table, prop_col_annotation, pii, t_idx)
                            
                            prop_name = prop_names[prop_col_annotation-2]
                            heading = r_table[0]
                            if unit=='':
                                unit = check_heading_for_unit(heading, prop_name)
                                if unit!='':
                                    unit = norm_unit(unit, prop_name)
                                    
                            check_unit = check_non_controv_unit(prop_name, unit)
                            if not check_unit:
                                new_unit_from_heading = check_heading_for_unit(heading, prop_name)
                                if new_unit_from_heading!='':
                                    unit = norm_unit(new_unit_from_heading, prop_name)
                            
                            prop_flag = check_whether_in_limit(t_name, r_table, prop_col_annotation)
                            if not prop_flag:
                                #change the label, delete the respective tuples
                                comp_gid_pred_dict['pred_col_label'][j] = 0
                                continue

                            
                                
                            # Creating tuples
                            for i in range(nr):
                                if gid_index == None:
                                    gid = f'{pii}_{t_idx}_{i}_{j}'
                                else:
                                    gid = f'{pii}_{t_idx}_{i}_{j}_{table[i][gid_index]}'
                                num = find_num(table[i][j])
                                # TODO: also identify prop name
                                if num != None and num != 0:
                                    #tuples.append((gid, f'prop_{prop_col_annotation}', num, unit))
                                    tuples.append((gid, prop_names[prop_col_annotation-2], num, unit))
        else:
            # orient is row
            if 2 in comp_gid_pred_alt[:nr]:
                gid_index = None
                if 1 in comp_gid_pred[:nr]:
                    gid_index = comp_gid_pred[:nr].index(1)
                # prop_row_annotation = 2
                for prop_row_annotation in range(2,21):
                    for i in range(nr):
                        if comp_gid_pred[:nr][i] == prop_row_annotation:
                            table_np = np.array(table)
                            r_table = list(table_np[i,:])
                            unit = set_units(r_table, prop_row_annotation, pii, t_idx)
                            
                            prop_name = prop_names[prop_row_annotation-2]
                            heading = r_table[0]
                            if unit=='':
                                unit = check_heading_for_unit(heading, prop_name)
                                if unit!='':
                                    unit = norm_unit(unit, prop_name)
                                    
                            check_unit = check_non_controv_unit(prop_name, unit)
                            if not check_unit:
                                new_unit_from_heading = check_heading_for_unit(heading, prop_name)
                                if new_unit_from_heading!='':
                                    unit = norm_unit(new_unit_from_heading, prop_name)
                            
                            prop_flag = check_whether_in_limit(t_name,r_table ,prop_row_annotation)
                            if not prop_flag:
                            #change the label, delete the respective tuples
                                comp_gid_pred_dict['pred_row_label'][i] = 0
                                continue

                            for j in range(nc):
                                if gid_index == None:
                                    gid = f'{pii}_{t_idx}_{i}_{j}'
                                else:
                                    gid = f'{pii}_{t_idx}_{i}_{j}_{table[gid_index][j]}'
                                num = find_num(table[i][j])
                                if not prop_flag:
                                #change the label, delete the respective tuples
                                    continue
                                pred_label = pr_cl[j]
                                head_flag = check_heading(t_name,r_table, caption, pred_label)
                                if not head_flag:
                                    continue
                                    
                                # TODO: also identify prop name
                                if num != None and num != 0:
                                    #tuples.append((gid, f'prop_{prop_row_annotation}', num, unit))
                                    tuples.append((gid, prop_names[prop_row_annotation-2], num, unit))
                                    
    return tuples, comp_gid_pred_dict



losses = ['row_col_label_loss', 'constraint']
coeffs = [1.0, args.c_loss_lambda]


def train_model(epoch):
    model.train()
    epoch_loss = {l: 0.0 for l in losses}
    curr_coeffs = coeffs.copy()
    if epoch < 3:
        curr_coeffs[1] = 0.0

    n_batches = len(loaders['train'])
    tepoch = tqdm(loaders['train'], unit='batch')
    batch_loss = dict()

    for batch_data in tepoch:
        tepoch.set_description(f'Epoch {epoch}')
        torch.cuda.empty_cache()
        row_col_logits, row_col_labels, (batch_loss[losses[1]], ), = model(batch_data)
        batch_loss[losses[0]] = row_col_loss_fn(row_col_logits, row_col_labels)
        for l in losses:
            epoch_loss[l] += batch_loss[l].item()
        loss = sum(curr_coeffs[i] * batch_loss[losses[i]] for i in range(len(losses)))
        optim.zero_grad()
        loss.backward()
        optim.step()
        scheduler.step()
        del row_col_logits, row_col_labels

    for l in losses:
        epoch_loss[l] /= n_batches
    return epoch_loss


def eval_model(split, debug=False):
    model.eval()
    identifier = []
    y_comp_true, y_comp_pred, ret_comp_pred = [], [], []
    # y_edge_true, y_edge_pred, ret_edge_pred = [], [], []

    with torch.no_grad():

        tepoch = tqdm(loaders[split], unit='batch')
        for batch_data in tepoch:
            tepoch.set_description(f'{split} mode')
            comp_gid_logits, comp_gid_labels = model(batch_data)
            y_comp_true += comp_gid_labels.cpu().detach().tolist()
            # y_edge_true += edge_labels.cpu().detach().tolist()
            y_comp_pred += comp_gid_logits.argmax(1).cpu().detach().tolist()
            # num_rows, num_cols = [x['num_rows'] for x in batch_data], [x['num_cols'] for x in batch_data]
            # pred_comp_gid_labels, pred_edge_labels, batch_pred_edges = get_batch_edge_gid_pred_labels(comp_gid_logits.cpu().detach(), edge_logits.cpu().detach(), num_rows, num_cols)
            pred_comp_gid_labels = comp_gid_logits.argmax(1).cpu().detach().tolist() # TODO: check
            # y_edge_pred += pred_edge_labels
            # ret_edge_pred += batch_pred_edges
            del comp_gid_logits, comp_gid_labels

            base_comp_gid = 0
            total_len = 0
            for x in batch_data:
                identifier.append((x['pii'], x['t_idx']))
                comp_dict = dict()
                comp_dict['pred_row_label'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_rows']]
                base_comp_gid += x['num_rows']
                comp_dict['pred_col_label'] = pred_comp_gid_labels[base_comp_gid:base_comp_gid+x['num_cols']]
                base_comp_gid += x['num_cols']
                total_len += x['num_rows'] + x['num_cols']
                comp_dict['act_table'] = x['act_table']
                if 'prop_orient' in x.keys():
                    comp_dict['prop_orient'] = x['prop_orient']
                else: comp_dict['prop_orient'] = None
                ret_comp_pred.append(comp_dict)
            assert len(pred_comp_gid_labels) == total_len



    ret_tuples_pred, all_pred_tuples = [], []
    ret_tuples_true, all_true_tuples = [], []
    y_comp_pred = []
    # for (pii, t_idx), edges, comp_gid_pred in zip(identifier, ret_edge_pred, ret_comp_pred):
    for (pii, t_idx), comp_gid_pred in zip(identifier, ret_comp_pred):
        # pred_tuples, label = get_pred_tuples(pii, t_idx, edges, comp_gid_pred, split)
        pred_tuples, mod_gid_pred_dict = get_pred_tuples(pii, t_idx, comp_gid_pred, split)
        ret_tuples_pred.append(pred_tuples)
        all_pred_tuples += pred_tuples
        true_tuples = get_gold_tuples(pii, t_idx)
        ret_tuples_true.append(true_tuples)
        all_true_tuples += true_tuples
        y_comp_pred += mod_gid_pred_dict['pred_row_label'] + mod_gid_pred_dict['pred_col_label']
        # type_labels.append(label)

    # print(f'ret comp pred {ret_comp_pred}')
    row_col_pred_dict = {id: item for id, item in zip(identifier, ret_comp_pred)}

    
    m_pred_tuples = temp_cut_off(all_pred_tuples)
    mod_pred_tuples = remove_tuples_on_units(m_pred_tuples)
    #sys.exit('Check now')
    
    
    prop_gid_metrics = pd.DataFrame(classification_report(y_comp_true, y_comp_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], output_dict=True)).round(3)
    print(f'{split} prop_gid_metrics = \n')
    print(prop_gid_metrics)
    print()
    
    # removing the tuples containing dielectric const as we arent interested in them
    new_pred_tup = [tup for tup in mod_pred_tuples if tup[1]!='Dielectric constant']
    new_gold_tup = [tup for tup in all_true_tuples if tup[1]!='Dielectric constant']
    
    
    # Final post-processing - only on tuples.
    # This post-processing is based on units, and it improved the score, specially Precision.
    for ind, tuples in enumerate(new_pred_tup):
#             print(f'tuples outside = {tuples}')
            pii = tuples[0].split('_')[0]
            prop_name = tuples[1]
            unit = tuples[-1]
            check_unit = check_non_controv_unit(prop_name, unit)
            if not check_unit:


                if prop_name == 'Density' and unit in ['A', 'Å']:
                    new_pred_tup.remove(tuples)
                    continue
               
                new_unit = find_unit_in_paper(pii, prop_name)
                new_unit = new_unit.strip()
                    
                if new_unit!='':
                    change = 1
                    new_unit_norm = norm_unit(new_unit, prop_name)
                    new_tuple = (*tuples[:-1], new_unit_norm)
#                     print(f'Removed tuple = {tuples}')
#                     print(f'New tuple = {new_tuple}')
                    #raise RuntimeError("stopping execution.")
                    new_pred_tup[ind] = new_tuple
#                     print(f'Updated pred tuples = {pred_tuples}')
#                     print()


                elif new_unit=='' and prop_name in ['Glass transition temperature', 'Crystallization temp', 'Melting temp', 'Softening Point (Temperature)', 'Annealing Point (Temperature)', 'Liquidus temperature','Softening Point (Viscosity)']:

                    new_unit = 'degC'
                    change = 1
                    new_unit_norm = norm_unit(new_unit, prop_name)
                    new_tuple = (*tuples[:-1], new_unit_norm)
                    new_pred_tup[ind] = new_tuple


                elif new_unit=='' and prop_name in ['Density', "Fracture toughness", 'Poisson ratio', 'Thermal expansion coefficient', 'Electric conductivity', 'Refractive index']:
                    if not check_paper_for_prop(pii, prop_name):
                        #remove tuples form pred_tuples
                        new_pred_tup.remove(tuples)
                        continue
                        
                    

    ulti_gold_tuples = []
    for tupp in new_gold_tup:
        n_unit = norm_unit(tupp[-1], tupp[1])
        n_tupp = (tupp[0], tupp[1], tupp[2], n_unit)
        ulti_gold_tuples.append(n_tupp)

    tuple_metrics = get_tuples_metrics(all_true_tuples, all_pred_tuples, split)
    # composition_metrics = get_composition_metrics(gold_tuples[split], all_pred_tuples)

    if not debug:
        return prop_gid_metrics, tuple_metrics
    else:
        return identifier, (prop_gid_metrics, y_comp_pred, ret_comp_pred), (tuple_metrics, ret_tuples_pred, ret_tuples_true)


best_val = 0.0
epoch_loss_list = []
epoch_list = []
for epoch in range(num_epochs):
    epoch_loss = train_model(epoch)
    print(f'Epoch {epoch} | Loss {epoch_loss}')
    epoch_loss_list.append(epoch_loss['row_col_label_loss'] + epoch_loss['constraint'] * 50)
    epoch_list.append(epoch)
    train_stats = eval_model('train')
    print('Train Stats\n', train_stats)
    val_stats = eval_model('val')
    print('Val Stats\n', val_stats)
    test_stats = eval_model('test')
    print('Test Stats\n', test_stats)
    print()

    if val_stats[-1]['fscore'] > best_val:
    # if val_stats[-1]['fscore'] > best_val:
        best_val = val_stats[-1]['fscore']
        torch.save(model.state_dict(), args.model_save_file)
        
model.load_state_dict(torch.load(args.model_save_file, map_location=torch.device(device)))
model = model.to(device)

res = {'train': dict(), 'val': dict(), 'test': dict()}

print('\nInference\n')
for split in res.keys():
    res[split]['identifier'], (res[split]['prop_gid_metrics'], res[split]['y_comp_pred'], res[split]['ret_comp_pred']), (res[split]['tuple_metrics'], res[split]['ret_tuples_pred'], res[split]['ret_tuples_true']) = eval_model(split, debug=True)
    print(f'Inference {split} stats')
    print('Prop gid metrics')
    print(res[split]['prop_gid_metrics'])
    print('Tuple metrics')
    print(res[split]['tuple_metrics'])
    print('Violations')
    for n, f in violation_funcs.items():
        violations, total = 0, 0
        for table in res[split]['ret_comp_pred']:
            v, t = f(table)
            violations += v
            total += t
        print(f'\t{split} {n} violations: {violations}/{total}')
        res[split][f'{n}_violations'] = violations

if args.res_file:
    os.makedirs(os.path.join(table_dir, 'res_dir'), exist_ok=True)
    pickle.dump(res, open(os.path.join(table_dir, 'res_dir', args.res_file), 'wb'))
    # os.remove(args.model_save_file)

print()
print(f'Epoch Loss List = {epoch_loss_list}')
print(epoch_list)
plt.plot(epoch_list, epoch_loss_list, label='training loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.savefig(f'loss_curve_5e-5_1e-5_{args.seed}.png')

