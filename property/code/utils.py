from collections import defaultdict, Counter
import os
import pickle
import re
import sys
sys.path.append('..')

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

# os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"

# from regex_lib import parse_composition

table_dir = '../data/'
# table_dir = '../data_old/'

## Chose this part carefully whether you are - training, evaluating on val/test dataset, or inferencing

comp_data = pickle.load(open(os.path.join(table_dir, 'final_augmented_train_data_with_tuples_verified_2.pkl'), 'rb')) + \
            pickle.load(open(os.path.join(table_dir, 'final_val_test_data_with_id_with_tuple_revised_updated.pkl'), 'rb'))
# comp_data = pickle.load(open(os.path.join(table_dir, 'all_disco_data.pkl'), 'rb'))
# comp_data = pickle.load(open(os.path.join(table_dir, 'final_val_test_data_with_id_with_tuple_revised.pkl'), 'rb'))
comp_data = pickle.load(open(os.path.join(table_dir, 'all_pega_data_test.pkl'), 'rb'))
# TODO: generate val_test_data.pkl
comp_data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}
print(len(comp_data_dict))
comp_keys = list(comp_data_dict.keys())
# TODO: generate train_val_test_split

# Comment out this part while inferencing:

train_val_test_split = pickle.load(open(os.path.join(table_dir, 'train_val_test_split_for_prop.pkl'), 'rb'))
# splits = ['train', 'val', 'test']
# # TODO: convert labels 1, 2 to 0
# # 0 (others), 1 (glass id), 2 (property)
# for split in splits:
#     for pii_t_idx in train_val_test_split[split]:
#         t = comp_data_dict[pii_t_idx]
#         r, c = t['num_rows'], t['num_cols']
#         t['prop_row_label'], t['prop_col_label'] = [0 for _ in range(r)], [0 for _ in range(c)]
#         # if t['prop_table']:
#         for i, l in enumerate(t['row_label']):
#             if l > 2:
#                 t['prop_row_label'][i] = l - 2
#         for i, l in enumerate(t['col_label']):
#             if l > 2:
#                 t['prop_col_label'][i] = l - 2




class TableDataset(Dataset):
    def __init__(self, data):
        self.inp = data

    def __getitem__(self, idx):
        return self.inp[idx]

    def __len__(self):
        return len(self.inp)




def get_gold_tuples(pii, t_idx):
    tuples = []
    c = comp_data_dict[(pii, t_idx)]
    if c['prop_table'] == False:
        return tuples
   
    for k in range(len(c['prop_tuples'])):
        for i in range(c['num_rows']):
            for j in range(c['num_cols']):
                tuples += c['prop_tuples'][k][i][j]
                

        
        
    return tuples



def match_tuple(p, g, split):
    
    if p[0].count('_') == 4:
        p0 = p[0][:p[0].rindex('_')]
    else:
        p0 = p[0]
        
    if g[0].count('_') == 4:
        g0 = g[0][:g[0].rindex('_')]
    else:
        g0 = g[0]
        

    try:
        return p0 == g0 and p[1] == g[1] and round(abs(p[2] - g[2]),2)==0 and p[3] == g[3]
    except TypeError:
        print('Check for type mismatch')
        print(p)
        print(g)
        return False
            

def get_tuples_metrics(gold_tuples, pred_tuples, split):
    global mismatch_train, mismatch_val, mismatch_test
    prec = 0
    for p in pred_tuples:
        for g in gold_tuples:
            if match_tuple(p, g, split):
                prec += 1
                break
        # if p in gold_tuples:
        #     prec += 1
    if len(pred_tuples) > 0:
        prec /= len(pred_tuples)
    else:
        prec = 0.0
    rec = 0
    for g in gold_tuples:
        for p in pred_tuples:
            if match_tuple(p, g, split):
                rec += 1
                break
        # if g in pred_tuples:
        #     rec += 1
    rec /= len(gold_tuples)
    fscore = 2 * prec * rec / (prec + rec) if (prec + rec > 0) else 0.0

    return {'precision': round(prec,4), 'recall': round(rec,4), 'fscore': round(fscore,4), 'support': len(gold_tuples)}

def get_property_metrics(gold_tuples, pred_tuples, prop_names, split):
    
    results = {}
    for prop in prop_names:
        if prop == 'Dielectric constant':
            continue
        # Filter tuples for the current property
        gold_filtered = [t for t in gold_tuples if t[1] == prop]
        pred_filtered = [t for t in pred_tuples if t[1] == prop]
        
        # Calculate metrics for the filtered tuples
        results[prop] = get_tuples_metrics(gold_filtered, pred_filtered, split)
    
    return results


def find_num(string):
    #remove tabs or unnecessary spaces
    try:
        string = string.strip()
    except:
        pass
    #e.g. already in int or float form: 12.5 -> 12.5
    try:
        if string[0] == '-':
            return float(string[1:])*(-1)
        else:
            return float(string)
    except:
        pass
    # e.g. 99.1x10-7
    range_regex = re.compile('^\d+\.?\d*\s*x\s*\d+\.?\d*[-|+]?\s*\d+\.?\d*$')
    try:
#         print('alla')
        ranges_ut = range_regex.search(string).group().split('x')
        ranges = [x.strip() for x in ranges_ut]
        #print(f'ranges = {ranges}')
        if '-' in ranges[1]:
            numu = ranges[1].split('-')
#             print(ranges)
#             print(float(numu[0]), float(numu[1]))
            num = float(ranges[0]) * pow(float(numu[0]), (-float(numu[1])))
            formatted_result = "{:.3e}".format(num)
            try:
                # Try to convert using literal_eval
                return float(formatted_result)
            except (ValueError, SyntaxError):
                # If literal_eval fails, return None or handle the error as needed
                return None
        elif '+' in ranges[1]:
            numu = ranges[1].split('+')
            num = float(ranges[0]) * pow(float(numu[0]), (float(numu[1])))
#             print(num)
            return num
        elif ranges[1].startswith('10') and 3<=len(ranges[1])<=4 and ranges[1].isdigit():
            numu0 = 10
            numu1 = ranges[1][2:]
            num = float(ranges[0]) * pow(float(numu0), (float(numu1)))
            return num
        num = float(ranges[0]) * float(ranges[1])
        formatted_result = "{:.3e}".format(num)
        return formatted_result
    except:
        pass
    #e.g. 12.5 - 13.5 -> 12.5
    range_regex = re.compile('^\d+\.?\d*\s*-\s*\d+\.?\d*')
    try:
        ranges = range_regex.search(string).group().split('-')
        num = float(ranges[0])
#         print(num)
        return num
    except:
        pass
#     print('alla')
    #e.g. 12.2 (5.2) -> 12.2
    bracket_regex = re.compile('(\d+\.?\d*)\s*\(\d*.?\d*\)')
    try:
        extracted_value = float(bracket_regex.search(string).group(1))
        return float(extracted_value)
    except:
        pass
    #e.g. 12.3 ± 0.5 -> 12.3
    plusmin_regex = re.compile('^(\d+\.?\d*)(\s*[±+-]+\s*\d+\.?\d*)')
    try:
        extracted_value = float(plusmin_regex.search(string).group(1))
        return extracted_value
    except AttributeError:
        pass
    #e.g. <0.05 -> 0.05  |  >72.0 -> 72.0    | ~12 -> 12
    lessthan_roughly_regex = re.compile('([<]|[~]|[>])=?\s*\d+\.*\d*')
    try:
        extracted_value = lessthan_roughly_regex.search(string).group()
        num_regex = re.compile('\d+\.*\d*')
        extracted_value = num_regex.search(extracted_value).group()
        return float(extracted_value)
    except:
        pass
    # e.g. 0.4:0.6 (ratios)
    if ':' in string:
        split = string.split(":")
        try:
            extracted_value = round(float(split[0])/float(split[1]), 3)
            return extracted_value
        except:
            pass
    # e.g. 220 [29] --> 220.0, where citations given, rejecting ab 220 [29] or 7 220 [29]
    if '[' in string and ']' in string:
        try:
            extracted_value = string[:string.index('[')]
            return float(extracted_value)
        except:
            pass
    # e.g. 723 (first peak or other text) --> 723.0
    if '(' in string and ')' in string:
        try:
            extracted_value = string[:string.index('(')]
            return float(extracted_value)
        except:
            pass
    # e.g. '150K' or '150 degC' --> 150.0 but not '1350degC/2h' or 'njn 150 njn'
    try:
        exact_number_regex = re.compile('^(\d+\.?\d*)\s*[a-zA-Z]+$')
        # Using search to find the first occurrence of the pattern in the string
        match = exact_number_regex.search(string)
#         print('Match')
#         print(match)
        
        # If a match is found, extract the numeric part and convert to float
        if match:
            numeric_part = match.group(1)
            extracted_value = float(numeric_part)
            return extracted_value
    except:
        pass
    return None


def cnt_4_4_violations(d: dict):
    # properties can be in either rows or in columns
    r, c = len(d['pred_row_label']), len(d['pred_col_label'])
    prop_label = 2
    d['pred_row_label_alt'] = [2 if i in range(2,22) else i for i in d['pred_row_label']]
    d['pred_col_label_alt'] = [2 if i in range(2,22) else i for i in d['pred_col_label']]
    return d['pred_row_label_alt'].count(prop_label) * d['pred_col_label_alt'].count(prop_label), r * c


def cnt_3_4_violations(d: dict):
    # glass id and prop should be parallel
    r, c = len(d['pred_row_label']), len(d['pred_col_label'])
    glass_id_label = 1
    prop_label = 2
    d['pred_row_label_alt'] = [2 if i in range(2,22) else i for i in d['pred_row_label']]
    d['pred_col_label_alt'] = [2 if i in range(2,22) else i for i in d['pred_col_label']]
    return d['pred_row_label_alt'].count(glass_id_label) * d['pred_col_label_alt'].count(prop_label) + d['pred_col_label_alt'].count(glass_id_label) * d['pred_row_label_alt'].count(prop_label), 2 * r * c

def cnt_3_3_violations(d: dict):
    # there should be only one glass id
    glass_id_label = 1 # change to 3 when you use 3
    r, c = len(d['pred_row_label']), len(d['pred_col_label'])
    cnt_3 = (d['pred_row_label'] + d['pred_col_label']).count(glass_id_label)
    return cnt_3 * (cnt_3 - 1) // 2, (r + c) * (r + c - 1) // 2

def cnt_3_2_violations(d: dict):
    # there should be only one glass id
    glass_id_label = 1 # change to 3 when you use 3
    r, c = len(d['pred_row_label']), len(d['pred_col_label'])
    #print('Keys of d  =  ')
    #print(d.keys())
    act_table = np.array(d['act_table'])
    cnt_6 = 0
    if 1 in d['pred_row_label']:
        index  = d['pred_row_label'].index(1)
        ele_list, ele_set = [], set()
        for ele in act_table[index,:]:
            ele_list.append(ele)
            ele_set.add(ele)
            #print(ele_list)
            #print(ele_set)
        unique = 0.5 - (len(ele_set)/len(ele_list))
        #constraints['gid_id'].append(torch.tensor(unique))
        if unique>0:
            cnt_6 += 1
                
    if 1 in  d['pred_col_label']:
        index = d['pred_col_label'].index(1)
        ele_list, ele_set = [], set()
        for ele in act_table[:,index]:
            ele_list.append(ele)
            ele_set.add(ele)
        unique = 0.5 - (len(ele_set)/len(ele_list))
        if unique>0:
            cnt_6 += 1
     
    return cnt_6, r*c


violation_funcs = {
    'prop_prop_violations': cnt_4_4_violations,
    'gid_prop_violations': cnt_3_4_violations,
    'gid_gid_violations': cnt_3_3_violations,
    'gid_id_violations' : cnt_3_2_violations,
}

