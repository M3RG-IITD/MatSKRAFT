from collections import Counter
import re
import sys
sys.path.append('..')

from torch.utils.data import Dataset


from regex_lib import parse_composition

import pickle
import os

def get_regex_feats(table):
    regex_feats = []
    for r in table:
        for cell in r:
            comp = parse_composition(cell)
            if len(comp) == 0 or len(comp[0][0]) == 1:
                regex_feats.append(1)
            else:
                regex_feats.append(2)
    return regex_feats


def get_max_freq_feat(table):
    max_freq_feat = []
    for r in table:
        cnt = Counter()
        for cell in r:
            if cell: cnt[cell] += 1
        max_freq_feat.append(cnt.most_common(1)[0][1] if cnt else 1)
    for j in range(len(table[0])):
        cnt = Counter()
        for i in range(len(table)):
            cell = table[i][j]
            if cell: cnt[cell] += 1
        max_freq_feat.append(cnt.most_common(1)[0][1] if cnt else 1)
    return max_freq_feat


class TableDataset(Dataset):
    def __init__(self, data):
        self.inp = data

    def __getitem__(self, idx):
        return self.inp[idx]

    def __len__(self):
        return len(self.inp)


table_dir = '../../data'
text_data = pickle.load(open(os.path.join(table_dir, 'inference_paper_text_test.pkl'), 'rb'))

mol_regex = re.compile(r'mol\.?\s*\%|molar\s*(%|percent)', re.IGNORECASE)
wt_regex = re.compile(r'mass\s*(%|percent)|weight\s*(%|percent)|wt\.?\s*\%', re.IGNORECASE)
at_regex = re.compile(r'at\.?\s*\%|atomic\s*(%|percent)', re.IGNORECASE)



def find_mol_wt_in_text(text):
    if re.search(mol_regex, text):
        return 'mol'
    if re.search(wt_regex, text):
        return 'wt'
#     if re.search(at_regex, text):
#         return 'at'
    if re.search(at_regex, text):
        return 'mol'
    return ''


def pred_cell_mol_wt(table_dict, i, j, pii):
    for k in reversed(range(i+j+1)): # increasing L1 distance from (i, j)
        for i_ in range(max(k-j, 0), min(k,i)+1):
            j_ = k - i_
            pred = find_mol_wt_in_text(table_dict['act_table'][i_][j_])
            if pred: return pred

    pred = find_mol_wt_in_text(table_dict['caption'])
    if pred: 
        return pred
    else:
        
        try:
            paper_text = text_data[pii]
        except KeyError:
            return 'mol'
        
        for key in paper_text.keys():
            if 'result' in key.lower() or 'discussion' in key.lower():
                text_intrstd = paper_text[key]
                pred2 = find_mol_wt_in_text(text_intrstd)
                if pred2 : return pred2
                
    return 'mol'


def cnt_3_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    cnt_3 = (d['row'] + d['col']).count(1)
    return cnt_3 * (cnt_3 - 1) // 2, (r + c) * (r + c - 1) // 2

