from collections import Counter, defaultdict
import os
import pickle
import re
import sys
import pdb

import numpy as np
from sklearn.linear_model import LogisticRegression
from sympy import sympify, solve
from torch.utils.data import Dataset

sys.path.append('..')
from regex_lib import *


table_dir = '../../data'

text_data = pickle.load(open(os.path.join(table_dir, 'inference_paper_text_test.pkl'), 'rb'))


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


def rectify_comp_labels(row_labels, col_labels):
    row_1, row_2, col_1, col_2 = row_labels.count(1), row_labels.count(2), col_labels.count(1), col_labels.count(2)
    if row_1 + col_1 == 0 or row_2 + col_2 == 0 or row_1 + row_2 == 0 or col_1 + col_2 == 0:
        return [0] * len(row_labels), [0] * len(col_labels)
    if (row_2 + col_1 == 0) or (row_1 + col_2 == 0):
        return row_labels, col_labels
    if row_1 == 0:
        return row_labels, [0 if c == 2 else c for c in col_labels]
    if row_2 == 0:
        return row_labels, [0 if c == 1 else c for c in col_labels]
    if col_1 == 0:
        return [0 if r == 2 else r for r in row_labels], col_labels
    if col_2 == 0:
        return [0 if r == 1 else r for r in row_labels], col_labels
    if row_1 > row_2 and col_2 > col_1:
        return [0 if r == 2 else r for r in row_labels], [0 if c == 1 else c for c in col_labels]
    if row_2 > row_1 and col_1 > col_2:
        return [0 if r == 1 else r for r in row_labels], [0 if c == 2 else c for c in col_labels]
    return [0] * len(row_labels), [0] * len(col_labels)


elements_compounds = pickle.load(open(os.path.join(table_dir, 'elements_compounds.pkl'), 'rb'))
comp_num_pattern = r'(\d+\.\d+|\d+/\d+|\d+)'
ele_num = r'((' + '|'.join(elements_compounds['elements']) + r')' + comp_num_pattern + r'?)'
many_ele_num = r'(' + ele_num + r')+'
ele_comp_pattern = r'(((\(' + many_ele_num + '\)' + comp_num_pattern + r')|(' + many_ele_num + r'))+)'
ele_comp_pattern = r'(?:^|\W)(' + ele_comp_pattern + r'|Others|Other)(?:\W|$)'
num = r'(\d+\.\d+|\d+\/\d+|\d+)'

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

def is_valid_number(value):
    """
    Check if the value is a valid number (not NaN, infinity, or zero).
    """
    try:
        sym_value = sympify(value)
        return sym_value.is_finite and sym_value != 0
    except (ValueError, TypeError):
        return False


def get_mcc_ci_pred_tuples(c, edges, comp_gid_pred):
    pii, t_idx = c['pii'], c['t_idx']
    row_labels, col_labels = comp_gid_pred['row'], comp_gid_pred['col']
    tuples = []
    if sum(row_labels) == 0 or sum(col_labels) == 0: return tuples
    
    def proc_edges(e):
        return (e[0][0] * c['num_cols'] + e[0][1], e[1][0] * c['num_cols'] + e[1][1])
    
    edges = np.array(list(map(proc_edges, edges)))
    if 1 in row_labels:
        comp_cols = [j for j in range(c['num_cols']) if col_labels[j] == 2]
        gid_cols = [j for j in range(c['num_cols']) if col_labels[j] == 3]
        gid_col = gid_cols[0] if gid_cols else None
        for i in range(c['num_rows']):
            if row_labels[i] != 1: continue
            g_id = f'{pii}_{t_idx}_{i}_{comp_cols[0]}_0'
            if gid_col is not None and c['act_table'][i][gid_col]:
                g_id += '_' + c['act_table'][i][gid_col]
            for j in comp_cols:
                s = re.search(num, c['act_table'][i][j])
#                 if s is None or float(s.group()) == 0: continue
                if s is None:
                    continue

                value = s.group()
                if '/' in value:
                    value = sympify(value)
                if not is_valid_number(value):
                    continue
                src = i * c['num_cols'] + j
                src_edges = edges[edges[:, 0] == src]
                dst = src_edges[0][1]
                dst_contents = c['act_table'][dst // c['num_cols']][dst % c['num_cols']]
                ele_comp = re.findall(ele_comp_pattern, dst_contents)
                if len(ele_comp) == 0: continue
                tuples.append((g_id, ele_comp[0][0] if ele_comp[0][0] != 'Other' else 'Others', float(value), pred_cell_mol_wt(c, i, j, pii)))
    else:
        comp_rows = [i for i in range(c['num_rows']) if row_labels[i] == 2]
        gid_rows = [i for i in range(c['num_rows']) if row_labels[i] == 3]
        gid_row = gid_rows[0] if gid_rows else None
        for j in range(c['num_cols']):
            if col_labels[j] != 1: continue
            g_id = f'{pii}_{t_idx}_{comp_rows[0]}_{j}_0'
            if gid_row is not None and c['act_table'][gid_row][j]:
                g_id += '_' + c['act_table'][gid_row][j]
            for i in comp_rows:
                s = re.search(num, c['act_table'][i][j])
#                 if s is None or float(s.group()) == 0: continue
                if s is None:
                    continue

                value = s.group()
                if '/' in value:
                    value = sympify(value)
                if not is_valid_number(value):
                    continue
                src = i * c['num_cols'] + j
                src_edges = edges[edges[:, 0] == src]
                dst = src_edges[0][1]
                dst_contents = c['act_table'][dst // c['num_cols']][dst % c['num_cols']]
                ele_comp = re.findall(ele_comp_pattern, dst_contents)
                if len(ele_comp) == 0: continue
                tuples.append((g_id, ele_comp[0][0] if ele_comp[0][0] != 'Other' else 'Others', float(value), pred_cell_mol_wt(c, i, j, pii)))
    return tuples


extracted_regex = pickle.load(open(os.path.join(table_dir, 'extracted_regex.pkl'), 'rb')) #updated for all
all_elements, all_compounds = elements_compounds['elements'], elements_compounds['compounds']
# sorted_compounds_elements = sorted(all_compounds + all_elements, key=len, reverse=True)
# comp_pattern = re.compile(r'(?:^|[^a-zA-Z])(' + '|'.join(sorted_compounds_elements) + r')(?:[^a-zA-Z]|$)')
# comp_pattern_1 = re.compile(r'(' + '|'.join(sorted_compounds_elements) + r')')
comp_pattern = re.compile(r'(?:^|[^a-zA-Z])(' + '|'.join(all_compounds + all_elements) + r')(?:[^a-zA-Z]|$)')
comp_pattern_1 = re.compile(r'(' + '|'.join(all_compounds+all_elements) + r')')
var_pattern = re.compile(r'(?:^|[^\w-])(' + '|'.join(comp_vars) + r')')
var_pattern_1 = re.compile(r'(' + '|'.join(comp_vars) + r')')

def add_space_before_mol_percent(text):
    # The pattern looks for mol% preceded by a letter or digit and captures that character
    pattern = re.compile(r'([A-Za-z0-9])(mol%|wt%|at%)')
    
    # This replacement adds a space between the captured letter/digit and "mol%"
    modified_text = re.sub(pattern, r'\1 \2', text)
    
    return modified_text


count_def = 0
def get_comp_vars_and_nums(table):
    global count_def
    comps, vars, nums = [], [], []
    for r in table:
        r_comps, r_vars, r_nums = [], [], []
        for cell in r:
            cell = add_space_before_mol_percent(cell)
            found_compounds = re.findall(comp_pattern, cell)
            r_comps.append(found_compounds)
            subs_cell = re.sub(comp_pattern_1, ' ', cell)
            found_vars = list(set(m.group(1) for m in re.finditer(var_pattern, subs_cell)))
            r_vars.append(found_vars)
            subs_cell = re.sub(var_pattern_1, ' ', subs_cell)
            cell_nums = re.findall(num, subs_cell)
#             print(f'cell_nums = {cell_nums}')
#             sys.exit(1)
            for indd, cell_n in enumerate(cell_nums):
                if '/' in cell_n:
                    new_vall = sympify(cell_n)
                    if new_vall.is_finite:
                        cell_nums[indd] = new_vall
                    else:
                        cell_nums[indd] = None
                    count_def+=1
#             print(count_def)
            cell_nums = [cell for cell in cell_nums if cell is not None]
            cell_nums = list(map(float, cell_nums))
            r_nums.append(cell_nums[0] if len(cell_nums) > 0 else None)
        comps.append(r_comps)
        vars.append(r_vars)
        nums.append(r_nums)
    return comps, vars, nums


def get_clf_feats(c, row_labels, col_labels, edges):
    comps, vars, nums = get_comp_vars_and_nums(c['act_table'])
    edge_dict = {src: dst for src, dst in edges}
    dsts = sorted(set([e[1] for e in edges]))
    found_vars, found_compounds = [], []
    for dst in dsts:
        found_compounds += comps[dst[0]][dst[1]]
        found_vars += vars[dst[0]][dst[1]]
    found_compounds, found_vars = set(found_compounds), set(found_vars)
    max_s, avg_s = 0, 0
    if 1 in row_labels:
        comp_rows = [r for r in range(c['num_rows']) if row_labels[r] == 1]
        comp_cols = [j for j in range(c['num_cols']) if col_labels[j] == 2]
        for r in comp_rows:
            s = 0
            for j in comp_cols:
                if nums[r][j]:
                    s += nums[r][j]
            max_s = max(max_s, s)
            avg_s += s
        avg_s /= len(comp_rows)
    else:
        comp_rows = [r for r in range(c['num_rows']) if row_labels[r] == 2]
        comp_cols = [j for j in range(c['num_cols']) if col_labels[j] == 1]
        for j in comp_cols:
            s = 0
            for r in comp_rows:
                if nums[r][j]:
                    s += nums[r][j]
            max_s = max(max_s, s)
            avg_s += s
        avg_s /= len(comp_cols)
    return [len(found_compounds), len(found_vars), avg_s, max_s, (row_labels + col_labels).count(2)]


model_dir = '../../models'
clf = pickle.load(open(os.path.join(model_dir, 'pi_predictor.pkl'), 'rb'))

# pii_tables = []
con_pii = set()

def common_value_in_list(elements):
    """
    Helper function to determine the common value in a list of lists.
    Returns the value if it appears in more than 50% of the lists, otherwise None.
    """
    from collections import Counter
    
    total_elements = len(elements)
    if total_elements == 0:
        return None
    
    # Flatten the list of lists and count occurrences of each value
    all_elements = [elem for sublist in elements for elem in sublist if isinstance(elem, str)]
    
    if not all_elements:
        return None
    
    count = Counter(all_elements)
    
    for value, freq in count.items():
        if freq >= 0.50 * total_elements:
            return value
    
    return None

def process_table(table, orientation, r, c):
    """
    Process the table based on the given orientation ('row' or 'col') to assign heading values 
    based on common values in columns or rows.
    """
    if orientation == 'col':
        num_columns = len(table[0])
        for col in range(num_columns):
            column_elements = [table[row][col] for row in range(1, len(table))]
            common_value = common_value_in_list(column_elements)
            if common_value is not None:
                table[r][col] = [common_value]
    elif orientation == 'row':
        num_rows = len(table)
        for row in range(num_rows):
            row_elements = table[row][1:]
            common_value = common_value_in_list(row_elements)
            if common_value is not None:
                table[row][c] = [common_value]
    else:
        raise ValueError("Orientation must be 'row' or 'col'")
    
    return table

def get_mcc_pi_pred_tuples(cc, edges, comp_gid_pred):
    pii, t_idx = cc['pii'], cc['t_idx'] #pii=='S0022309300002568' and t_idx==0
#     if (pii=='S002230930300379X' and t_idx==0) or (pii=='S002230930300019X' and t_idx==0):
#         pdb.set_trace()
#     pii_tables.append((pii, t_idx))
#     if pii=='S0022309314005638' and t_idx==0:
#         pdb.set_trace()
    row_labels, col_labels = comp_gid_pred['row'], comp_gid_pred['col'] #13, 10
    tuples = []
    if sum(row_labels) == 0 or sum(col_labels) == 0: return tuples
    comps, vars, nums = get_comp_vars_and_nums(cc['act_table']) #13, 10, no of comp/vars in each cell stored in list, 3d array, 13*10*1/2
    
    # Access the first column
    cv_first_column = []
    for sublist in vars:
        cv_first_column.append(sublist[0])
    for ind, sublist in enumerate(comps):
        if any(sublist[0]):
            cv_first_column[ind].append(sublist[0])   
    num_first_column = [row[0] for row in nums]
    filtered_a = [a_fil for a_fil, b_fil in zip(cv_first_column, num_first_column) if a_fil and b_fil is not None]
    threshold = len(filtered_a)/len(cv_first_column)*100
    if threshold>=50:
        flat_list = [item for sublist in filtered_a for item in sublist if isinstance(item, str)]
        unique_elements = set(flat_list)
        if len(unique_elements)== 1:
            comps[0][0].append(flat_list[0])
    
    for i in range(cc['num_rows']):
        for j in range(cc['num_cols']):
            comps[i][j] += vars[i][j]
    del vars
    edge_dict = {src: dst for src, dst in edges}

    keys_left = set(k for k in extracted_regex[pii].keys() if type(k) == str) - set(['Title', 'Abstract'])
    keys_left = sorted([k for k in keys_left if not k.startswith('Appendix') and not k.endswith('_footer')], key=lambda x: int(x.split('_')[0]))
    regex_comps = []
    for k in [t_idx, f'{t_idx}_footer', 'Title', 'Abstract'] + keys_left:
        try:
            regex_comps += extracted_regex[pii][k]
        except:
            regex_comps += []
    all_nums = []
    for r in range(cc['num_rows']):
        for c in range(cc['num_cols']):
            if row_labels[r] * col_labels[c] == 2 and nums[r][c] is not None: #looking for cells with intersection of 1 and 2
                cx, cy = edge_dict[(r, c)] #connecting cells to headers
                if len(comps[cx][cy]) > 0: #if the headers contains any variable or compound name
                    all_nums.append(nums[r][c]) #append the corresponding numbers
    if any(n > 1 for n in all_nums):
        ds = [100]
    else:
        ds = [1, 100]

    if 1 in row_labels:
        comp_cols = [j for j in range(cc['num_cols']) if col_labels[j] == 2] #storing indexes where 2 is present in columns
        gid_cols = [j for j in range(cc['num_cols']) if col_labels[j] == 3] 
        gid_col = gid_cols[0] if gid_cols else None
        for r in range(cc['num_rows']):
            if row_labels[r] != 1: continue
            vars, compounds = dict(), dict()
            first_comp_col = -1
            for c in comp_cols:
                if nums[r][c] is None: continue
                cx, cy = edge_dict[(r, c)] #header index
                comps = process_table(comps, 'col', cx, cy)
                if len(comps[cx][cy]) == 0: continue #extracting the variable of compound present in header
                if first_comp_col == -1: first_comp_col = c
                if comps[cx][cy][0] in comp_vars: #comp_vars = ['x', 'y', 'z', 'X']
                    if comps[cx][cy][0].lower() not in vars:
                        vars[comps[cx][cy][0].lower()] = nums[r][c] #storing variables with their rexpective value as dict
                else:
                    if comps[cx][cy][0] not in compounds:
                        compounds[comps[cx][cy][0]] = nums[r][c]
            if len(vars) == 0 and len(compounds) == 0: continue #no variable or comp with value present
            assert first_comp_col != -1
            found = False
            if len(vars) > 0:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_comp_vars = set()
                    for _, p in comp: #if variable is present in text
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower()) # next line, this_comp_vars == variables present in text
                    if this_comp_vars != set(vars.keys()): continue #variables present in text corresponding to compound==found in table header
                    new_comp = []
                    for ele_comp, p in comp: #comp = extracted composition=(compound, value); from text
                        if type(p) != str: #value present for compound
                            new_comp.append((ele_comp, p))
                        else: #value given as variable for compound
                            p = p.lower()
                            for var, val in vars.items():
                                p = p.replace(var, str(val)) #replace value of variable with value extracted from table
                            new_comp.append((ele_comp, eval_expr(p)))
                    if all(0 <= p <= 1 for _, p in new_comp):
                        found = True
                        break
            else:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    #regex_comp = [([('B', 0.2), ('O', 0.8)], (1687, 1690)), ([('SiO2', '(60)/(60+15+x+25-x+y)'), ('B2O3', '(15)/(60+15+x+25-x+y)'), ('Na2O', '(x)/(60+15+x+25-x+y)'), ('Al2O3', '(25-x)/(60+15+x+25-x+y)'), ('Nd2O3', '(y)/(60+15+x+25-x+y)')], (37, 81))]
                    this_compounds = [x[0] for x in comp] #storing the compound's name obtained in each composition of text
                    if len(set(compounds.keys()) - set(this_compounds)) > 0: continue #compounds contain compounds found in table, this compound contains compounds obtained from one glass composition expression in text
                    comp_dict = {x[0]: x[1] for x in comp}
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars: #comp_vars = ['x', 'y', 'z', 'X']
                            if v in p:
                                this_comp_vars.add(v.lower())
                    for d in ds:
                        var_mapping = dict()
                        suitable_comp = True
                        for compound, p in compounds.items(): #compounds = {'PbF2': 29.1} compound and their value obtained from table for current row
                            if p / d > 1:
                                suitable_comp = False
                                break
                            try:
                                sol = solve(sympify(f'Eq({comp_dict[compound]}, {p/d})'))
                            except NotImplementedError:
                                sol = ''
                            if len(sol) == 0:
                                suitable_comp = False
                                break
                            if type(sol[0]) == dict or float(sol[0]) < 0:
                                suitable_comp = False
                                break
                            m = re.search(var_pattern_1, comp_dict[compound])
                            assert m is not None
                            var = m.group().lower()
                            if var in var_mapping and var_mapping[var] != float(sol[0]):
                                suitable_comp = False
                                break
                            var_mapping[var] = float(sol[0])
                        
                        new_comp = []

                        if (comp_dict[compound] != p/d) and (not suitable_comp or set(var_mapping.keys()) != this_comp_vars): continue
                        
                        if (comp_dict[compound] == p/d): 
                            if (not suitable_comp or set(var_mapping.keys()) != this_comp_vars):
                                con_pii.add(pii)
                        
                        for ele_comp, p in comp:
                            if type(p) != str:
                                new_comp.append((ele_comp, p))
                            else:
                                p = p.lower()
                                for var, val in var_mapping.items():
                                    p = p.replace(var, str(val))
                                new_comp.append((ele_comp, eval_expr(p)))
                        if all(0 <= p <= 1 for _, p in new_comp):
                            found = True
                            break
                    if found: break
            if found:
                g_id = f'{pii}_{t_idx}_{r}_{first_comp_col}_0'
                if gid_col is not None and cc['act_table'][r][gid_col]:
                    g_id += '_' + cc['act_table'][r][gid_col]
                for ele_comp, p in new_comp:
                    if p > 0:
                        tuples.append((g_id, ele_comp, round(p, 5), pred_cell_mol_wt(cc, r, first_comp_col, pii)))
    else:
        assert 1 in col_labels
        comp_rows = [i for i in range(cc['num_rows']) if row_labels[i] == 2]
        gid_rows = [i for i in range(cc['num_rows']) if row_labels[i] == 3]
        gid_row = gid_rows[0] if gid_rows else None
        for c in range(cc['num_cols']):
            if col_labels[c] != 1: continue
            vars, compounds = dict(), dict()
            first_comp_row = -1
            for r in comp_rows:
                if nums[r][c] is None: continue
                cx, cy = edge_dict[(r, c)]
                comps = process_table(comps, 'row', cx, cy)
                if len(comps[cx][cy]) == 0: continue
                if first_comp_row == -1: first_comp_row = r
                if comps[cx][cy][0] in comp_vars:
                    if comps[cx][cy][0].lower() not in vars:
                        vars[comps[cx][cy][0].lower()] = nums[r][c]
                else:
                    if comps[cx][cy][0] not in compounds:
                        compounds[comps[cx][cy][0]] = nums[r][c]
            
            if len(vars) == 0 and len(compounds) == 0: continue
            assert first_comp_row != -1
            found = False
            if len(vars) > 0:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower()) # next line, this_comp_vars == variables present in table
                    if this_comp_vars != set(vars.keys()): continue #take only table values, not text values
                    new_comp = []
                    for ele_comp, p in comp:
                        if type(p) != str:
                            new_comp.append((ele_comp, p))
                        else:
                            p = p.lower()
                            for var, val in vars.items():
                                p = p.replace(var, str(val))
                            new_comp.append((ele_comp, eval_expr(p)))
                    if all(0 <= p <= 1 for _, p in new_comp):
                        found = True
                        break
            else:
                for comp, _ in regex_comps:
                    assert isinstance(comp, list)
                    this_compounds = [x[0] for x in comp]
                    if len(set(compounds.keys()) - set(this_compounds)) > 0: continue
                    comp_dict = {x[0]: x[1] for x in comp}
                    this_comp_vars = set()
                    for _, p in comp:
                        if type(p) != str: continue
                        for v in comp_vars:
                            if v in p:
                                this_comp_vars.add(v.lower())
                    for d in ds:
                        var_mapping = dict()
                        suitable_comp = True
                        for compound, p in compounds.items():
                            if p / d > 1:
                                suitable_comp = False
                                break
                            sol = solve(sympify(f'Eq({comp_dict[compound]}, {p/d})'))
                            if len(sol) == 0:
                                suitable_comp = False
                                break
                            if type(sol[0]) == dict or float(sol[0]) < 0:
                                suitable_comp = False
                                break
                            m = re.search(var_pattern_1, comp_dict[compound])
                            assert m is not None
                            var = m.group().lower()
                            if var in var_mapping and var_mapping[var] != float(sol[0]):
                                suitable_comp = False
                                break
                            var_mapping[var] = float(sol[0])

                        if (comp_dict[compound] != p/d) and (not suitable_comp or set(var_mapping.keys()) != this_comp_vars): continue
                            
                        if (comp_dict[compound] == p/d): 
                            if (not suitable_comp or set(var_mapping.keys()) != this_comp_vars):
                                con_pii.add(pii)
                        
                        new_comp = []
                        for ele_comp, p in comp:
                            if type(p) != str:
                                new_comp.append((ele_comp, p))
                            else:
                                p = p.lower()
                                for var, val in var_mapping.items():
                                    p = p.replace(var, str(val))
                                new_comp.append((ele_comp, eval_expr(p)))
                        if all(0 <= p <= 1 for _, p in new_comp):
                            found = True
                            break
                    if found: break
            if found:
                g_id = f'{pii}_{t_idx}_{first_comp_row}_{c}_0'
                if gid_row is not None and cc['act_table'][gid_row][c]:
                    g_id += '_' + cc['act_table'][gid_row][c]
                for ele_comp, p in new_comp:
                    if p > 0:
                        tuples.append((g_id, ele_comp, round(p, 5), pred_cell_mol_wt(cc, first_comp_row, c, pii)))
                        
    
#     if (pii, t_idx) in pii_list:
#         pdb.set_trace()
                        
#     if pii=='S0022309300002568' and t_idx==0:
#         print(tuples)
        
    return tuples


def get_pred_tuples(c, edges, comp_gid_pred):

    def proc_edges(e):
        return (e[0][0] * c['num_cols'] + e[0][1], e[1][0] * c['num_cols'] + e[1][1])

    if sum(comp_gid_pred['row']) == 0 or sum(comp_gid_pred['col']) == 0:
        return [], 3 # NC

    y_pred = clf.predict([get_clf_feats(c, comp_gid_pred['row'], comp_gid_pred['col'], edges)])[0]
    if y_pred == 1:
        return get_mcc_pi_pred_tuples(c, edges, comp_gid_pred), 2 # MCC-PI
    else:
        return get_mcc_ci_pred_tuples(c, edges, comp_gid_pred), 1 # MCC-CI


def cnt_1_2_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    return d['row'].count(1) * d['col'].count(1) + d['row'].count(2) * d['col'].count(2), 2 * r * c


def cnt_1_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    return d['row'].count(1) * d['row'].count(3) + d['col'].count(1) * d['col'].count(3), r * (r - 1) + c * (c - 1)


def cnt_2_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    return d['row'].count(2) * d['col'].count(3) + d['row'].count(3) * d['col'].count(2), 2 * r * c


def cnt_3_3_violations(d: dict):
    r, c = len(d['row']), len(d['col'])
    cnt_3 = (d['row'] + d['col']).count(3)
    return cnt_3 * (cnt_3 - 1) // 2, (r + c) * (r + c - 1) // 2


violation_funcs = {
    '1_2_violations': cnt_1_2_violations,
    '1_3_violations': cnt_1_3_violations,
    '2_3_violations': cnt_2_3_violations,
    '3_3_violations': cnt_3_3_violations,
}

