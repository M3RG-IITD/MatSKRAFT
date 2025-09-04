from collections import defaultdict
import pickle
import os
import sys
sys.path.append('..')

from tqdm import tqdm
import time

from regex_lib import parse_composition
import pdb


table_dir = '../../data'
comp_data = pickle.load(open(os.path.join(table_dir, 'infer_table_data_test.pkl'), 'rb')) #list of 6457 dicts - each dict containing keys - dict_keys(['doi', 'act_table', 'caption', 'footer', 'pii', 't_idx', 'num_rows', 'num_cols', 'num_cells', 'input_ids', 'attention_mask', 'caption_input_ids', 'caption_attention_mask'])

#pdb.set_trace()
data_dict = {(c['pii'], c['t_idx']): c for c in comp_data}
text_data = pickle.load(open(os.path.join(table_dir, 'inference_paper_text_test.pkl'), 'rb')) #Extracted text of all the papers

test_keys = pickle.load(open(os.path.join(table_dir, 'all_test_keys.pkl'), 'rb'))



# pii_tidx_check_list = [('S002230930300019X', 0), ('S0022309303001030', 0), ('S0022309303001030', 0), ('S002230930300379X', 0),
# ('S002230939800578X', 0), ('S0022309398007194', 2), ('S0022309301008006', 0)]

# pii_tidx_check_list = [('S0022309302015223', 0)]

extracted_regex = defaultdict(dict)
# for pii, t_idx in tqdm(data_dict.keys()):
for pii, t_idx in tqdm(test_keys):
#     print(pii, t_idx)
# for pii, t_idx in pii_tidx_check_list:
#     pdb.set_trace()
    c = data_dict[(pii, t_idx)]
    extracted_regex[pii][t_idx] = parse_composition(c['caption'].replace('\n', ' '))
    extracted_regex[pii][f'{t_idx}_footer'] = []
    for f in c['footer'].values():
        extracted_regex[pii][f'{t_idx}_footer'] += parse_composition(f)

    if pii in text_data:
        for section, text in text_data[pii].items():
            if section not in extracted_regex[pii].keys():
                extracted_regex[pii][section] = parse_composition(text)
    else:
        extracted_regex[pii]['Title'] = []
        extracted_regex[pii]['Abstract'] = []

pickle.dump(extracted_regex, open(os.path.join(table_dir, 'extracted_regex.pkl'), 'wb'))