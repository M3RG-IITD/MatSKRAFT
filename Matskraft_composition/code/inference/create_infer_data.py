import os
import pickle
import sys
sys.path.append('..')

from transformers import AutoTokenizer
from tqdm import tqdm

from normalize_text import normalize


table_dir = '../../data'
pii_table_dict = pickle.load(open(os.path.join(table_dir, 'pii_table_dict.pkl'), 'rb'))
# pii_table_dict = pickle.load(open(os.path.join(table_dir, 'all_test_keys.pkl'), 'rb'))

lm_name = 'm3rg-iitd/matscibert'
cache_dir = os.path.join(table_dir, '.cache')
tokenizer = AutoTokenizer.from_pretrained(lm_name, cache_dir=cache_dir, model_max_length=512)


for pii in tqdm(pii_table_dict):
    for t_idx, d in enumerate(pii_table_dict[pii]):
#     for t_idx, d in enumerate(pii_table_dict):
        d['pii'] = pii
        d['t_idx'] = t_idx
        d['num_rows'] = len(d['act_table'])
        d['num_cols'] = len(d['act_table'][0])
        d['num_cells'] = d['num_rows'] * d['num_cols']
        
        table_vec = []
        for row in d['act_table']:
            table_vec += row
        table_vec = [normalize(cell) for cell in table_vec]
        tok = tokenizer(table_vec, max_length=50, truncation=True)
        d['input_ids'] = tok['input_ids']
        d['attention_mask'] = tok['attention_mask']

        tok = tokenizer([normalize(d['caption'])])
        d['caption_input_ids'] = tok['input_ids']
        d['caption_attention_mask'] = tok['attention_mask']
        assert len(d['caption_input_ids'][0]) <= 512


data = []
for pii in sorted(pii_table_dict.keys()):
    assert isinstance(pii_table_dict[pii], list)
    data += pii_table_dict[pii]

print(len(data))
pickle.dump(data, open(os.path.join(table_dir, 'infer_table_data.pkl'), 'wb'))
