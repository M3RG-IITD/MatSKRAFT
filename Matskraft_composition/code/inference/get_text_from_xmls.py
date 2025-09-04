""""
input: train_val_test_split.pkl (dictonary of lists)
    keys: split name (eg train, val, test)
    values: list of tuples (pii, table_idx)

output: train_val_test_paper_data.pkl (dictionary with all piis as keys)
    values: dictionary (section names as keys, section text as values)
"""

import os
import pickle

from bs4 import BeautifulSoup
from tqdm import tqdm


table_dir = '../../data'

piis = sorted(os.listdir(os.path.join(table_dir, 'piis')))#[:100]


def get_contents(pii):
    # return dictionary with keys as section names and values as corresponding section text
    path = os.path.join(table_dir, 'piis', pii, f'{pii}.xml')
    with open(path, 'r', encoding='utf-8') as file:
        soup = BeautifulSoup(file.read(), 'xml')

    sec = soup.find('xocs:item-toc')
    en = sec.findAll('xocs:item-toc-entry')

    snum, sname = [], []
    for s in en:
        try:
            snum.append(s.find('xocs:item-toc-label').contents[0])
            sname.append(s.find('xocs:item-toc-section-title').contents[0])
        except:
            pass

    paper = {
        'Title': ' '.join(soup.find('dc:title').text.split(',')).strip() if soup.find('dc:title') else '',
        'Abstract': soup.find('dc:description').text.replace('Abstract', '').replace('\n', '').strip() if soup.find('dc:description') else '',
    }

    sname.insert(0, 'Abstract')
    snum.insert(0, '')

    all_sections = soup.find_all('ce:section-title')
    for sec in all_sections:
        strr = ''
        if sec.text in sname:
            secid = sname.index(sec.text)
            if '.' not in snum[secid]:
                for tx in sec.find_next_siblings():
                    strr += tx.text.strip().replace('\n', ' ') + '\n'
                paper[f'{snum[secid]}_{sec.text}'] = strr.strip()

    if paper['Abstract'] == '':
        paper['Abstract'] = paper['_Abstract']
        paper.pop('_Abstract')

    return paper


text_data = dict()
for pii in tqdm(piis):
    try:
        text_data[pii] = get_contents(pii)
    except AttributeError as e:
        continue

pickle.dump(text_data, open(os.path.join(table_dir, 'inference_paper_text.pkl'), 'wb'))
