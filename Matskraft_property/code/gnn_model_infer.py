import numpy as np
import pandas as pd

import torch
from torch import nn, Tensor, LongTensor, tensor
import torch.nn.functional as F

from transformers import AutoModel, AutoConfig

import dgl
from dgl.nn import GATConv


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class GNN_Model(nn.Module):

    @staticmethod
    def get_all_pair(ar):
        return np.array(np.meshgrid(ar, ar)).T.reshape(-1, 2)

    @staticmethod
    def get_edges(r, c, ret_extra_edges=False, ret_caption_edges=False):
        edges = np.empty((0, 2), dtype=int)
        row_edges = GNN_Model.get_all_pair(np.arange(c))
        for i in range(r):
            edges = np.concatenate((edges, row_edges + i * c), axis=0)
        col_edges = GNN_Model.get_all_pair(np.arange(0, r * c, c))
        for i in range(c):
            edges = np.concatenate((edges, col_edges + i), axis=0)
        edges = np.unique(edges, axis=0)
        table_edges = LongTensor(edges[np.lexsort((edges[:, 1], edges[:, 0]))])
        if ret_extra_edges:
            table_cells = torch.arange(r * c)
            row_edges = torch.stack([table_cells, r * c + table_cells // c]).T
            col_edges = torch.stack([table_cells, r * c + r + table_cells % c]).T
            row_self_edges = torch.stack([r * c + torch.arange(r), r * c + torch.arange(r)]).T
            col_self_edges = torch.stack([r * c + r + torch.arange(c), r * c + r + torch.arange(c)]).T
            if not ret_caption_edges:
                return table_edges, torch.cat([row_edges, col_edges, row_self_edges, col_self_edges])
            row_col_edges = torch.cat([row_edges, col_edges, torch.flip(row_edges, (1, )), torch.flip(col_edges, (1, )), row_self_edges, col_self_edges])
            caption_edges = torch.stack([(r * c + r + c) * torch.ones(r + c + 1).long(), r * c + torch.arange(r + c + 1)]).T
            return table_edges, torch.cat([row_col_edges, caption_edges])
        return table_edges


    @staticmethod
    def get_all_pairs_torch(n, ordered=False):
        if ordered:
            return torch.cat([torch.combinations(torch.arange(n)), torch.combinations(torch.arange(n-1, -1, -1))])
        return torch.combinations(torch.arange(n))

    @staticmethod
    def get_row_col_pairs(r, c):
        all_pairs = torch.combinations(torch.arange(r+c))
        v1 = torch.reshape(all_pairs[:,0]<r, (-1,1))
        v2 = torch.reshape(all_pairs[:,1]>=r, (-1,1))
        accept_mask = torch.all(torch.cat((v1, v2), dim=1), dim=1)
        del v1, v2
        return all_pairs[accept_mask]

    @staticmethod
    def get_block(h_in, h_out):
        return nn.Sequential(
            nn.Linear(h_in, h_out),
            nn.BatchNorm1d(h_out),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

    @staticmethod
    def validate_args(args):
        assert isinstance(args['hidden_layer_sizes'], list)
        assert isinstance(args['num_heads'], list)
        assert isinstance(args['use_caption'], bool)
        assert len(args['hidden_layer_sizes']) == len(args['num_heads'])
        return args

    def __init__(self, args: dict):
        super(GNN_Model, self).__init__()
        self.args = self.validate_args(args)

        config = AutoConfig.from_pretrained(args['lm_name'], cache_dir=args['cache_dir'])
#         config = AutoConfig.from_pretrained('../../../saved_matscibert')
        self.encoder = AutoModel.from_pretrained(args['lm_name'], config=config, cache_dir=args['cache_dir'])
#         self.encoder = AutoModel.from_pretrained('../../../saved_matscibert', config=config)

        in_dim = config.hidden_size
        self.default_embedding = nn.Embedding(1, in_dim)
        self.positional_embeddings = nn.Embedding(4, in_dim)


        self.gat_layers = nn.ModuleList()
        self.gat_layers.append(
            GATConv(in_dim, self.args['hidden_layer_sizes'][0], num_heads=self.args['num_heads'][0], residual=False))

        for l in range(1, len(self.args['hidden_layer_sizes'])):
            self.gat_layers.append(
                GATConv(self.args['hidden_layer_sizes'][l-1] * self.args['num_heads'][l-1], self.args['hidden_layer_sizes'][l], \
                num_heads=self.args['num_heads'][l], residual=True))

        out_dim = self.args['hidden_layer_sizes'][-1] * self.args['num_heads'][-1]
        self.dropout = nn.Dropout(0.2)
        
        self.gid_and_prop_layer = nn.Sequential(self.get_block(out_dim, 256), nn.Linear(256, 21))

    def _encoder_forward(self, input_ids, attention_mask):
        lm_inp = {'input_ids': input_ids, 'attention_mask': attention_mask}
        max_len = max(len(s) for s in lm_inp['input_ids'])
        for k in lm_inp.keys():
            lm_inp[k] = [s + [0] * (max_len - len(s)) for s in lm_inp[k]]
            lm_inp[k] = LongTensor(lm_inp[k]).to(device)
        return self.encoder(**lm_inp)[0][:, 0]


    def calc_constraint_loss(self, inps, row_col_logits):
        # print("gid_logits shape", gid_logits.shape)
        all_probs = F.softmax(row_col_logits, dim=1)
        gid_index = 1
        prop_index = 2
        all_gid_probs = all_probs[:, gid_index]
        # all_prop_probs_old = all_probs[:, prop_index]
        all_prop_probs = all_probs[:, prop_index: prop_index+19].sum(axis=1) # add up probabilities of all prop labels: 2, 3, 4, 5
        # print('all_prop_probs_old.shape', all_prop_probs_old.shape)
        # print('all_prop_probs.shape', all_prop_probs.shape)

        base = 0
        constraints = {'prop_prop': [], 'gid_prop': [], 'gid_gid': [], 'gid_id' : []}
        #constraints = {'prop_prop': [], 'gid_prop': [], 'gid_gid': []}
        constraints['gid_id'].append(tensor([-0.1]))
        for x in inps:
            gid_probs = all_gid_probs[base:base+x['num_rows']+x['num_cols']]
            prop_probs = all_prop_probs[base:base+x['num_rows']+x['num_cols']]
            # gid_probs_rows = gid_probs[:x['num_rows']]
            # gid_probs_cols = gid_probs[]
            
            
            act_table = np.array(x['act_table'])
            unique_list = []
            for index, eleme in enumerate(all_gid_probs[base: base + x['num_rows']]):
                ele_list, ele_set = [], set()
                if eleme>0.25:
                    for ele in act_table[index,:]:
                        ele_list.append(ele)
                        ele_set.add(ele)
                    #print(ele_list)
                    #print(ele_set)
                    unique = 0.5 - (len(ele_set)/len(ele_list))
                    unique_list.append(unique)
                    #constraints['gid_id'].append(tensor(unique))
        
            for index, eleme in enumerate(all_gid_probs[base + x['num_rows']:base + x['num_rows'] + x['num_cols']]):
                ele_list, ele_set = [], set()
                if eleme>0.25:
                    for ele in act_table[:,index]:
                        ele_list.append(ele)
                        ele_set.add(ele)
                    #print(ele_list)
                    #print(ele_set)
                    unique = 0.5 - (len(ele_set)/len(ele_list))
                    #constraints['gid_id'].append(torch.tensor(unique))
                    unique_list.append(unique)
            
            # prop_probs_rows = prop_probs[:x['num_rows']]
            # prop_probs_cols = prop_probs[x['num_rows']:]
            
            constraints['gid_id'].append(tensor(unique_list))
            
            base += x['num_rows'] + x['num_cols']

            row_col_pairs = self.get_all_pairs_torch(x['num_rows'] + x['num_cols'])
            constraints['gid_gid'].append(gid_probs[row_col_pairs[:, 0]] + gid_probs[row_col_pairs[:, 1]] - 1)

            row_col_cross_pairs = self.get_row_col_pairs(x['num_rows'], x['num_cols'])

            constraints['gid_prop'].append(torch.cat((gid_probs[row_col_cross_pairs[:, 0]] + prop_probs[row_col_cross_pairs[:, 1]] -1, prop_probs[row_col_cross_pairs[:, 0]] + gid_probs[row_col_cross_pairs[:, 1]] -1)))

            constraints['prop_prop'].append(prop_probs[row_col_cross_pairs[:, 0]] + prop_probs[row_col_cross_pairs[:, 1]] - 1)
            
            

            del gid_probs, prop_probs, row_col_pairs, row_col_cross_pairs
        del all_probs, all_gid_probs, all_prop_probs
        torch.cuda.empty_cache()

        constraint_loss = tensor(0.0).to(device)
        #print(constraints)
        for c in constraints.keys():
            constraint_loss += F.relu(torch.cat(constraints[c])).mean()
        return constraint_loss

    def forward(self, inps):
        # print(type(inps), len(inps), len(inps[0]))
#         for key in inps[0]:
#             print(key, inps[0][key])
#         print()
        lm_inp = {'input_ids': [], 'attention_mask': []}
        for x in inps:
            for k in lm_inp.keys():
                lm_inp[k] += x[k]
        embs = []
        for idx in range(0, len(lm_inp['input_ids']), 160):
            embs.append(self._encoder_forward(lm_inp['input_ids'][idx:idx+160], lm_inp['attention_mask'][idx:idx+160]))
        del lm_inp
        cell_h = torch.cat(embs)
        del embs
        
        if self.args['use_caption']:
            lm_inp = {'input_ids': [], 'attention_mask': []}
            for x in inps:
                for k in lm_inp.keys():
                    lm_inp[k] += x[f'caption_{k}']
            caption_h = self._encoder_forward(lm_inp['input_ids'], lm_inp['attention_mask'])
            del lm_inp

        mask_keys = ['cell', 'row', 'col']
        if self.args['use_caption']:
            mask_keys.append('caption')
        mask = {k: [] for k in mask_keys}

        row_positional_idxs, col_positional_idxs = [], []
        for x in inps:
            mask['cell'] += [1] * x['num_cells'] + [0] * (x['num_rows'] + x['num_cols'] )
            mask['row'] += [0] * x['num_cells'] + [1] * x['num_rows'] + [0] * (x['num_cols'])
            mask['col'] += [0] * (x['num_cells'] + x['num_rows']) + [1] * x['num_cols']
            

            table_cells = np.arange(x['num_cells']).reshape(x['num_rows'], x['num_cols'])
            row_nums = table_cells // x['num_cols']
            row_nums[row_nums > 2] = 2
            row_nums += 1
            row_positional_idxs += row_nums.flatten().tolist() + row_nums[:, 0].tolist() + [0] * x['num_cols']

            col_nums = table_cells % x['num_cols']
            col_nums[col_nums > 2] = 2
            col_nums += 1
            col_positional_idxs += col_nums.flatten().tolist() + [0] * x['num_rows'] + col_nums[0].tolist()
            
            if self.args['use_caption']:
                mask['caption'] += [0] * (x['num_cells'] + x['num_rows'] + x['num_cols'])
                for k in mask_keys:
                    mask[k].append(1 if k == 'caption' else 0)
                row_positional_idxs.append(0)
                col_positional_idxs.append(0)

        for k in mask_keys:
            mask[k] = Tensor(mask[k]).bool().to(device)

        h = self.default_embedding(LongTensor([0] * len(mask['cell'])).to(device))
        # print("h.shape", h.shape)
        # dimension of h will be (num_cells + num_rows + num_cols + 1) * 768
        h[mask['cell']] = cell_h
        del cell_h
        if self.args['use_caption']:
            h[mask['caption']] = caption_h
            del caption_h
        h += self.positional_embeddings(LongTensor(row_positional_idxs).to(device)) + \
        self.positional_embeddings(LongTensor(col_positional_idxs).to(device))

        base = 0
        batch_all_edges = []
        for x in inps:
            table_edges, extra_edges = self.get_edges(x['num_rows'], x['num_cols'], ret_extra_edges=True, ret_caption_edges=self.args['use_caption'])
            batch_all_edges.append(torch.cat([table_edges, extra_edges]) + base)
            base += x['num_cells'] + x['num_rows'] + x['num_cols']
            if self.args['use_caption']:
                base += 1

        batch_all_edges = torch.cat(batch_all_edges)
        batch_g = dgl.graph((batch_all_edges[:, 0], batch_all_edges[:, 1])).to(device)

        for l in range(len(self.gat_layers)):
            h = F.elu(self.gat_layers[l](batch_g, h)).flatten(1)
        h = self.dropout(h)

        batch_row_col_gid_labels = []


        row_col_logits = self.gid_and_prop_layer(h[mask['row'] | mask['col']])


        ret = (row_col_logits, batch_row_col_gid_labels)

        if self.training:
            if self.args['add_constraint']:
                constraint_loss = self.calc_constraint_loss(inps, row_col_logits)
            else:
                constraint_loss = tensor(0.0).to(device)
            ret += (constraint_loss, ),

        return ret
