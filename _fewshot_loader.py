import math
import os
import json
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

import numpy as np
from collections import defaultdict


class FewShotLoader(object):
    def __init__(self, config):
        super(FewShotLoader, self).__init__()
        self.dataset_folder = config.dataset
        self.max_adj = config.max_adj
        self.fold = config.fold
        self.embedding_type = config.embedding_type
        self._check_file(self.dataset_folder)
        self._load_files()
        self._format_everything()
        self._check_valid_candidates()



    def _format_everything(self):
        # leave id = 0 for dummy padding index
        self.ent2id = {k: v + 1 for k, v in self.ent2id.items()}
        self.rel2id = {k: v + 1 for k, v in self.rel2id.items()}
        self.ent_embed = np.vstack([np.random.normal(size=(1, self.ent_embed.shape[1])),
                                    self.ent_embed])
        self.rel_embed = np.vstack([np.random.normal(size=(1, self.rel_embed.shape[1])),
                                    self.rel_embed])
        self.g = defaultdict(dict)
        # self.d = np.ones((len(self.ent2id) + 1), dtype=int)
        self.np_g = np.zeros((2, len(self.ent2id) + 1, self.max_adj), dtype=int)
        with open(os.path.join(self.dataset_folder, 'path_graph'), 'r', encoding='utf-8') as in_f :
            for line in in_f:
                _src, _rel, _dst = line.strip().split('\t')
                _src_idx = self.ent2id[_src]
                _rel_idx = self.rel2id[_rel]
                _dst_idx = self.ent2id[_dst]
                _rel_inv_idx = self.rel2id[_rel + '_inv']
                if _src_idx not in self.g.keys() :
                    self.g[_src_idx]['adj'] = list()
                    self.g[_src_idx]['rel'] = list()
                    # self.g[_src_idx]['adj'].append(_src_idx)
                    # self.g[_src_idx]['rel'].append(0)
                self.g[_src_idx]['adj'].append(_dst_idx)
                self.g[_src_idx]['rel'].append(_rel_idx)


                if _dst_idx not in self.g.keys() :
                    self.g[_dst_idx]['adj'] = list()
                    self.g[_dst_idx]['rel'] = list()
                    # self.g[_dst_idx]['adj'].append(_dst_idx)
                    # self.g[_dst_idx]['rel'].append(0)
                self.g[_dst_idx]['adj'].append(_src_idx)
                self.g[_dst_idx]['rel'].append(_rel_inv_idx)

            for _node_idx, _d in self.g.items():
                _adj = _d['adj'][:self.max_adj]
                _rel = _d['rel'][:self.max_adj]
                self.np_g[0, _node_idx, : (len(_rel))] = _rel
                self.np_g[1, _node_idx, : (len(_adj))] = _adj

                # self.d[_node_idx] = len(self.g[_node_idx]['adj']) # degree


    def _load_files(self):
        with open(os.path.join(self.dataset_folder, 'fold{}'.format(self.fold), 'train_tasks.json'), 'r', encoding='utf-8') as in_f :
            self.train_tasks = json.loads(in_f.readline())
        with open(os.path.join(self.dataset_folder, 'fold{}'.format(self.fold), 'dev_tasks.json'), 'r', encoding='utf-8') as in_f :
            self.dev_tasks = json.loads(in_f.readline())
        with open(os.path.join(self.dataset_folder, 'fold{}'.format(self.fold), 'test_tasks.json'), 'r', encoding='utf-8') as in_f :
            self.test_tasks = json.loads(in_f.readline())
        with open(os.path.join(self.dataset_folder, 'ent2ids'), 'r', encoding='utf-8') as in_f :
            self.ent2id = json.loads(in_f.readline())
        with open(os.path.join(self.dataset_folder, 'relation2ids'), 'r', encoding='utf-8') as in_f :
            self.rel2id = json.loads(in_f.readline())
        with open(os.path.join(self.dataset_folder, 'rel2candidates.json'), 'r', encoding='utf-8') as in_f :
            self.rel2candi = json.loads(in_f.readline())
        with open(os.path.join(self.dataset_folder, 'e1rel_e2.json'), 'r', encoding='utf-8') as in_f :
            self.exclude = json.loads(in_f.readline())

        self.ent_embed = np.loadtxt(os.path.join(self.dataset_folder, 'entity2vec.{}'.format(self.embedding_type)))
        self.rel_embed = np.loadtxt(os.path.join(self.dataset_folder, 'relation2vec.{}'.format(self.embedding_type)))

    def _check_valid_candidates(self):
        def _valid(tasks):
            task_keys = list(tasks.keys())
            valid_tasks = defaultdict(list)
            # task_values = list(tasks.values())

            for task_rel in task_keys:
                task_triples = tasks[task_rel]
                for src, rel, dst in task_triples:
                    flag = 0
                    for candi in self.rel2candi[task_rel]:
                        if candi == dst or candi in self.exclude[src + task_rel]:
                            continue
                        else:
                            flag = 1
                            break
                    if flag:
                        valid_tasks[task_rel].append([src, rel, dst])
            return valid_tasks




        self.train_tasks = _valid(self.train_tasks)
        # self.dev_tasks = _valid(self.dev_tasks)
        # self.test_tasks = _valid(self.test_tasks)

    @staticmethod
    def _check_file(dataset_folder):
        _filename_list = [
            # 'train_tasks.json',
            # 'dev_tasks.json',
            # 'test_tasks.json',
            'ent2ids',
            'relation2ids',
            'rel2candidates.json',
            'entity2vec.TransE',
            'relation2vec.TransE',
            'path_graph',
            'e1rel_e2.json'
        ]
        for filename in _filename_list:
            if not os.path.exists(os.path.join(dataset_folder, filename)):
                raise FileExistsError("File {} is missing, please check the dataset folder")





class FSDataSet(Dataset):
    def __init__(self, config, data_set, mode):
        super(FSDataSet, self).__init__()
        self.data = data_set
        self.mode = mode.lower()
        if self.mode not in ['train', 'dev', 'test']:
            raise NotImplementedError('The argument mode must be one of \{train, dev, test\}.')

        self.k = config.k
        self.bs = config.batch_size
        self.num_positive_samples = config.num_positive_samples
        self.max_step = config.max_step
        self._step_per_epoch = len(self.data.train_tasks)
        self.epoch = math.ceil(config.max_step * self.bs / self._step_per_epoch)

        if self.mode == 'train':
            self._generate_train_items()
        else:
            self._generate_eval_items()


    def _generate_train_items(self):
        self.iter_items = list()

        _train_rels = list(self.data.train_tasks.keys())

        for _ in range(self.epoch - 1):
            self.iter_items.extend(_train_rels)
            random.shuffle(_train_rels)
        self.iter_items.extend(_train_rels[:(self.max_step - len(self.iter_items))])



    def _generate_eval_items(self):
        self.iter_items = list()
        if self.mode == 'dev':
            mode_task = self.data.dev_tasks
        else:
            mode_task = self.data.test_tasks
        for task_rel, triples in mode_task.items():
            task_candi_list = self.data.rel2candi[task_rel]
            support_pair = list()
            # query_pair = list()
            for _src, _rel, _dst in triples[: self.k]:
                _src_idx = self.data.ent2id[_src]
                _dst_idx = self.data.ent2id[_dst]
                support_pair.append([_src_idx, _dst_idx])
            for _src, _rel, _dst in triples[self.k :]:
                _src_idx = self.data.ent2id[_src]
                _dst_idx = self.data.ent2id[_dst]

                _dst_candi_idx_list = [_dst_idx] + [self.data.ent2id[_candi] for _candi in task_candi_list
                                       if _candi != _dst and _candi not in self.data.exclude[_src + _rel]]
                _src_idx_list = [_src_idx] * len(_dst_candi_idx_list)
                query_pair = list(zip(_src_idx_list, _dst_candi_idx_list))
                self.iter_items.append({'sup': support_pair,
                                        'que': query_pair})


    def _fetch_ids(self, src, dst):
        _src_adj = pad_sequence([torch.tensor(self.data.g[_u]['adj'][:100], dtype=torch.long) for _u in src],
                                batch_first=True, padding_value=0).unsqueeze(dim=0)  # [1, k, max_num_neighbor_src]
        _src_rel = pad_sequence([torch.tensor(self.data.g[_u]['rel'][:100], dtype=torch.long) for _u in src],
                                batch_first=True, padding_value=0).unsqueeze(dim=0)  # [1, k, max_num_neighbor_src]
        _dst_adj = pad_sequence([torch.tensor(self.data.g[_u]['adj'][:100], dtype=torch.long) for _u in dst],
                                batch_first=True, padding_value=0).unsqueeze(dim=0)  # [1, k, max_num_neighbor_src]
        _dst_rel = pad_sequence([torch.tensor(self.data.g[_u]['rel'][:100], dtype=torch.long) for _u in dst],
                                batch_first=True, padding_value=0).unsqueeze(dim=0)  # [1, k, max_num_neighbor_src]

        return torch.cat((_src_rel, _src_adj), dim=0), \
               torch.cat((_dst_rel, _dst_adj), dim=0)

    def _fetch_ids_npg(self, src, dst):
        _src = self.data.np_g[:, src, :]
        _dst = self.data.np_g[:, dst, :]
        # return torch.from_numpy(_src), torch.from_numpy(_dst)
        return _src, _dst

    def collate(self, batch):
        batch_dict_list = []
        batch_dict = defaultdict(list)
        for batch_item in batch:
            # print(batch_item)
            # batch = batch[0]
            task_rel = batch_item

            triples = self.data.train_tasks[task_rel]
            task_candi_list = self.data.rel2candi[task_rel]
            support_pair = list()
            positive_pair = list()
            negative_pair = list()

            for _src, _rel, _dst in triples[: self.k] :
                _src_idx = self.data.ent2id[_src]
                _dst_idx = self.data.ent2id[_dst]
                support_pair.append([_src_idx, _dst_idx])
            if len(triples[self.k :]) < self.num_positive_samples :
                positive_triples = [random.choice(triples[self.k :]) for _x in range(self.num_positive_samples)]
            else :
                positive_triples = random.sample(triples[self.k :], self.num_positive_samples)
            for _src, _rel, _dst in positive_triples :
                _src_idx = self.data.ent2id[_src]
                _dst_idx = self.data.ent2id[_dst]
                positive_pair.append([_src_idx, _dst_idx])

                _corrupt_dst = _dst
                while _corrupt_dst == _dst or _corrupt_dst in self.data.exclude[_src + _rel] :
                    _corrupt_dst = random.choice(task_candi_list)
                _corrupt_dst_idx = self.data.ent2id[_corrupt_dst]
                negative_pair.append([_src_idx, _corrupt_dst_idx])
            random.shuffle(self.data.train_tasks[task_rel])
            # print(support_pair)

            _batch_data = {'sup' : support_pair, 'pos' : positive_pair, 'neg' : negative_pair}
            sup_src, sup_dst = list(zip(*_batch_data['sup']))
            pos_src, pos_dst = list(zip(*_batch_data['pos']))
            neg_src, neg_dst = list(zip(*_batch_data['neg']))
            sup_src_meta, sup_dst_meta = self._fetch_ids_npg(sup_src, sup_dst)
            pos_src_meta, pos_dst_meta = self._fetch_ids_npg(pos_src, pos_dst)
            neg_src_meta, neg_dst_meta = self._fetch_ids_npg(neg_src, neg_dst)

            batch_dict['sup'].append(np.vstack([sup_src, sup_dst]))
            batch_dict['pos'].append(np.vstack([pos_src, pos_dst]))
            batch_dict['neg'].append(np.vstack([neg_src, neg_dst]))
            batch_dict['sup_src_meta'].append(sup_src_meta)
            batch_dict['sup_dst_meta'].append(sup_dst_meta)
            batch_dict['pos_src_meta'].append(pos_src_meta)
            batch_dict['pos_dst_meta'].append(pos_dst_meta)
            batch_dict['neg_src_meta'].append(neg_src_meta)
            batch_dict['neg_dst_meta'].append(neg_dst_meta)

            # batch_dict = {
            #     'sup' : torch.tensor((sup_src, sup_dst), dtype=torch.int64),
            #     'pos' : torch.tensor((pos_src, pos_dst), dtype=torch.int64),
            #     'neg' : torch.tensor((neg_src, neg_dst), dtype=torch.int64),
            #     'sup_src_meta' : sup_src_meta,  # [2, k, max_adj]
            #     'sup_dst_meta' : sup_dst_meta,  # [2, k, max_adj]
            #     'pos_src_meta' : pos_src_meta,  # [2, n_pos, max_adj]
            #     'pos_dst_meta' : pos_src_meta,  # [2, n_pos, max_adj]
            #     'neg_src_meta' : neg_src_meta,  # [2, n_pos, max_adj]
            #     'neg_dst_meta' : neg_dst_meta,  # [2, n_pos, max_adj]
            # }
            # batch_dict_list.append(batch_dict)
        batch_dict = {k: torch.LongTensor(v) for k, v in batch_dict.items()}

        return batch_dict


    def collate_eval(self, batch) :
        batch_dict = defaultdict(list)
        for batch_item in batch :
            sup_src, sup_dst = list(zip(*batch_item['sup']))
            que_src, que_dst = list(zip(*batch_item['que']))
            sup_src_meta, sup_dst_meta = self._fetch_ids_npg(sup_src, sup_dst)
            que_src_meta, que_dst_meta = self._fetch_ids_npg(que_src, que_dst)

            batch_dict['sup'].append(np.vstack([sup_src, sup_dst]))
            batch_dict['que'].append(np.vstack([que_src, que_dst]))
            batch_dict['sup_src_meta'].append(sup_src_meta)
            batch_dict['sup_dst_meta'].append(sup_dst_meta)
            batch_dict['que_src_meta'].append(que_src_meta)
            batch_dict['que_dst_meta'].append(que_dst_meta)

            # batch_dict = {
            #     'sup' : torch.tensor((sup_src, sup_dst), dtype=torch.int64),
            #     'que' : torch.tensor((que_src, que_dst), dtype=torch.int64),
            #     'sup_src_meta' : sup_src_meta,  # [2, k, max_num_neighbor_src]
            #     'sup_dst_meta' : sup_dst_meta,  # [2, k, max_num_neighbor_src]
            #     'que_src_meta' : que_src_meta,
            #     # [2, num_candidates, max_num_neighbor_src], num_candidates can be very large!
            #     'que_dst_meta' : que_dst_meta,
            #     # [2, num_candidates, max_num_neighbor_src], num_candidates can be very large!
            # }
        batch_dict = {k : torch.LongTensor(v) for k, v in batch_dict.items()}
        return batch_dict


    def __getitem__(self, item):
        return self.iter_items[item]


    def __len__(self):
        return len(self.iter_items)


