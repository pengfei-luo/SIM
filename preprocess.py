
import os
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def entity_id_alignment(ent_encoder):
    with open('../data/NELL/ent2ids', 'r', encoding='utf-8') as in_f:
        _original_entity2id_dict = json.loads(in_f.readline())
    original_entity2id_dict = dict()
    encoder_classes = ent_encoder.classes_
    for _entity, entity_id in _original_entity2id_dict.items():
        try:
            _, _ent_type, _entity = _entity.split(':')
        except:
            continue
        else:
            _entity = ':'.join((_entity, _ent_type))
            original_entity2id_dict[_entity] = entity_id
    entity_name_of_encoder = ent_encoder.inverse_transform(range(len(encoder_classes)))
    entity_ids_of_previous_dict = [original_entity2id_dict[_ent_name] for _ent_name in entity_name_of_encoder]

    entity_embed = np.loadtxt('../data/NELL/entity2vec.TransE', dtype=np.float)
    entity_embed = entity_embed[entity_ids_of_previous_dict]
    entity_embed = np.concatenate([np.zeros((1, entity_embed.shape[1]), dtype=np.float), entity_embed],
                                  axis=0) # we leave id=0 for dummy embedding
    return entity_embed

def relation_id_alignment(rel_encoder):
    with open('../data/NELL/relation2ids', 'r', encoding='utf-8') as in_f:
        _original_rel2id_dict = json.loads(in_f.readline())
    original_rel2id_dict = dict()

    encoder_classes = rel_encoder.classes_
    for _rel, rel_id in _original_rel2id_dict.items() :
        try :
            _, _rel = _rel.split(':')
        except :
            continue
        else :
            original_rel2id_dict[_rel] = rel_id
    rel_name_of_encoder = rel_encoder.inverse_transform(range(len(encoder_classes)))
    rel_ids_of_previous_dict = [original_rel2id_dict[_rel_name] if _rel_name in _original_rel2id_dict.keys() else 0 for _rel_name in rel_name_of_encoder ]

    rel_embed = np.loadtxt('../data/NELL/relation2vec.TransE', dtype=np.float)
    rel_embed = rel_embed[rel_ids_of_previous_dict]
    rel_embed = np.concatenate([np.zeros((1, rel_embed.shape[1]), dtype=np.float), rel_embed],
                                  axis=0)  # we leave id=0 for dummy embedding
    return rel_embed
def format_nell(folder_path):
    schema = set()
    relation = set()
    entities = defaultdict(set)
    relationship = list()

    train_schema, dev_schema, test_schema = set(), set(), set()
    train_relationship, dev_relationship, test_relationship = list(), list(), list()

    ###### NOTE ######
    # we load original files, and do some simplification.
    # Each entity is like 'entity_name:entity_type' instead of 'concept:entity_type:entity_name'.
    # Each relation is like 'relation_name' instead of 'cpncept:relation_name'.
    ###### NOTE ######

    with open("{}/path_graph".format(folder_path), 'r', encoding='utf-8') as f:
        for line in f:
            head, rel, tail = line.strip().split('\t')
            try :
                _, head_ntype, head = head.split(':')
                _, rel = rel.split(':')
                _, tail_ntype, tail = tail.split(':')
                head = ':'.join((head, head_ntype))
                tail = ':'.join((tail, tail_ntype))
                # print(head, rel, tail)
            except:
                # print(head, rel, tail)
                continue
            else:
                schema.add(rel)
                entities[head_ntype].add(head)
                entities[tail_ntype].add(tail)
                relationship.append((head, rel, tail))
            relation.add(rel)
            relation.add(rel + '_inv')

    with open("{}/train_tasks.json".format(folder_path), 'r', encoding='utf-8') as f:
        train_tasks = json.loads(f.readline())
        for rel, triple_list in train_tasks.items():
            _, rel = rel.split(':')
            for head, _rel, tail in triple_list:
                _, head_ntype, head = head.split(':')
                _, _rel = _rel.split(':')
                _, tail_ntype, tail = tail.split(':')
                head = ':'.join((head, head_ntype))
                tail = ':'.join((tail, tail_ntype))

                entities[head_ntype].add(head)
                entities[tail_ntype].add(tail)
                train_schema.add(rel)
                train_relationship.append((head, rel, tail))
            relation.add(rel)

    with open("{}/dev_tasks.json".format(folder_path), 'r', encoding='utf-8') as f :
        dev_tasks = json.loads(f.readline())
        for rel, triple_list in dev_tasks.items() :
            _, rel = rel.split(':')
            for head, _rel, tail in triple_list :
                _, head_ntype, head = head.split(':')
                _, _rel = _rel.split(':')
                _, tail_ntype, tail = tail.split(':')
                head = ':'.join((head, head_ntype))
                tail = ':'.join((tail, tail_ntype))

                entities[head_ntype].add(head)
                entities[tail_ntype].add(tail)
                dev_schema.add(rel)
                dev_relationship.append((head, rel, tail))
            relation.add(rel)

    with open("{}/test_tasks.json".format(folder_path), 'r', encoding='utf-8') as f :
        test_tasks = json.loads(f.readline())
        for rel, triple_list in test_tasks.items() :
            _, rel = rel.split(':')
            for head, _rel, tail in triple_list :
                _, head_ntype, head = head.split(':')
                _, _rel = _rel.split(':')
                _, tail_ntype, tail = tail.split(':')
                head = ':'.join((head, head_ntype))
                tail = ':'.join((tail, tail_ntype))

                entities[head_ntype].add(head)
                entities[tail_ntype].add(tail)
                test_schema.add(rel)
                test_relationship.append((head, rel, tail))
            relation.add(rel)

    with open('{}/rel2candidates.json'.format(folder_path), 'r', encoding='utf-8') as f:
        candidates = json.loads(f.readline())
        candidates_formatted = dict()
        candidate_pair = list()
        for rel, candidates_list in candidates.items():
            _, rel = rel.split(':')
            if rel not in relation:
                continue
            candidates_list_formatted = []
            for candidate in candidates_list:
                _, candi_type, candi = candidate.split(':')
                candi = ':'.join((candi, candi_type))
                candidate_pair.append((rel, candi))
                candidates_list_formatted.append(candi)
            candidates_formatted[rel] = candidates_list_formatted

    with open('{}/e1rel_e2.json'.format(folder_path), 'r', encoding='utf-8') as f:
        e1rel_e2 = json.loads(f.readline())
        exclude = defaultdict(dict)
        exclude_triples = list()
        for e1rel, e2_list in e1rel_e2.items():
            sep_pos = e1rel.rfind('concept')
            head = e1rel[:sep_pos]
            rel = e1rel[sep_pos:]
            # assert head.startswith('concept') and rel.startswith('concept')
            _, head_type, head = head.split(':')
            _, rel = rel.split(':')
            head = ':'.join((head, head_type))
            exclude[rel][head] = list()
            if rel not in relation:
                continue
            for tail in e2_list:
                _, tail_type, tail = tail.split(':')
                tail = ':'.join((tail, tail_type))
                exclude[rel][head].append(tail)
                exclude_triples.append((head, rel, tail))
        exclude_df = pd.DataFrame(exclude_triples, columns=['head', 'rel', 'tail'])
        exclude_df.to_csv('../data/nell_formatted/exclude_df.csv', index=False)
    # convert all set object to list object for storage
    schema = list(schema)
    train_schema = list(train_schema)
    dev_schema = list(dev_schema)
    test_schema = list(test_schema)
    relation = list(relation)
    entities = {ntype: list(node_list) for ntype, node_list in entities.items()}

    # save formatted files
    if not os.path.exists('../data/nell_formatted'):
        os.makedirs('../data/nell_formatted', exist_ok=True)

    with open('../data/nell_formatted/schema.json', 'w', encoding='utf-8') as out_f:
        schema_all = schema + train_schema + dev_schema + test_schema
        # schema_rel = {rel: (head, rel, tail) for head, rel, tail in schema_all}
        schema_dict = {'path_graph': schema, 'train': train_schema, 'dev': dev_schema, 'test': test_schema,
                       'all': schema_all, 'rel': relation}
        out_f.write(json.dumps(schema_dict, ensure_ascii=False))

    with open('../data/nell_formatted/entities.json', 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(entities, ensure_ascii=False))

    with open('../data/nell_formatted/relationship.json', 'w', encoding='utf-8') as out_f:
        relation_dict = {'path_graph': relationship, 'train': train_relationship, 'dev': dev_relationship, 'test': test_relationship}
        out_f.write(json.dumps(relation_dict, ensure_ascii=False))

    with open('../data/nell_formatted/candidates.json', 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(candidates_formatted, ensure_ascii=False))

    with open('../data/nell_formatted/exclude.json', 'w', encoding='utf-8') as out_f:
        out_f.write(json.dumps(exclude, ensure_ascii=False))

    ###### NOTE ######
    # we convert evert thing into the format we need, i.e., everything into index or embedding.
    ###### NOTE ######
    entity_encoder = LabelEncoder()
    relation_encoder = LabelEncoder()
    all_relation = schema_dict['rel']
    all_entity = sum([nlist for ntype, nlist in entities.items()], [])
    entity_encoder.fit(all_entity)
    relation_encoder.fit(all_relation)

    schemas = {'train' : np.array(relation_encoder.transform(schema_dict['train']) + 1).tolist(),
               # we add one because we leave 0 as dummy idx
               'dev' : np.array(relation_encoder.transform(schema_dict['dev']) + 1).tolist(),
               'test' : np.array(relation_encoder.transform(schema_dict['test']) + 1).tolist()}
    df = pd.DataFrame(relation_dict['path_graph'], columns=['head', 'rel', 'tail'])
    inv_df = df.copy(True)
    inv_df[['head', 'tail']] = inv_df[['tail', 'head']]
    inv_df['rel'] = inv_df['rel'] + '_inv'
    df = pd.concat([df, inv_df], axis=0)
    df['head'] = entity_encoder.transform(df['head']) + 1  # we add one because we leave 0 as dummy idx
    df['rel'] = relation_encoder.transform(df['rel']) + 1
    df['tail'] = entity_encoder.transform(df['tail']) + 1
    # head_tail = (df['head'].tolist(), df['tail'].tolist(),)
    # rel = [{'rel' : x} for x in df['rel'].tolist()]

    rel2triple = dict()
    for mode in ['train', 'dev', 'test'] :
        mode_df = pd.DataFrame(relation_dict[mode], columns=['head', 'rel', 'tail'])
        mode_df['head'] = entity_encoder.transform(mode_df['head']) + 1
        mode_df['rel'] = relation_encoder.transform(mode_df['rel']) + 1
        mode_df['tail'] = entity_encoder.transform(mode_df['tail']) + 1
        for _schema in schemas[mode] :
            _schema_df = mode_df[mode_df['rel'] == _schema]
            rel2triple[_schema] = list(
                zip(_schema_df['head'].tolist(), [_schema] * len(_schema_df), _schema_df['tail'].tolist()))

    exclude = defaultdict(dict)
    exclude_df['head'] = entity_encoder.transform(exclude_df['head']) + 1
    exclude_df['rel'] = relation_encoder.transform(exclude_df['rel']) + 1
    exclude_df['tail'] = entity_encoder.transform(exclude_df['tail']) + 1

    for _row_idx, row in exclude_df.iterrows() :
        _head, _rel, _tail = row
        if _head not in exclude[_rel].keys() :
            exclude[_rel][_head] = list()
        exclude[_rel][_head].append(_tail)

    candidates = {
        relation_encoder.transform([rel])[0] + 1 : (np.array(entity_encoder.transform(candi_list)) + 1).tolist()
        for rel, candi_list in candidates_formatted.items() if rel in all_relation}

    entity_embed = entity_id_alignment(entity_encoder)
    relation_embed = relation_id_alignment(relation_encoder)

    g = defaultdict(dict)
    for _row_idx, row in df.iterrows() :
        _head, _rel, _tail = row
        if _head not in g.keys() :
            g[_head]['adj'] = list()
            g[_head]['rel'] = list()
        g[_head]['adj'].append(_tail)
        g[_head]['rel'].append(_rel)

    with open('../data/nell_formatted/g.bin', 'wb') as out_f:
        pickle.dump(g, out_f)
    with open('../data/nell_formatted/schemas.bin', 'wb') as out_f:
        pickle.dump(schemas, out_f)
    with open('../data/nell_formatted/rel2triple.bin', 'wb') as out_f:
        pickle.dump(rel2triple, out_f)
    with open('../data/nell_formatted/candidates.bin', 'wb') as out_f:
        pickle.dump(candidates, out_f)
    with open('../data/nell_formatted/exclude.bin', 'wb') as out_f:
        pickle.dump(exclude, out_f)
    np.save('../data/nell_formatted/ent_embed.npy', entity_embed)
    np.save('../data/nell_formatted/rel_embed.npy', relation_embed)


if __name__ == '__main__':

    format_nell('../data/NELL')
    pass




