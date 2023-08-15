import collections
import logging
import os
import math
import time
import json
import types
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import utils
import scipy
import random
import sklearn
import argparse
import prettytable
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict

import wandb

from module.model import Model as Model
import _fewshot_loader as dh


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def warm_up_with_decay():
    if global_step <= config.warm_up_step:
        lr = config.lr * (global_step / config.warm_up_step)
    else:
        lr = config.lr * (1 - (global_step - config.warm_up_step) / (config.max_step - config.warm_up_step))
    # optimizer.param_groups[0]['lr'] = lr
    for param_group in optimizer.param_groups :
        param_group['lr'] = lr
    return lr


def metric(pred_score):
    rank = np.argsort(pred_score, kind='stable').tolist()[::-1].index(0) + 1
    hit_1 = 1 if rank == 1 else 0
    hit_5 = 1 if rank <= 5 else 0
    hit_10 = 1 if rank <= 10 else 0
    return hit_1, hit_5, hit_10, rank


def evaluate():
    hit_1, hit_5, hit_10, mrr = list(), list(), list(), list()
    with torch.no_grad():
        for batch_dict in dev_loader:
            batch_dict = {k : v.to(device) for k, v in batch_dict.items()}
            score = model(batch_dict)
            score = score.detach().cpu().numpy().flatten().tolist()
            _hit1, _hit_5, _hit_10, rank = metric(score)
            hit_1.append(_hit1)
            hit_5.append(_hit_5)
            hit_10.append(_hit_10)
            mrr.append(1.0 / rank)
        result_dict = {'hit@1' : round(np.mean(hit_1).item(), 5),
                       'hit@5' : round(np.mean(hit_5).item(), 5),
                       'hit@10' : round(np.mean(hit_10).item(), 5),
                       'MRR' : round(np.mean(mrr).item(), 5),
                       'step': global_step}
    logger.info('\n'
                '----------------------------------------------\n'
                '|STEP    |HIT@1   |HIT@5   |HIT@10  |MRR     |\n'
                '---------------------------------------------\n'
                '|{:7} |{:.5f} |{:.5f} |{:.5f} |{:.5f} |\n'.format(result_dict['step'],
                                                                   result_dict['hit@1'],
                                                                   result_dict['hit@5'],
                                                                   result_dict['hit@10'],
                                                                   result_dict['MRR']) +
                '----------------------------------------------\n'
                '|{:7} |{:.5f} |{:.5f} |{:.5f} |{:.5f} |\n'.format(best_result_dict['step'],
                                                                   best_result_dict['hit@1'],
                                                                   best_result_dict['hit@5'],
                                                                   best_result_dict['hit@10'],
                                                                   best_result_dict['MRR']) +
                '----------------------------------------------\n'
                '| CHANGE |{:+.4f} |{:+.4f} |{:+.4f} |{:+.4f} |\n'.format(result_dict['hit@1'] - best_result_dict['hit@1'],
                                                                          result_dict['hit@5'] - best_result_dict['hit@5'],
                                                                          result_dict['hit@10'] - best_result_dict['hit@10'],
                                                                          result_dict['MRR'] - best_result_dict['MRR']) +
                '----------------------------------------------'
                )
    return result_dict


def test():
    hit_1, hit_5, hit_10, mrr = list(), list(), list(), list()
    with torch.no_grad() :
        for batch_dict in test_loader :
            batch_dict = {k : v.to(device) for k, v in batch_dict.items()}
            score = model(batch_dict)
            score = score.detach().cpu().numpy().flatten().tolist()
            _hit1, _hit_5, _hit_10, rank = metric(score)
            hit_1.append(_hit1)
            hit_5.append(_hit_5)
            hit_10.append(_hit_10)
            mrr.append(1.0 / rank)
        result_dict = {'hit@1' : round(np.mean(hit_1).item(), 5),
                       'hit@5' : round(np.mean(hit_5).item(), 5),
                       'hit@10' : round(np.mean(hit_10).item(), 5),
                       'MRR' : round(np.mean(mrr).item(), 5)}
    logger.info('\n'
                '-------------------------------------\n'
                '|HIT@1   |HIT@5   |HIT@10  |MRR     |\n'
                '-------------------------------------\n'
                '|{:.5f} |{:.5f} |{:.5f} |{:.5f} | \n'.format(result_dict['hit@1'],
                                                            result_dict['hit@5'],
                                                            result_dict['hit@10'],
                                                            result_dict['MRR']) +
                '-------------------------------------\n')
    return result_dict

if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--hidden_size', type=int)
    parser.add_argument('--num_attention_heads', type=int)
    parser.add_argument('--num_hidden_layers', type=int)
    parser.add_argument('--intermediate_size', type=int)
    parser.add_argument('--layer_norm_eps', type=float)
    parser.add_argument('--hidden_dropout_prob', type=float)
    parser.add_argument('--attention_probs_dropout_prob', type=float)
    parser.add_argument('--chunk_size_feed_forward', type=float)
    parser.add_argument('--cnet_hidden_size', type=int)
    parser.add_argument('--cnet_num_attention_heads', type=int)
    parser.add_argument('--cnet_num_hidden_layers', type=int)
    parser.add_argument('--cnet_intermediate_size', type=int)
    parser.add_argument('--embedding_type', type=str)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--max_step', type=int)
    parser.add_argument('--max_adj', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--num_positive_samples', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--grad_clip', type=float)
    parser.add_argument('--finetune', type=int)
    parser.add_argument('--weight_decay', type=float)
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--llambda', type=int)
    parser.add_argument('--k', type=int)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--num_workers', type=int)
    parser.add_argument('--report_step', type=int)
    parser.add_argument('--eval_per_step', type=int)
    parser.add_argument('--warm_up_step', type=int)
    parser.add_argument('--early_stop_step', type=int)
    parser.add_argument('--ckpt_save_step', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--ckpt', type=int, default=0)

    args = parser.parse_args()
    args.finetune = True if args.finetune == 1 else False
    args.run_name = '{}_fold{}'.format(args.run_name, args.fold)

    if not args.test :
        wandb.init(project='YOUR_PROJECT', entity='YOUR_NAME', config=vars(args))
        wandb.run.name = args.run_name
        wandb.run.save()



    utils.make_directory()
    if not os.path.exists('../user_data/model/{}'.format(args.run_name)) :
        os.makedirs('../user_data/model/{}'.format(args.run_name))

    logger = utils.init_logger('../user_data/logs/{}{}.log'.format(args.run_name, '' if not args.test else '_test_{}'.format(args.ckpt)))
    pt = prettytable.PrettyTable()
    pt.field_names = ['params', 'values']
    for k, v in vars(args).items() :
        pt.add_row([k, v])
    logger.info('\n' + str(pt))

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    logger.info('Current device is {}'.format(device))

    set_seed(args.seed)
    config = args

    logger.info('Loading data & building graph ...')
    data_set = dh.FewShotLoader(config)
    print('.', end='')
    train_data_set = dh.FSDataSet(config , data_set, mode='train')
    train_loader = DataLoader(train_data_set, config.batch_size, False, collate_fn=train_data_set.collate, num_workers=args.num_workers, pin_memory=False)
    print('.', end='')
    dev_data_set = dh.FSDataSet(config, data_set, mode='dev')
    dev_loader = DataLoader(dev_data_set, 1, False, collate_fn=dev_data_set.collate_eval, num_workers=args.num_workers, pin_memory=False)
    print('.', end='')
    test_data_set = dh.FSDataSet(config, data_set, mode='test')
    test_loader = DataLoader(test_data_set, 1, False, collate_fn=test_data_set.collate_eval, num_workers=args.num_workers, pin_memory=False)
    print('.')


    logger.info('Initializing model ...')

    if not args.test :
        with open('../user_data/model/{}/config.json'.format(args.run_name), 'w', encoding='utf-8') as f:
            f.write(json.dumps(vars(args), ensure_ascii=False, indent=4))

        config.rel_embed = train_data_set.data.rel_embed
        config.ent_embed = train_data_set.data.ent_embed


        model = Model(config).to(device)


        # net_params = filter(lambda p: id(p) not in embedding_params_id, model.parameters())
        optimizer_grouped_parameters = [
            {'params' : model.parameters()},
        ]

        global_step = 0
        loss_deque = collections.deque([], maxlen=args.report_step)
        proto_loss_deque = collections.deque([], maxlen=args.report_step)
        task_loss_deque = collections.deque([], maxlen=args.report_step)
        optimizer = torch.optim.AdamW(optimizer_grouped_parameters, args.lr, weight_decay=args.weight_decay)
        grad_parameter = filter(lambda p: p.requires_grad, model.parameters())
        lr = warm_up_with_decay()
        loss_func = nn.MarginRankingLoss(args.llambda)
        model.train()

        # todo: This should be comment if the code will be published
        wandb.watch(model)
        best_result_dict = {'hit@1':0, 'hit@5':0, 'hit@10':0, 'MRR':0, 'step':0}
        best_model = None

        # Training Loop
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}


            positive_score, negative_score = model(batch)
            loss = loss_func(positive_score, negative_score, torch.tensor([1], device=positive_score.device))
            # proto_loss = 0.05 * F.relu(10 - proto_loss).mean()
            # task_loss = 0.1 * F.relu(15 - task_loss).mean()

            # loss = margin_loss
            # _loss /= args.batch_size
            loss.backward()
            # if loss is None:
            #     loss = _loss
            # else:
            #     loss += _loss

            loss_deque.append(loss.item())
            # proto_loss_deque.append(proto_loss.item())
            # task_loss_deque.append(task_loss.item())
            nn.utils.clip_grad_norm_(grad_parameter, args.grad_clip)
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
            lr = warm_up_with_decay()


            if global_step % args.report_step == 0:
                info_dict = {'loss': round(np.mean(loss_deque).item(), 4),
                             # 'proto_loss': round(np.mean(proto_loss_deque).item(), 4),
                             # 'task_loss': round(np.mean(task_loss_deque).item(), 4),
                             'step': global_step,
                             'lr': round(optimizer.param_groups[0]['lr'], 7)}
                logger.info(info_dict)

                # todo: This should be comment if the code will be published
                wandb.log(info_dict, step=global_step)


            # t.set_postfix(loss='{:.5f}'.format(np.mean(loss_deque).item()),
            #               lr='{:.7f}'.format(optimizer.param_groups[0]['lr']))
            # t.update()

            if global_step % args.eval_per_step == 0:
                logger.info('Evaluate at step {}'.format(global_step))
                model.eval()
                result_dict = evaluate()

                # todo: This should be comment if the code will be published
                wandb.log({'val/'+k: v for k, v in result_dict.items()}, step=global_step)

                if result_dict['MRR'] >= best_result_dict['MRR']:
                    best_result_dict = result_dict
                    best_model = deepcopy(model)
                    torch.save(model.state_dict(), '../user_data/model/{}/ckpt_best.pt'.format(args.run_name))
                    logging.info('Current best MRR is {} at step {}. Best model saved.\n'.format(
                        best_result_dict['MRR'],
                        best_result_dict['step']))
                else:
                    logging.info('Current best MRR is {} at step {}. The model does not improve for {} steps.\n'.format(
                        best_result_dict['MRR'],
                        best_result_dict['step'],
                        global_step - best_result_dict['step']))

                if args.ckpt_save_step != 0 and global_step % args.ckpt_save_step == 0:
                    torch.save(model.state_dict(), '../user_data/model/{}/ckpt_{}.pt'.format(args.run_name, global_step))

                if global_step - best_result_dict['step'] >= args.early_stop_step:
                    logger.info('Early stop!\n')
                    break
                model.train()

        if global_step != args.max_step:
            logging.info('Testing on the latest model')
            model.eval()
            test_dict = test()

        # if best_result_dict['step'] != args.max_step:
        logging.info('Testing on the best model')
        model = best_model
        model.eval()
        test_dict = test()
        # else:
        #     logging.info('Best model has been tested!')

        # todo: This should be comment if the code will be published
        # wandb.log({'test/'+k: v for k, v in test_dict.items()})
        for k, v in test_dict.items():
            wandb.run.summary[k] = v

    else:
        if args.ckpt == 0 :
            state_dict = torch.load('../user_data/model/{}/ckpt_best.pt'.format(args.run_name))
            logger.info('Testing on model [{}]'.format('../user_data/model/{}/ckpt_best.pt'.format(args.run_name)))
        else:
            state_dict = torch.load('../user_data/model/{}/ckpt_{}.pt'.format(args.run_name, args.ckpt))
            logger.info('Testing on model [{}]'.format('../user_data/model/{}/ckpt_{}.pt'.format(args.run_name, args.ckpt)))

        if os.path.exists('../user_data/model/{}/config.json'.format(args.run_name)):
            with open('../user_data/model/{}/config.json'.format(args.run_name), 'r', encoding='utf-8') as f:
                config = json.loads(f.read())
        config = types.SimpleNamespace(**config)
        config.rel_embed = train_data_set.data.rel_embed
        config.ent_embed = train_data_set.data.ent_embed

        model = Model(config)
        model.load_state_dict(state_dict)
        model = model.to(device)

        model.eval()
        test_dict = test()











