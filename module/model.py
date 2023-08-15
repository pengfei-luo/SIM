import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

from module.modules import *
from module.my_model import MyEntityEncoder, CompareNet

class MLP(nn.Module):
    def __init__(self, in_feat, hidden_feat, out_feat, n_layer, dropout=0.1, activation=nn.LeakyReLU):
        super(MLP, self).__init__()
        self.mlp = nn.ModuleList()
        self.activation = activation
        remain_layer = n_layer

        self.mlp.append(nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_feat, out_feat if remain_layer == 1 else hidden_feat),
            self.activation(),
        ))
        remain_layer -= 1

        if remain_layer:
            for layer in range(0, n_layer-1):
                self.mlp.append(nn.Sequential(
                    nn.Dropout(dropout),
                    nn.Linear(hidden_feat, hidden_feat),
                    self.activation()
                ))

            self.mlp.append(nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(hidden_feat, out_feat),
                self.activation(),
            ))
        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        for _layer in self.mlp:
            for _module in _layer:
                if isinstance(_module, nn.Linear):
                    nn.init.xavier_normal_(_module.weight, gain=gain)
                    if _module.bias is not None:
                        nn.init.constant_(_module.bias, 0)



    def forward(self, output):
        for layer in self.mlp:
            output = layer(output)
        return output


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.ent_embed = nn.Parameter(torch.from_numpy(config.ent_embed).float(), requires_grad=config.finetune)
        self.rel_embed = nn.Parameter(torch.from_numpy(config.rel_embed).float(), requires_grad=config.finetune)

        self.ent_encoder = MyEntityEncoder(config)
        self.drop_out = nn.Dropout(config.dropout)

        self.compare_net = CompareNet(config)


    def GNN_aggregate(self, src, src_rel, src_adj, dst, dst_rel, dst_adj):
        """

        :param src:         [batch_size, num_nodes]
        :param src_rel:     [batch_size, num_nodes, max_adj]
        :param src_adj:     [batch_size, num_nodes, max_adj]
        :param dst:         [batch_size, num_nodes]
        :param dst_rel:     [batch_size, num_nodes, max_adj]
        :param dst_adj:     [batch_size, num_nodes, max_adj]
        :return:
                attn_src :  [batch_size, num_nodes, hidden_size]
                attn_dst :  [batch_size, num_nodes, hidden_size]
        """
        batch_size = src.shape[0]
        num_nodes = src.shape[1]
        max_adj = src_rel.shape[-1]

        src = src.reshape(-1, )
        src_rel = src_rel.reshape(-1, max_adj)
        src_adj = src_adj.reshape(-1, max_adj)
        dst = dst.reshape(-1, )
        dst_rel = dst_rel.reshape(-1, max_adj)
        dst_adj = dst_adj.reshape(-1, max_adj)

        src = self.drop_out(self.ent_embed[src])  # [batch_size * num_nodes, embed_dim]
        dst = self.drop_out(self.ent_embed[dst])  # [batch_size * num_nodes, embed_dim]


        src_mask = (src_rel != 0).int()           # [batch_size * num_nodes, max_adj]
        dst_mask = (dst_rel != 0).int()
        src_rel = self.drop_out(self.rel_embed[src_rel])    # [batch_size * num_nodes, max_adj, embed_dim]
        src_adj = self.drop_out(self.ent_embed[src_adj])
        dst_rel = self.drop_out(self.rel_embed[dst_rel])
        dst_adj = self.drop_out(self.ent_embed[dst_adj])

        # src_rel = src_adj - src.unsqueeze(dim=1)
        # dst_rel = dst_adj - dst.unsqueeze(dim=1)

        attn_src, attn_dst = self.ent_encoder(src, src_rel, src_adj, src_mask,
                                              dst, dst_rel, dst_adj, dst_mask)

        attn_src = attn_src.reshape(-1, num_nodes, attn_src.shape[-1]) # [batch_size, num_nodes, hidden_size]
        attn_dst = attn_dst.reshape(-1, num_nodes, attn_dst.shape[-1]) # [batch_size, num_nodes, hidden_size]
        return attn_src, attn_dst




    def forward(self, batch_dict):
        """

        :param batch_dict:  if eval then the keys are:
               sup : [batch_size, 2, k]
               pos : [batch_size, 2, k]
               neg : [batch_size, 2, k]
               sup_src_meta : [batch_size, 2, max_adj]
               sup_dst_meta : [batch_size, 2, max_adj]
               neg_src_mata : [batch_size, 2, max_adj]
               neg_dst_meta : [batch_size, 2, max_adj]
               pos_src_meta : [batch_size, 2, max_adj]
               pos_dst_meta : [batch_size, 2, max_adj]
        :return:
        """
        is_eval = 'que' in batch_dict.keys()

        if is_eval:
            sup_src = batch_dict['sup'][:, 0, :]
            sup_dst = batch_dict['sup'][:, 1, :]
            sup_src_rel = batch_dict['sup_src_meta'][:, 0, :]
            sup_src_adj = batch_dict['sup_src_meta'][:, 1, :]
            sup_dst_rel = batch_dict['sup_dst_meta'][:, 0, :]
            sup_dst_adj = batch_dict['sup_dst_meta'][:, 1, :]
            sup_src, sup_dst = self.GNN_aggregate(sup_src, sup_src_rel, sup_src_adj,
                                                  sup_dst, sup_dst_rel, sup_dst_adj)


            que_src = batch_dict['que'][:, 0, :]
            que_dst = batch_dict['que'][:, 1, :]
            que_src_rel = batch_dict['que_src_meta'][:, 0, :]
            que_src_adj = batch_dict['que_src_meta'][:, 1, :]
            que_dst_rel = batch_dict['que_dst_meta'][:, 0, :]
            que_dst_adj = batch_dict['que_dst_meta'][:, 1, :]

            que_src, que_dst = self.GNN_aggregate(que_src, que_src_rel, que_src_adj,
                                                  que_dst, que_dst_rel, que_dst_adj)



            support = torch.cat([sup_src, sup_dst], dim=-1) # [batch_size, k, hidden_size * 2]
            query = torch.cat([que_src, que_dst], dim=-1)   # [batch_size, num_nodes, hidden_size * 2]

            score = self.compare_net(support, query)

            return score
        else:
            sup_src = batch_dict['sup'][:, 0, :]
            sup_dst = batch_dict['sup'][:, 1, :]
            sup_src_rel = batch_dict['sup_src_meta'][:, 0, :]
            sup_src_adj = batch_dict['sup_src_meta'][:, 1, :]
            sup_dst_rel = batch_dict['sup_dst_meta'][:, 0, :]
            sup_dst_adj = batch_dict['sup_dst_meta'][:, 1, :]
            sup_src, sup_dst = self.GNN_aggregate(sup_src, sup_src_rel, sup_src_adj,
                                                  sup_dst, sup_dst_rel, sup_dst_adj)


            pos_src = batch_dict['pos'][:, 0, :]
            pos_dst = batch_dict['pos'][:, 1, :]
            pos_src_rel = batch_dict['pos_src_meta'][:, 0, :]
            pos_src_adj = batch_dict['pos_src_meta'][:, 1, :]
            pos_dst_rel = batch_dict['pos_dst_meta'][:, 0, :]
            pos_dst_adj = batch_dict['pos_dst_meta'][:, 1, :]

            pos_src, pos_dst = self.GNN_aggregate(pos_src, pos_src_rel, pos_src_adj,
                                                  pos_dst, pos_dst_rel, pos_dst_adj)


            neg_src = batch_dict['neg'][:, 0, :]
            neg_dst = batch_dict['neg'][:, 1, :]
            neg_src_rel = batch_dict['neg_src_meta'][:, 0, :]
            neg_src_adj = batch_dict['neg_src_meta'][:, 1, :]
            neg_dst_rel = batch_dict['neg_dst_meta'][:, 0, :]
            neg_dst_adj = batch_dict['neg_dst_meta'][:, 1, :]

            neg_src, neg_dst = self.GNN_aggregate(neg_src, neg_src_rel, neg_src_adj,
                                                  neg_dst, neg_dst_rel, neg_dst_adj)

            # neg_sub_graph = self.GNN_aggregate(neg_src, neg_src_rel, neg_src_adj,
            #                                       neg_dst, neg_dst_rel, neg_dst_adj)

            support = torch.cat([sup_src, sup_dst], dim=-1)
            positive = torch.cat((pos_src, pos_dst), dim=-1)
            negative = torch.cat((neg_src, neg_dst), dim=-1)


            positive_score = self.compare_net(support, positive)
            negative_score = self.compare_net(support, negative)

            # prototype loss
            # prototype_score = torch.mean(positive_batch_proto * negative_batch_proto, dim=-1)

            # task loss
            # [batch_size, hidden_size] * [hidden_size, batch_size] =
            # task_score = torch.matmul(positive_batch_proto, positive_batch_proto.T)
            # task_mask = torch.eye(task_score.shape[1], device=task_score.device).bool()
            # task_score = task_score.masked_fill(task_mask, 0.).flatten()


            return positive_score, negative_score
