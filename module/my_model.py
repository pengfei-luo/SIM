import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from module.GraphTransformer import GraphTransformer


class MyEntityEncoder(nn.Module) :
    def __init__(self, config) :
        super(MyEntityEncoder, self).__init__()
        # self.gat = GAT(config.embed_dim, config.embed_dim, config.gat_num_heads, config.dropout)
        self.fc = nn.Linear(config.embed_dim, config.embed_dim)
        self.embedding_dropout = nn.Dropout(config.dropout)
        # self.cls = nn.Parameter(torch.rand(size=(1, config.hidden_size)), requires_grad=True)

        self.GraphTransformer = GraphTransformer(config)
        self.dense = nn.Linear(config.embed_dim * 2, config.hidden_size)
        self.activation = nn.Tanh()

        self.hidden_out_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.hidden_out_act = nn.Tanh()

        self.out_dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.out_act = nn.Tanh()

        # self.cls.data.normal_(mean=0.0, std=0.02)

    def forward(self, src, src_rel, src_adj, src_mask,
                dst, dst_rel, dst_adj, dst_mask) :
        """

        :param src:         [batch_size * num_nodes, embed_dim]
        :param src_rel:     [batch_size * num_nodes, max_adj, embed_dim]
        :param src_adj:     [batch_size * num_nodes, max_adj, embed_dim]
        :param src_mask:    [batch_size * num_nodes, max_adj]
        :param dst:
        :param dst_rel:
        :param dst_adj:
        :param dst_mask:
        :return:
                attn_src : [batch_size * num_nodes, hidden_size]
                attn_dst : [batch_size * num_nodes, hidden_size]
        """
        # [n_node, 1+max_adj, embed_dim]

        _n_node = src.shape[0]
        _max_adj = src_rel.shape[1]

        src_relative_rel = self.fc(dst - src)
        src_relative_rel_pad = src_relative_rel.unsqueeze(dim=1)
        src_rel_with_neighbor = torch.cat([src_relative_rel_pad, src_rel], dim=1)

        src_pad = src.unsqueeze(dim=1)
        src_with_neighbor = torch.cat([src_pad, src_adj], dim=1)
        src_with_neighbor = torch.cat([src_with_neighbor, src_rel_with_neighbor], dim=-1)
        src_attn_mask = torch.cat([torch.ones(src.shape[0], 1, device=src.device), src_mask], dim=1)

        src_with_neighbor = self.activation(self.dense(src_with_neighbor))

        src_hidden, src_output = self.GraphTransformer(inputs_embeds=src_with_neighbor,
                                                       attention_mask=src_attn_mask,
                                                       output_attentions=False,
                                                       output_hidden_states=False)[: 2]
        src_hidden = torch.sum(src_hidden * src_attn_mask.unsqueeze(dim=-1), dim=1) / torch.sum(src_attn_mask, dim=-1, keepdim=True)
        src_output = self.out_act(self.out_dense(src_output + src_hidden))

        dst_relative_rel = self.fc(src - dst)
        dst_relative_rel_pad = dst_relative_rel.unsqueeze(dim=1)
        dst_rel_with_neighbor = torch.cat([dst_relative_rel_pad, dst_rel], dim=1)

        dst_pad = dst.unsqueeze(dim=1)
        dst_with_neighbor = torch.cat([dst_pad, dst_adj], dim=1)
        dst_with_neighbor = torch.cat([dst_with_neighbor, dst_rel_with_neighbor], dim=-1)
        dst_attn_mask = torch.cat([torch.ones(dst.shape[0], 1, device=dst.device), dst_mask], dim=1)

        dst_with_neighbor = self.activation(self.dense(dst_with_neighbor))

        dst_hidden, dst_output = self.GraphTransformer(inputs_embeds=dst_with_neighbor,
                                                       attention_mask=dst_attn_mask,
                                                       output_attentions=False,
                                                       output_hidden_states=False)[: 2]
        dst_hidden = torch.sum(dst_hidden * dst_attn_mask.unsqueeze(dim=-1), dim=1) / torch.sum(dst_attn_mask, dim=-1, keepdim=True)
        dst_output = self.out_act(self.out_dense(dst_output + dst_hidden))


        # src_dst_with_neighbor = torch.cat([src_with_neighbor, dst_with_neighbor], dim=1) # [n_node, 1+max_adj+1+max_adj, embed_dim]
        # src_dst_mask = torch.cat([src_attn_mask, dst_attn_mask], dim=1) # [n_node, 1+max_adj+1+max_adj]

        # print(src_dst_with_neighbor.shape)
        # print(src_dst_mask.shape)

        # hidden_state = self.GraphTransformer(inputs_embeds=src_dst_with_neighbor,
        #                                    attention_mask=src_dst_mask,
        #                                    output_attentions=False,
        #                                    output_hidden_states=False)[0]
        # hidden_state = self.hidden_out_act(self.hidden_out_dense(hidden_state))
        # src_output = hidden_state[:, 0]
        # dst_output = hidden_state[:, 1+_max_adj]
        # src_neighbor_hidden = torch.sum(hidden_state[:, :1+_max_adj] * src_attn_mask.unsqueeze(dim=-1), dim=1) / torch.sum(src_attn_mask, dim=-1, keepdim=True)
        # dst_neighbor_hidden = torch.sum(hidden_state[:, 1+_max_adj:] * dst_attn_mask.unsqueeze(dim=-1), dim=1) / torch.sum(dst_attn_mask, dim=-1, keepdim=True)
        # src_output = self.out_act(self.out_dense(src_output + src_neighbor_hidden))
        # dst_output = self.out_act(self.out_dense(dst_output + dst_neighbor_hidden))


        return src_output, dst_output


class CompareNet(nn.Module) :
    def __init__(self, config) :
        super(CompareNet, self).__init__()
        _config = copy.deepcopy(config)
        _config.hidden_size = _config.cnet_hidden_size
        _config.intermediate_size = _config.cnet_intermediate_size
        _config.num_attention_heads = _config.cnet_num_attention_heads
        _config.num_hidden_layers = _config.cnet_num_hidden_layers

        # self.sep = nn.Parameter(torch.rand(size=(1, _config.hidden_size)), requires_grad=True)
        # self.cls = nn.Parameter(torch.rand(size=(1, _config.hidden_size)), requires_grad=True)
        # self.proto = nn.Parameter(torch.rand(size=(1, _config.hidden_size)), requires_grad=True)
        self.GraphTransformer = GraphTransformer(_config)
        self.fc = nn.Linear(config.hidden_size * 2, config.cnet_hidden_size, bias=True)
        self.activation = nn.Tanh()

        # self.fc = nn.Linear(_config.hidden_size * 2, _config.hidden_size * 2)
        # self.sep = nn.Parameter(torch.FloatTensor(torch.randn(size=(1, self.hidden_size))), requires_grad=True)

        # self.Prototype = SoftSelectPrototype(200)

        self.query_out_dense = nn.Linear(config.cnet_hidden_size, config.cnet_hidden_size)
        self.query_out_act = nn.Tanh()

    def forward(self, support, query) :
        """
        #
        :param support:     [batch_size, k, hidden_size * 2]
        :param query:       [batch_size, num_nodes,  hidden_size * 2]
        :return:
        """
        # support = self.activation(self.fc(support))
        # query = self.activation(self.fc(query))
        _k = support.shape[1]
        _batch_size, _num_nodes, _hidden_size = query.shape


        query = query.unsqueeze(dim=2)  # [batch_size, num_nodes, 1, hidden_size * 2]
        support = support.unsqueeze(dim=1).tile(1, query.shape[1], 1, 1)  # [batch_size, num_nodes, k, hidden_size * 2]
        query = torch.cat([query, support], dim=2)  # [batch_size, num_nodes, 1 + k, hidden_size * 2]
        query = query.reshape(-1, query.shape[-2], query.shape[-1])  # [batch_size * num_nodes, 1 + k, hidden_size * 2]
        query = self.activation(self.fc(query))

        attn_mask = torch.ones(size=query.shape[:2], device=query.device)

        # Ablation
        output = self.GraphTransformer(inputs_embeds=query, attention_mask=attn_mask,
                                       output_attentions=False, output_hidden_states=False)[0]


        support_hidden = output[:, 1 :, :]  # [batch_size * num_nodes, k, hidden_size * 2]
        query_hidden = output[:, 0, :].unsqueeze(dim=1).tile(1, support_hidden.shape[1],
                                                             1)  # [batch_size * num_nodes, k, hidden_size * 2]
        scalar = support_hidden.shape[-1] ** -0.5
        attn_score = torch.sum(query_hidden * support_hidden, dim=2) * scalar
        attn_probs = F.softmax(attn_score, dim=1).unsqueeze(dim=1)  # [batch_size * num_nodes, 1, k]
        _attn_probs_out = attn_probs.reshape(_batch_size, _num_nodes, _k)
        print(_attn_probs_out.shape, _attn_probs_out[:, 0, :])
        proto = torch.bmm(attn_probs, support_hidden).squeeze()  # [batch_size * num_nodes, hidden_size * 2]


        score = torch.sum(output[:, 0, :] * proto, dim=-1)

        # attn_mask = torch.ones(size=support.shape[:2], device=support.device)
        #
        # output = self.GraphTransformer(inputs_embeds=support, attention_mask=attn_mask,
        #                                output_attentions=False, output_hidden_states=False)[1]
        #
        # support_hidden = output.unsqueeze(dim=1).tile(1, _num_nodes, 1, 1) # [bs, 1, k, hidden * 2]
        # query_hidden = query.unsqueeze(dim=2)    # [bs, num_nodes, 1, hidden * 2]
        # scalar = support_hidden.shape[-1] ** -0.5
        # attn_score = torch.sum(query_hidden * support_hidden, dim=-1) * scalar # [bs, num_nodes, k]
        # attn_probs = F.softmax(attn_score, dim=1).unsqueeze(dim=2)  # [bs, num_nodes, 1, k]
        # proto = torch.matmul(attn_probs, support_hidden).squeeze()  # [bs, num_nodes, hidden ]
        #
        # query_hidden = self.query_out_act(self.query_out_dense(query_hidden))
        # score = torch.sum(query_hidden.squeeze() * proto, dim=-1)


        return score
