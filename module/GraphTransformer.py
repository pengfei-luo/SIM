import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import apply_chunking_to_forward
from typing import Callable, Tuple
from torch import Tensor, device


class GraphTransformerEmbedding(nn.Module) :
    def __init__(self, config) :
        super(GraphTransformerEmbedding, self).__init__()
        self.token_type_embedding = nn.Embedding(2, config.hidden_size)
        self.max_adj = config.max_adj
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_embed):
        token_type_ids = torch.tensor([0] + [1] * self.max_adj, dtype=torch.int64, device=input_embed.device)
        token_type_embedding = self.token_type_embedding(token_type_ids)
        embeddings = input_embed + token_type_embedding
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings




class GTSelfAttention(nn.Module) :
    def __init__(self, config) :
        super(GTSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size") :
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x) :
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            # head_mask=None,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_value=None,
            output_attentions=False,
    ) :
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None :
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


class GTSelfOutput(nn.Module):
    def __init__(self, config):
        super(GTSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GTAttention(nn.Module):
    def __init__(self, config) :
        super().__init__()
        self.self = GTSelfAttention(config)
        self.output = GTSelfOutput(config)
        # self.pruned_heads = set()

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            # head_mask=None,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_value=None,
            output_attentions=False,
    ) :
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            # head_mask,
            # encoder_hidden_states,
            # encoder_attention_mask,
            # past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1 :]  # add attentions if we output them
        return outputs


class GTIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = F.gelu
        # if isinstance(config.hidden_act, str):
        #     self.intermediate_act_fn = ACT2FN[config.hidden_act]
        # else:
        #     self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class GTOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class GTLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = GTAttention(config)

        self.intermediate = GTIntermediate(config)
        self.output = GTOutput(config)

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            # head_mask=None,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_value=None,
            output_attentions=False,
    ) :
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            # head_mask,
            output_attentions=output_attentions,
            # past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output



class GTEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        # self.config = config
        self.layer = nn.ModuleList([GTLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(
            self,
            hidden_states,
            attention_mask=None,
            output_attentions=False,
            output_hidden_states=False,
    ) :
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        # all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for i, layer_module in enumerate(self.layer) :
            if output_hidden_states :
                all_hidden_states = all_hidden_states + (hidden_states,)
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                # layer_head_mask,
                # encoder_hidden_states,
                # encoder_attention_mask,
                # past_key_value,
                output_attentions,
            )

            hidden_states = layer_outputs[0]

            if output_attentions :
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                # if self.config.add_cross_attention :
                #     all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

            if output_hidden_states :
                all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [
                hidden_states,
                all_hidden_states,
                all_self_attentions,
                # all_cross_attentions,
            ]
            if v is not None
        )


class GTPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class GraphTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()

        # self.config = config

        # self.embeddings = BertEmbeddings(config)
        self.initializer_range = 0.02
        self.max_adj = config.max_adj
        self.output_attentions = False
        self.output_hidden_states = False
        self.embedding = GraphTransformerEmbedding(config)
        self.encoder = GTEncoder(config)
        self.pooler = GTPooler(config)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.initializer_range)
            # nn.init.xavier_normal_(module.weight.data)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def get_extended_attention_mask(self, attention_mask: Tensor, input_shape: Tuple[int], device: device) -> Tensor :
        """
        Makes broadcastable attention and causal masks so that future and masked tokens are ignored.

        Arguments:
            attention_mask (:obj:`torch.Tensor`):
                Mask with ones indicating tokens to attend to, zeros for tokens to ignore.
            input_shape (:obj:`Tuple[int]`):
                The shape of the input to the model.
            device: (:obj:`torch.device`):
                The device of the input to the model.

        Returns:
            :obj:`torch.Tensor` The extended attention mask, with a the same dtype as :obj:`attention_mask.dtype`.
        """
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3 :
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2 :
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = attention_mask[:, None, None, :]
        else :
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -1000000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        # extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -100000000.0
        return extended_attention_mask

    def forward(
            self,
            inputs_embeds=None,
            attention_mask=None,
            # encoder_hidden_states=None,
            # encoder_attention_mask=None,
            # past_key_values=None,
            # use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
    ) :
        # embedding = self.embedding(inputs_embeds)
        output_attentions = output_attentions if output_attentions is not None else self.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.output_hidden_states
        )
        # if inputs_embeds is not None:
        input_shape = inputs_embeds.size()
        batch_size, seq_length, embedding_dim = input_shape
        device = inputs_embeds.device
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        encoder_outputs = self.encoder(
            inputs_embeds,
            attention_mask=extended_attention_mask,
            # encoder_hidden_states=encoder_hidden_states,
            # encoder_attention_mask=encoder_extended_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        return (sequence_output, pooled_output) + encoder_outputs[1 :]

        # sequence_output = encoder_outputs[0]
        # src_pooled_output = self.pooler(sequence_output) if self.pooler is not None else None
        #
        # dst_pooled_output = self.pooler(sequence_output[:, (1 + self.max_adj) :]) if self.pooler is not None else None
        #
        # return src_pooled_output, dst_pooled_output