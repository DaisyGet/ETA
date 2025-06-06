import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import scipy.stats as stats
from pytorch_transformers import modeling_bert

# from model.gate import HighwayGateLayer
from utils import constant


def truncated_normal(shape, mean=0.0, stddev=1.0, dtype=np.float32):

    lower = -2 * stddev + mean
    upper = 2 * stddev + mean
    X = stats.truncnorm((lower - mean) / stddev,
                        (upper - mean) / stddev,
                        loc=mean,
                        scale=stddev)
    values = X.rvs(size=shape)
    return torch.from_numpy(values.astype(dtype))


def pool(h, mask, type='max'):
    if type == 'max':
        h = h.masked_fill(mask, -constant.INFINITY_NUMBER)
        return torch.max(h, 1)[0]
    elif type == 'avg':
        h = h.masked_fill(mask, 0)
        return h.sum(1) / (mask.size(1) - mask.float().sum(1))
    else:
        h = h.masked_fill(mask, 0)
        return h.sum(1)


class HighwayGateLayer(nn.Module):
    def __init__(self, in_out_size, bias=True):
        super(HighwayGateLayer, self).__init__()
        self.transform = nn.Linear(in_out_size, in_out_size, bias=bias)

    def forward(self, x, y):
        out_transform = torch.sigmoid(self.transform(x))
        return out_transform * x + (1 - out_transform) * y


class ScaledEmbedding(nn.Embedding):
    """
    Embedding layer that initialises its values
    to using a truncated normal variable scaled by the inverse
    of the embedding dimension.
    """

    def reset_parameters(self):
        """
        Initialize parameters using Truncated Normal Initializer (default in Tensorflow)
        """
        self.weight.data = truncated_normal(shape=(self.num_embeddings,
                                                   self.embedding_dim),
                                            stddev=1.0 / math.sqrt(self.embedding_dim))
        if self.padding_idx is not None:
            self.weight.data[self.padding_idx].fill_(0)


class RelationalBertSelfAttention(nn.Module):
    def __init__(self, config):
        super(RelationalBertSelfAttention, self).__init__()
        if config.hidden_size % config.num_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_heads))
        self.config = config
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_heads
        self.attention_head_size = int(config.hidden_size / config.num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size,
                               self.all_head_size)
        self.key = nn.Linear(config.hidden_size,
                             self.all_head_size)
        self.value = nn.Linear(config.hidden_size,
                               self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, dep_rel_matrix=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        rel_attention_scores = 0
        if self.config.syntax['use_dep_rel']:
            rel_attention_scores = query_layer[:, :, :, None, :] * dep_rel_matrix[:, None, :, :, :]
            rel_attention_scores = torch.sum(rel_attention_scores, -1)

        attention_scores = (attention_scores + rel_attention_scores) / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        context_layer = torch.matmul(attention_probs,
                                     value_layer)

        if self.config.syntax['use_dep_rel']:
            val_edge = attention_probs[:, :, :, :, None] * dep_rel_matrix[:, None, :, :, :]
            context_layer = context_layer + torch.sum(val_edge, -2)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


class FeedForwardLayer(nn.Module):
    def __init__(self, config, activation_dropout=0.1):
        super(FeedForwardLayer, self).__init__()
        self.W_1 = nn.Linear(config.hidden_size,
                             config.intermediate_size)
        self.act = modeling_bert.ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(activation_dropout)
        self.W_2 = nn.Linear(config.intermediate_size,
                             config.hidden_size)

    def forward(self, e):
        e = self.dropout(self.act(self.W_1(e)))
        e = self.W_2(e)
        return e


class GATEncoderLayer(nn.Module):
    def __init__(self, config):
        super(GATEncoderLayer, self).__init__()
        self.config = config
        self.syntax_attention = RelationalBertSelfAttention(config)
        self.finishing_linear_layer = nn.Linear(config.hidden_size,
                                                config.hidden_size)
        self.dropout1 = nn.Dropout(config.syntax['layer_prepostprocess_dropout'])
        self.ln_2 = nn.LayerNorm(config.hidden_size,
                                 eps=1e-6)
        if config.syntax['tf_enc_use_ffn']:
            self.feed_forward = FeedForwardLayer(config,
                                                 config.syntax['gelu_dropout'])
            self.dropout2 = nn.Dropout(config.syntax['layer_prepostprocess_dropout'])
            self.ln_3 = nn.LayerNorm(config.hidden_size,
                                     eps=1e-6)
        if self.config.syntax['tf_enc_gated_connection']:
            self.gate1 = eval(config.syntax['tf_enc_gate'])(config.hidden_size)
            if config.syntax['tf_enc_use_ffn']:
                self.gate2 = eval(config.syntax['tf_enc_gate'])(config.hidden_size)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        sub = self.finishing_linear_layer(self.syntax_attention(self.ln_2(e),
                                                                attention_mask,
                                                                dep_rel_matrix)[0])
        sub = self.dropout1(sub)
        if self.config.syntax['tf_enc_act_at_skip_connection']:
            sub = F.gelu(sub)
        if self.config.syntax['tf_enc_gated_connection']:
            e = self.gate1(e, sub)
        else:
            e = e + sub

        if self.config.syntax['tf_enc_use_ffn']:
            sub = self.feed_forward(self.ln_3(e))
            sub = self.dropout2(sub)
            if self.config.syntax['tf_enc_act_at_skip_connection']:
                sub = F.gelu(sub)
            if self.config.syntax['tf_enc_gated_connection']:
                e = self.gate2(e, sub)
            else:
                e = e + sub
        return e


class GATEncoder(nn.Module):
    def __init__(self, config):
        super(GATEncoder, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(config.syntax['num_layers']):
            layer = GATEncoderLayer(config)
            self.layers.append(layer)
        self.ln = nn.LayerNorm(config.hidden_size,
                               eps=1e-6)

    def forward(self, e, attention_mask, dep_rel_matrix=None):
        for layer in self.layers:
            e = layer(e,
                      attention_mask,
                      dep_rel_matrix)
        e = self.ln(e)
        return e


class GNNRelationModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # create embedding layers
        if config.model_type == "late_fusion":
            in_dim = config.hidden_size
        else: # for pre-trained models such as "joint_fusion", or for randomly initialized ones like "gat"
            self.emb = ScaledEmbedding(config.vocab_size,
                                       config.syntax['emb_size'],
                                       padding_idx=constant.PAD_ID)
            in_dim = config.syntax['emb_size']
            self.input_dropout = nn.Dropout(config.syntax['input_dropout'])

            if config.syntax['embed_position']:
                self.embed_pos = ScaledEmbedding(config.max_position_embeddings,
                                                 config.syntax['emb_size'])

        if config.syntax['use_dep_rel']:
            self.rel_emb = ScaledEmbedding(len(constant.DEPREL_TO_ID),
                                           int(config.hidden_size / config.num_attention_heads),
                                           padding_idx=constant.DEPREL_TO_ID['<PAD>'])

        # Graph Attention layer
        if config.use_syntax:
            self.syntax_encoder = GATEncoder(config)
            # self.syntax_encoder = eval(config.syntax['syntax_encoder'])(config)

            if config.model_type == 'late_fusion' and config.syntax['late_fusion_gated_connection']:
                self.gate = HighwayGateLayer(config.syntax['hidden_size'])

            out_dim = config.syntax['hidden_size']
        else:
            out_dim = in_dim

        if config.syntax['use_subj_obj']:
            out_dim *= 3

        # output MLP layers
        layers = [nn.Linear(out_dim,
                            config.syntax['hidden_size']),
                  nn.Tanh()]
        for _ in range(config.syntax['mlp_layers'] - 1):
            layers += [nn.Linear(config.syntax['hidden_size'],
                                 config.syntax['hidden_size']),
                       nn.Tanh()]
        self.out_mlp = nn.Sequential(*layers)
        self.pool_mask, self.subj_mask, self.obj_mask = (None, None, None)

    def resize_token_embeddings(self, new_num_tokens=None):
        if new_num_tokens is None:
            return

        old_num_tokens, old_embedding_dim = self.emb.weight.size()
        if old_num_tokens == new_num_tokens:
            return

        # Build new embeddings
        new_embeddings = ScaledEmbedding(new_num_tokens,
                                         old_embedding_dim,
                                         padding_idx=constant.PAD_ID)
        new_embeddings.to(self.emb.weight.device)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = self.emb.weight.data[:num_tokens_to_copy, :]
        self.emb = new_embeddings

#输入input_ids，dep_mask, dep_rel
    def forward(self, input_ids_or_bert_hidden, adj=None, dep_rel_matrix=None,
                wp_seq_lengths=None):

        if self.config.model_type == 'late_fusion':
            if self.config.syntax['finetune_bert']:
                embeddings = input_ids_or_bert_hidden
            else:
                embeddings = Variable(input_ids_or_bert_hidden.data)
        else:
            embeddings = self.emb(input_ids_or_bert_hidden)

            if self.config.syntax['embed_position']:
                seq_length = input_ids_or_bert_hidden.size(1)
                position_ids = torch.arange(seq_length,
                                            dtype=torch.long,
                                            device=input_ids_or_bert_hidden.device)
                position_ids = position_ids.unsqueeze(0).expand_as(input_ids_or_bert_hidden)
                position_embeddings = self.embed_pos(position_ids)
                embeddings += position_embeddings

            embeddings = self.input_dropout(embeddings)

            if self.config.syntax['contextual_rnn']:
                embeddings = self.rnn_dropout(self.encode_with_rnn(embeddings,
                                                                   wp_seq_lengths))
        syntax_inputs = embeddings

        dep_rel_emb = None
        if self.config.syntax['use_dep_rel']:
            dep_rel_emb = self.rel_emb(dep_rel_matrix)

        if self.config.use_syntax:
            attention_mask = adj.clone().detach().unsqueeze(1)
            attention_mask = attention_mask.to(dtype=next(self.parameters()).dtype)
            attention_mask = (1.0 - attention_mask) * -10000.0

            h = self.syntax_encoder(syntax_inputs,
                                    attention_mask,
                                    dep_rel_emb)
            if self.config.model_type == 'late_fusion' and self.config.syntax['late_fusion_gated_connection']:
                h = self.gate(syntax_inputs,
                              h)
        else:
            h = syntax_inputs

        return h
