import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from .relative_deberta import DisentangledSelfAttention
from .ops import *
import math


__all__ = ['BertEncoder', 'BertEmbeddings', 'ACT2FN', 'BertLayerNorm']

def gelu(x):
  """Implementation of the gelu activation function.
    For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
    0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
  """
  return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
  return x * torch.sigmoid(x)

def linear_act(x):
  return x

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "tanh": torch.nn.functional.tanh, "linear": linear_act, 'sigmoid': torch.sigmoid}


class BertLayerNorm(nn.Module):
  """LayerNorm module in the TF style (epsilon inside the square root).
  """

  def __init__(self, size, eps=1e-12):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(size))
    self.bias = nn.Parameter(torch.zeros(size))
    self.variance_epsilon = eps

  def forward(self, x):
    input_type = x.dtype
    x = x.float()
    u = x.mean(-1, keepdim=True)
    s = (x - u).pow(2).mean(-1, keepdim=True)
    x = (x - u) / torch.sqrt(s + self.variance_epsilon)
    x = x.to(input_type)
    y = self.weight * x + self.bias
    return y

class BertSelfOutput(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.LayerNorm = BertLayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def forward(self, hidden_states, input_states, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
    return hidden_states


class BertAttention(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.self = DisentangledSelfAttention(config)
    self.output = BertSelfOutput(config)
    self.config = config

  def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None, rel_embeddings=None):
    self_output = self.self(hidden_states, attention_mask, return_att, query_states=query_states,
                            relative_pos=relative_pos, rel_embeddings=rel_embeddings)
    if return_att:
      self_output, att_matrix = self_output
    if query_states is None:
      query_states = hidden_states
    attention_output = self.output(self_output, query_states, attention_mask)

    if return_att:
      return (attention_output, att_matrix)
    else:
      return attention_output


class BertIntermediate(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.intermediate_act_fn = ACT2FN[config.hidden_act] \
      if isinstance(config.hidden_act, str) else config.hidden_act

  def forward(self, hidden_states):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.intermediate_act_fn(hidden_states)
    return hidden_states

class BertOutput(nn.Module):
  def __init__(self, config):
    super(BertOutput, self).__init__()
    self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.LayerNorm = BertLayerNorm(config.hidden_size, config.layer_norm_eps)
    self.dropout = StableDropout(config.hidden_dropout_prob)
    self.config = config

  def forward(self, hidden_states, input_states, mask=None):
    hidden_states = self.dense(hidden_states)
    hidden_states = self.dropout(hidden_states)
    hidden_states += input_states
    hidden_states = MaskedLayerNorm(self.LayerNorm, hidden_states)
    return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask, return_att=False, query_states=None, relative_pos=None,
                rel_embeddings=None):
        attention_output = self.attention(hidden_states, attention_mask, return_att=return_att,
                                          query_states=query_states, relative_pos=relative_pos,
                                          rel_embeddings=rel_embeddings)
        if return_att:
            attention_output, att_matrix = attention_output
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output, attention_mask)
        if return_att:
            return (layer_output, att_matrix)
        else:
            return layer_output


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

        self.relative_attention = getattr(config, 'relative_attention', False)
        if self.relative_attention:
            self.max_relative_positions = getattr(config, 'max_relative_positions', -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.rel_embeddings = nn.Embedding(self.max_relative_positions * 2, config.hidden_size)

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, return_att=False,
                query_states=None, relative_pos=None):
        """
        :params hidden_states: []

        """
        attention_mask = self.get_attention_mask(attention_mask)

    def get_attention_mask(self, attention_mask):
        # padding mask
        if attention_mask.dim() <= 2:  # [batch, length]
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = extended_attention_mask * extended_attention_mask.squeeze(-2).unsqueeze(-1)
            attention_mask = attention_mask.byte()
        # future mask  [batch, length, length]
        elif attention_mask.dim() == 3:
            attention_mask = attention_mask.unsqueeze(1)

        return attention_mask
