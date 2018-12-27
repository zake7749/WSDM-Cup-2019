"""Relational Neural Network idea."""
import math
import copy
import itertools
import torch
import numpy as np
from torch import nn
import torch.nn.functional as F
from pytorch_pretrained_bert.modeling import PreTrainedBertModel
from pytorch_pretrained_bert import BertModel


class RNModel(PreTrainedBertModel):

    def __init__(self, config, num_labels):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.attention = MultiHeadedAttention(4, 768)
        self.d_k = self.attention.d_k
        self.relation_network = RelationNetwork(self.d_k)
        self.classifier = nn.Linear(self.d_k, 1)  # TODO: d_k too small? MLP?
        self.loss = nn.CrossEntropyLoss()

    def forward(self, batch):
        # returns loss, acc, preds
        logits = torch.cat(
            [self.scoring(batch.pair_0), self.scoring(batch.pair_1)],
            dim=1)
        loss = self.loss(logits, batch.labels)
        preds = logits.max(1)[1].detach().cpu().numpy()
        acc = np.mean(batch.labels.detach().cpu().numpy() == preds)
        return loss, preds, acc

    def scoring(self, pair):
        # use BERT as an encoder
        context, _ = self.bert(*pair.context)  # [b, n, d]
        query, _ = self.bert(*pair.query)      # [b, n, d]

        # generate attention perspectives on context and query
        attn = self.attention(
            torch.cat([context, query], dim=1))

        # relation network
        relations = self.relation_network(attn)

        # scores
        return self.classifier(relations)


class RelationNetwork(nn.Module):
    """https://arxiv.org/abs/1706.01427"""

    def __init__(self, d_model):
        """Create a new Relation Network.

        Args:
          d_model: Int, dimension of model.
        """
        super().__init__()
        self.W_g = nn.Linear(d_model * 2, d_model)
        self.relu = nn.ReLU()
        self.W_f = nn.Linear(d_model, d_model)

    def forward(self, objects, combinations=None):
        """Compute relations vectors for a batch.

        Args:
          objects: Tensor of shape [batch, num_objects, d_model].
          combinations: List of int tuples with combination ixs. Can be left as
            None in which case the combinations will automatically be calculated
            as \binom{num_objects}{2}.

        Returns:
          Tensor of shape [batch, d_model].
        """
        summands = self.gather_combinations(objects)
        summands = self.W_g(summands)
        summands = torch.cat(summands, dim=1)
        return self.W_f(summands.sum(dim=1))

    @staticmethod
    def gather_combinations(t):
        """Gathers all length two combinations of the object vectors.

        Args:
          t: Tensor of shape [batch, num_objects, d_model].

        Returns:
          Tensor of shape [batch, \binom{num_objects}{2}, d_model * 2].
        """
        n = t.size(1)
        combinations = itertools.combinations(range(n), 2)
        o = []
        for i, j in combinations:
            o.append(torch.cat(
                [t[:, i, :].unsqueeze(1), t[:, j, :].unsqueeze(1)],
                dim=2))
        return torch.cat(o, dim=1)


class MultiHeadedAttention(nn.Module):
    """Bastardized http://nlp.seas.harvard.edu/2018/04/03/attention.html."""

    def __init__(self, h, d_model, dropout=0.1):
        """"Create a new MultiHeadedAttention.

        Args:
          h: Int, number of heads.
          d_model: Int, dimension of model.
          dropout: Float.
        """
        super().__init__()
        if d_model % h != 0:
            raise ValueError('d_model (%s) %% h(%s) != 0.' % (d_model, h))
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None  # for saving attentions (and therefore viewing)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """"Compute multi-headed attention.

        Args:
          query: Tensor of shape [batch, n_queries, d_model].
          key: Tensor of shape [batch, n_keys, d_model].
          value: Tensor of shape [batch, n_values, d_model].
          mask: NFI.

        Returns:
          Tensor of shape [batch, n_heads, d_k], where d_k is the
            dimension of the attention subspace.
        """
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # [batch, n_heads, n_values, d_k]
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        # [batch, n_heads, n_values, d_k]
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        return x.sum(dim=-2)


def attention(query, key, value, mask=None, dropout=None):
    """"Compute Scaled Dot Product Attention.

    Args:
      query: Tensor of shape [batch, n_heads, n_queries, d_k].
      key: Tensor of shape [batch, n_heads, n_keys, d_k].
      value: Tensor of shape [batch, n_heads, n_values, d_k].n m
      mask: NFI.
      dropout: Function.

    Returns:
      Tensor of shape [batch, n_heads, num_values, d_k].
    """
    d_k = query.size(-1)
    # [b, h, n, k] x [b, h, k, n] \in [b, h, n, n]
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)  # [b, h, n, n]
    if dropout is not None:
        p_attn = dropout(p_attn)
    # [b, h, n, n] x [b, h, n, k] \in [b, h, n, k]
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    """"Produce N identical layers.

    Args:
      module: torch.nn.Module.
      N: Int, number of clones.

    Returns:
      torch.nn.ModuleList.
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
