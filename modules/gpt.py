import numpy as np
import torch
import torch.nn as nn

import colossalai.nn as col_nn
import model_zoo.gpt.gpt as col_gpt
from colossalai.nn.layer.utils import CheckpointModule
from colossalai.core import global_context as gpc
from colossalai.logging import get_dist_logger
from colossalai.context.parallel_mode import ParallelMode
from colossalai.nn.layer.wrapper import PipelineSharedModuleWrapper
from colossalai.builder.pipeline import partition_uniform


class GPTEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_dim, max_seq_len, padding_idx=None, emb_dropout=0.):
        super(GPTEmbedding, self).__init__()
        self.word_embeddings = col_nn.Embedding(vocab_size, hidden_dim, padding_idx=padding_idx)
        self.position_embeddings = col_nn.Embedding(max_seq_len, hidden_dim)

        self.dropout = col_nn.Dropout(emb_dropout)
        self.d_model = hidden_dim

    @property
    def word_embedding_weight(self):
        return self.word_embeddings.weight

    def forward(self, x, position_ids=None):
        seq_len = x.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, dtype=torch.long, device=x.device).unsqueeze(0)
        x = self.word_embeddings(x) * np.sqrt(self.d_model) + self.position_embeddings(position_ids)
        return self.dropout(x)


class GPTSelfAttention(nn.Module):
    def __init__(self, hidden_dim, num_head, attn_dropout, dropout):
        super(GPTSelfAttention, self).__init__()
        assert hidden_dim % num_head == 0

        self.num_head = num_head
        self.head_dim = hidden_dim // num_head

        self.linears = nn.ModuleList([col_nn.Linear(hidden_dim, hidden_dim) for _ in range(4)])
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = col_nn.Dropout(dropout)
        self.attn_dropout = col_nn.Dropout(attn_dropout)

    def forward(self, x, attention_mask=None):
        """
        x:              batch size, seq len, hidden dim
        attention mask: batch size, 1, seq len, seq len
        """
        nbatches, seq_len = x.shape[:2]

        q, k, v = [l(x).view(nbatches, -1, self.num_head, self.head_dim).transpose(1, 2) for l in self.linears[:3]]

        scores = torch.matmul(q, k.transpose(-1, -2)) / np.sqrt(self.head_dim)

        # seq len, seq len ???
        causal_mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.uint8, device=x.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0).bool()

        if attention_mask is not None:
            causal_mask = causal_mask & attention_mask

        scores = scores.masked_fill(causal_mask == 0, -1e5)
        attn = self.attn_dropout(self.softmax(scores))

        x = torch.matmul(attn, v)
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.num_head * self.head_dim)

        return self.dropout(self.linears[-1](x))


class GPTMLP(nn.Module):
    def __init__(self, hidden_dim, mlp_ratio, activation, mlp_dropout, dropout):
        super(GPTMLP, self).__init__()
        inter_dim = int(hidden_dim*mlp_ratio)
        self.w_1 = col_nn.Linear(hidden_dim, inter_dim)
        self.w_2 = col_nn.Linear(inter_dim, hidden_dim)
        self.activation = activation
        self.dropout = col_nn.Dropout(dropout)
        self.mlp_dropout = col_nn.Dropout(mlp_dropout)

    def forward(self, x):
        return self.mlp_dropout(self.w_2(self.dropout(self.activation(self.w_1(x)))))


class GPTBlock(CheckpointModule):
    def __init__(self, hidden_dim, layer_norm_eps,
                 num_head, attn_dropout, dropout,
                 mlp_ratio, activation, mlp_dropout,
                 checkpoint=False, activation_offload=False):
        super(GPTBlock, self).__init__(checkpoint, activation_offload)

        self.attn_norm = col_nn.LayerNorm(normalized_shape=hidden_dim, eps=layer_norm_eps)
        self.attn = GPTSelfAttention(hidden_dim, num_head, attn_dropout, dropout)

        self.mlp_norm = col_nn.LayerNorm(normalized_shape=hidden_dim, eps=layer_norm_eps)
        self.mlp = GPTMLP(hidden_dim, mlp_ratio, activation, mlp_dropout, dropout)

    def _forward(self, x, attention_mask=None):
        # attention sublayer
        x = x + self.attn(self.attn_norm(x), attention_mask)

        # ffn sublayer
        x = x + self.mlp(self.mlp_norm(x))
        return x, attention_mask


class PipelineGPT(nn.Module):
    def __init__(self, vocab_size=50304, max_seq_len=1024, padding_idx=None, emb_dropout=0.1,
                 hidden_dim=1600, num_head=32, num_layer=48, attn_dropout=0.1, dropout=0.1,
                 layer_norm_eps=1e-5, mlp_ratio=4, mlp_dropout=0.1, activation=nn.functional.gelu,
                 first=False, last=False, checkpoint=False, activation_offload=False):
        super(PipelineGPT, self).__init__()
        self.first = first
        self.last = last

        if first:
            self.embedding = GPTEmbedding(vocab_size, hidden_dim, max_seq_len, padding_idx, emb_dropout)

        self.blocks = nn.ModuleList([
            GPTBlock(hidden_dim, layer_norm_eps, num_head, attn_dropout, dropout, mlp_ratio, activation, mlp_dropout,
                     checkpoint, activation_offload) for _ in range(num_layer)
        ])

        if last:
            self.norm = col_nn.LayerNorm(normalized_shape=hidden_dim, eps=layer_norm_eps)
            self.head = col_gpt.GPTLMHead(vocab_size=vocab_size, dim=hidden_dim, bias=False)

    def forward(self, x=None, input_ids=None, attention_mask=None):
        if self.first:
            x = self.embedding(input_ids)

        if attention_mask is not None:
            # batch size, 1, 1, seq len
            attention_mask = col_nn.partition_batch(attention_mask).unsqueeze(1).unsqueeze(2).bool()

        for block in self.blocks:
            x, attention_mask = block(x, attention_mask)

        if self.last:
            x = self.head(self.norm(x))

        return x


def build_pipeline_model(num_layers, num_chunks, device=torch.device('cuda'), **kwargs):
    logger = get_dist_logger()
    pipeline_size = gpc.get_world_size(ParallelMode.PIPELINE)
    pipeline_rank = gpc.get_local_rank(ParallelMode.PIPELINE)
    rank = gpc.get_global_rank()
    wrapper = PipelineSharedModuleWrapper([0, pipeline_size - 1])
    parts = partition_uniform(num_layers, pipeline_size, num_chunks)[pipeline_rank]

    models = []
    for start, end in parts:
        kwargs['num_layer'] = end - start
        kwargs['first'] = start == 0
        kwargs['last'] = end == num_layers
        logger.info(f'Rank{rank} build layer {start}-{end}, {end - start}/{num_layers} layers')
        chunk = PipelineGPT(**kwargs).to(device)
        if start == 0:
            wrapper.register_module(chunk.embedding.word_embeddings)
        elif end == num_layers:
            wrapper.register_module(chunk.head)
        models.append(chunk)
    if len(models) == 1:
        model = models[0]
    else:
        model = nn.ModuleList(models)
    return model
