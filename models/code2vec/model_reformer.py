import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from reformer_pytorch import Reformer
from utils import *

class MODEL(nn.Module):
    def __init__(self, num_concepts, num_problems, hidden_dim, hidden_layers, nodes_dim,
                 paths_dim, codevec_size, concept_embed_dim, np, gpu):
        super(MODEL, self).__init__()

        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.gpu = gpu
        self.num_concepts = num_concepts
        self.codevec_size = codevec_size
        self.np = np
        self.concept_embed_dim = concept_embed_dim

        self.dropout = nn.Dropout(0.6)

        # Embeddings
        self.p_id_embed = nn.Embedding(num_problems + 1, concept_embed_dim)
        self.c_id_embed = nn.Parameter(
            torch.empty(2 * num_concepts + 1, concept_embed_dim),
            requires_grad=True
        )

        # Input feature dimension
        self.LSTM_feature_dim = 2 + codevec_size + num_concepts + 1

        # code2vec network
        self.node_embedding = nn.Embedding(nodes_dim, codevec_size)
        self.path_embedding = nn.Embedding(paths_dim, codevec_size)
        self.cv_fc = nn.Linear(3 * codevec_size, codevec_size, bias=False)

        a = torch.nn.init.uniform_(
            torch.empty(codevec_size, 1, dtype=torch.float32, requires_grad=True)
        )
        self.a = nn.parameter.Parameter(a, requires_grad=True)

        # ✅ Reformer replaces LSTM
        self.reformer = Reformer(
            dim=self.LSTM_feature_dim,
            depth=hidden_layers,
            heads=8,
            lsh_dropout=0.1,
            causal=True
        )

        self.predict_Linear = nn.Linear(hidden_dim, num_concepts + 1, bias=True)

    # -----------------------------
    # Initialization
    # -----------------------------
    def init_embeddings(self):
        nn.init.kaiming_normal_(self.node_embedding.weight)
        nn.init.kaiming_normal_(self.path_embedding.weight)
        nn.init.kaiming_normal_(self.cv_fc.weight)

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_Linear.weight)
        nn.init.constant_(self.predict_Linear.bias, 0)
        nn.init.kaiming_normal_(self.c_id_embed.data)
        nn.init.kaiming_normal_(self.p_id_embed.weight)

    # -----------------------------
    # Code2Vec
    # -----------------------------
    def code2vec(self, paths, context_embedded, bs, seqlen):

        num_paths = paths.shape[2]

        context_after_dense = variable(torch.tanh(context_embedded), self.gpu)

        attention_weight = self.a.repeat(bs, seqlen, 1, 1)

        attention_weight = torch.bmm(
            context_after_dense.view(bs * seqlen, num_paths, self.codevec_size),
            attention_weight.view(bs * seqlen, self.codevec_size, 1)
        ).view(bs, seqlen, num_paths, 1)

        code_vectors = torch.sum(
            context_after_dense * attention_weight.expand_as(context_after_dense),
            dim=2
        )

        return code_vectors

    # -----------------------------
    # Preprocess
    # -----------------------------
    def preprocess(self, paths, starts, ends):

        if self.np < 200:
            paths = paths[:, :, :self.np]
            starts = starts[:, :, :self.np]
            ends = ends[:, :, :self.np]

        starts_embedded = self.node_embedding(starts)
        paths_embedded = self.path_embedding(paths)
        ends_embedded = self.node_embedding(ends)

        context_embedded = torch.cat(
            (starts_embedded, paths_embedded, ends_embedded), dim=3
        )

        context_embedded = self.dropout(context_embedded)
        context_embedded = self.cv_fc(context_embedded)

        return context_embedded, starts, paths, ends, paths_embedded

    # -----------------------------
    # Forward
    # -----------------------------
    def forward(self, p_id, c_id, starts, paths, ends, masks, target_c, result, c_embed, cur_result):

        bs = p_id.shape[0]
        seqlen = p_id.shape[1]

        # Code2Vec
        context_embedded, starts, paths, ends, paths_embedded = self.preprocess(paths, starts, ends)
        code_vectors = self.code2vec(paths, context_embedded, bs, seqlen)

        # Input to Reformer
        reformer_input = torch.cat([c_embed, code_vectors, cur_result], 2)

        # ✅ Reformer forward
        reformer_out = self.reformer(reformer_input)

        # ⚠️ Match dimension if needed
        if reformer_out.shape[-1] != self.hidden_dim:
            reformer_out = nn.Linear(reformer_out.shape[-1], self.hidden_dim).to(reformer_out.device)(reformer_out)

        reformer_out = reformer_out.contiguous()

        # Prediction
        num_concepts = torch.sum(target_c, 2).view(-1)

        prediction = self.predict_Linear(reformer_out.view(bs * seqlen, -1))

        prediction_1d = torch.bmm(
            prediction.unsqueeze(1),
            target_c.view(bs * seqlen, -1).unsqueeze(2)
        ).squeeze(2)

        mask = num_concepts.gt(0)

        num_concepts = torch.masked_select(num_concepts, mask)
        filtered_pred = torch.masked_select(prediction_1d.squeeze(1), mask)

        filtered_pred = torch.div(filtered_pred, num_concepts)

        filtered_target = torch.masked_select(result.squeeze(1), mask)

        loss = F.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target