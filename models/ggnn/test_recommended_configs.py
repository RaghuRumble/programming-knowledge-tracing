import torch
import argparse
import numpy as np
import pandas as pd
import os
import json
import random
import torch.nn as nn

try:
    from linformer import Linformer
except ImportError:
    Linformer = None

from load_data_ggnn import DATA_ggnn
from run import train, test
from model import MODEL
from utils import variable


class LinformerMODEL(MODEL):
    def __init__(self, num_concepts, num_problems, hidden_dim, hidden_layers,
                 concept_embed_dim, vocablen, node_embed_dim, num_nodes, gpu,
                 seqlen, linformer_depth=2, linformer_heads=4, linformer_k=64,
                 dropout=0.0):
        super().__init__(num_concepts=num_concepts,
                         num_problems=num_problems,
                         hidden_dim=hidden_dim,
                         hidden_layers=hidden_layers,
                         concept_embed_dim=concept_embed_dim,
                         vocablen=vocablen,
                         node_embed_dim=node_embed_dim,
                         num_nodes=num_nodes,
                         gpu=gpu,
                         ggnn_layers=4,
                         dropout=dropout)

        if Linformer is None:
            raise ImportError(
                "Linformer is not installed. Run `pip install linformer` before using the Linformer config."
            )

        self.sequence_encoder = Linformer(
            dim=self.LSTM_feature_dim,
            seq_len=seqlen,
            depth=linformer_depth,
            k=linformer_k,
            heads=linformer_heads,
            one_kv_head=True,
            share_kv=True,
            reversible=False,
            dropout=dropout,
        )
        self.sequence_norm = nn.LayerNorm(self.LSTM_feature_dim)
        self.sequence_dropout = nn.Dropout(dropout)
        self.predict_Linear = nn.Linear(self.LSTM_feature_dim, num_concepts + 1, bias=True)

        for parameter in self.LSTM.parameters():
            parameter.requires_grad = False

    def forward(self, p_id, c_id, node_id, edge, edge_type, target_c, result, c_embed, cur_result):
        bs = p_id.shape[0]
        seqlen = p_id.shape[1]
        nodelen = node_id.shape[2]

        if self.num_nodes < 200:
            sample = random.sample(range(0, node_id.shape[2]), self.num_nodes)
            edge = edge[:, :, :, sample]
            edge_type = edge_type[:, :, sample]

        node_embed = self.node_embed(node_id).view(bs, seqlen, nodelen, -1)
        edge_weight = self.edge_embed(edge_type).mean(3).unsqueeze(3)

        codevec = []
        for i in range(bs):
            for j in range(seqlen):
                out = self.ggnnlayer(node_embed[i, j], edge[i, j], edge_weight[i, j])
                gate = variable(torch.zeros(out.size(0), dtype=torch.long), self.gpu)
                code_embedding = self.pool(out, batch=gate).squeeze(0)
                codevec.append(code_embedding)

        codevec = torch.cat([codevec[i].unsqueeze(0) for i in range(bs * seqlen)], 0).view(bs, seqlen, -1)

        sequence_input = torch.cat([c_embed, codevec, cur_result], 2)
        sequence_input = self.sequence_norm(sequence_input)
        sequence_output = self.sequence_encoder(sequence_input)
        sequence_output = self.sequence_dropout(sequence_output)

        num_concepts = torch.sum(target_c, 2).view(-1)

        prediction = self.predict_Linear(sequence_output.view(bs * seqlen, -1))
        prediction_1d = torch.bmm(prediction.unsqueeze(1),
                                  target_c.view(bs * seqlen, -1).unsqueeze(2)).squeeze(2)
        mask = num_concepts.gt(0)
        num_concepts = torch.masked_select(num_concepts, mask)
        filtered_pred = torch.masked_select(prediction_1d.squeeze(1), mask)
        filtered_pred = torch.div(filtered_pred, num_concepts)
        filtered_target = torch.masked_select(result.squeeze(1), mask)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(filtered_pred, filtered_target)

        return loss, torch.sigmoid(filtered_pred), filtered_target


def seed_torch(seed=0):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True


def train_with_params(fold, params_dict, gpu=0):
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu", type=int, default=gpu)
    parser.add_argument("--EPOCH", type=int, default=params_dict.get("EPOCH", 300))
    parser.add_argument("--fold", type=int, default=fold)
    parser.add_argument("--dataset", type=str, default="codeforces")
    parser.add_argument("--set_seed", type=bool, default=True)

    parser.add_argument("--node_embed_dim", type=int, default=params_dict.get("node_embed_dim", 64))
    parser.add_argument("--batch_size", type=int, default=params_dict.get("batch_size", 1))
    parser.add_argument("--init_lr", type=float, default=params_dict.get("init_lr", 0.01))
    parser.add_argument("--weight_decay", type=float, default=params_dict.get("weight_decay", 0.00001))
    parser.add_argument("--concept_embed_dim", type=int, default=params_dict.get("concept_embed_dim", 128))
    parser.add_argument("--num_nodes", type=int, default=params_dict.get("num_nodes", 200))
    parser.add_argument("--hidden_dim", type=int, default=params_dict.get("hidden_dim", 64))
    parser.add_argument("--hidden_layers", type=int, default=params_dict.get("hidden_layers", 2))
    parser.add_argument("--ggnn_layers", type=int, default=params_dict.get("ggnn_layers", 4))
    parser.add_argument("--dropout", type=float, default=params_dict.get("dropout", 0.0))
    parser.add_argument("--optimizer_type", type=str, default=params_dict.get("optimizer_type", "adam"))
    parser.add_argument("--sequence_encoder_type", type=str, default=params_dict.get("sequence_encoder_type", "lstm"))
    parser.add_argument("--linformer_depth", type=int, default=params_dict.get("linformer_depth", 2))
    parser.add_argument("--linformer_heads", type=int, default=params_dict.get("linformer_heads", 4))
    parser.add_argument("--linformer_k", type=int, default=params_dict.get("linformer_k", 64))

    parser.add_argument("--num_concepts", type=int, default=37)
    parser.add_argument("--num_problems", type=int, default=7152)
    parser.add_argument("--seqlen", type=int, default=200)
    parser.add_argument("--vocablen", type=int, default=115730)

    params = parser.parse_args([])

    for key, value in params_dict.items():
        setattr(params, key, value)

    if params.set_seed:
        seed_torch(0)

    root = "../../data/codeforces/ggnn/"
    train_path = root + "ggnn_train" + str(fold)
    val_path = root + "ggnn_valid" + str(fold)
    test_path = root + "ggnn_test"

    try:
        data = DATA_ggnn(num_concepts=params.num_concepts, seqlen=params.seqlen)

        train_p_id, train_c_id, train_node_id, train_edge, train_edge_type, train_target_c, train_result, train_c_embed, train_x_result = data.load_data(train_path)

        val_p_id, val_c_id, val_node_id, val_edge, val_edge_type, val_target_c, val_result, val_c_embed, val_x_result = data.load_data(val_path)

        if params.sequence_encoder_type.lower() == "linformer":
            model = LinformerMODEL(
                num_concepts=params.num_concepts,
                num_problems=params.num_problems,
                hidden_dim=params.hidden_dim,
                hidden_layers=params.hidden_layers,
                concept_embed_dim=params.concept_embed_dim,
                vocablen=params.vocablen,
                node_embed_dim=params.node_embed_dim,
                num_nodes=params.num_nodes,
                gpu=params.gpu,
                seqlen=params.seqlen,
                linformer_depth=params.linformer_depth,
                linformer_heads=params.linformer_heads,
                linformer_k=params.linformer_k,
                dropout=params.dropout,
            )
        else:
            model = MODEL(
                num_concepts=params.num_concepts,
                num_problems=params.num_problems,
                hidden_dim=params.hidden_dim,
                hidden_layers=params.hidden_layers,
                concept_embed_dim=params.concept_embed_dim,
                vocablen=params.vocablen,
                node_embed_dim=params.node_embed_dim,
                num_nodes=params.num_nodes,
                gpu=params.gpu,
                ggnn_layers=params.ggnn_layers,
                dropout=params.dropout
            )

        model.init_params()
        model.init_embeddings()

        if params.optimizer_type.lower() == "sgd":
            optimizer = torch.optim.SGD(
                params=model.parameters(),
                lr=params.init_lr,
                weight_decay=params.weight_decay,
                momentum=0.9
            )
        else:
            optimizer = torch.optim.Adam(
                params=model.parameters(),
                lr=params.init_lr,
                weight_decay=params.weight_decay
            )

        if params.gpu >= 0 and torch.cuda.is_available():
            torch.cuda.set_device(params.gpu)
            model.cuda()
        else:
            params.gpu = -1

        best_valid_auc = 0
        best_train_auc = 0
        best_model = model
        count = 0

        for epoch in range(params.EPOCH):
            train_loss, train_accuracy, train_auc = train(
                model,
                params,
                optimizer,
                train_p_id,
                train_c_id,
                train_node_id,
                train_edge,
                train_edge_type,
                train_target_c,
                train_result,
                train_c_embed,
                train_x_result
            )

            valid_loss, valid_accuracy, valid_auc = test(
                model,
                params,
                val_p_id,
                val_c_id,
                val_node_id,
                val_edge,
                val_edge_type,
                val_target_c,
                val_result,
                val_c_embed,
                val_x_result
            )

            print(
                f"Epoch {epoch + 1}/{params.EPOCH} | "
                f"Train AUC: {train_auc:.5f} | "
                f"Valid AUC: {valid_auc:.5f} | "
                f"Valid Acc: {valid_accuracy:.5f}"
            )

            if valid_auc > best_valid_auc:
                count = 0
                best_valid_auc = valid_auc
                best_train_auc = train_auc
                best_model = model
            else:
                count += 1
                if count == 5:
                    break

        test_p_id, test_c_id, test_node_id, test_edge, test_edge_type, test_target_c, test_result, test_c_embed, test_x_result = data.load_data(test_path)

        test_loss, test_accuracy, test_auc = test(
            best_model,
            params,
            test_p_id,
            test_c_id,
            test_node_id,
            test_edge,
            test_edge_type,
            test_target_c,
            test_result,
            test_c_embed,
            test_x_result
        )

        return {
            "fold": fold,
            "best_train_auc": best_train_auc,
            "best_valid_auc": best_valid_auc,
            "test_auc": test_auc,
            "test_accuracy": test_accuracy,
            "params": params_dict
        }

    except Exception as e:
        print(f"Error in fold {fold}: {e}")
        return None


def main():
    params = {
        "EPOCH": 300,
        "batch_size": 4,
        "init_lr": 0.005,
        "hidden_dim": 64,
        "node_embed_dim": 64,
        "ggnn_layers": 4,
        "dropout": 0.1,
        "weight_decay": 0.00001,
        "optimizer_type": "adam",

        "num_nodes": 200,
        "hidden_layers": 2,
        "concept_embed_dim": 128,

        "sequence_encoder_type": "linformer",
        "linformer_depth": 2,
        "linformer_heads": 4,
        "linformer_k": 64,
    }

    print("=" * 70)
    print("Running GGNN + Linformer")
    print(params)
    print("=" * 70)

    result = train_with_params(fold=1, params_dict=params, gpu=0)

    if result is not None:
        print("\nFinal Result:")
        print(f"Test AUC: {result['test_auc']:.5f}")
        print(f"Test Accuracy: {result['test_accuracy']:.5f}")
        print(f"Best Validation AUC: {result['best_valid_auc']:.5f}")
    else:
        print("Training failed.")
        
if __name__ == "__main__":
    main()