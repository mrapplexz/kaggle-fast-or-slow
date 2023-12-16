import math
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import networkx as nx
import pandas as pd

import torch
from gspan_mining.config import parser
from gspan_mining.main import main
import torch.nn.functional as F
import numpy as np
from madgrad import MADGRAD
from torch import nn, Tensor
from torch.nn import MSELoss
from torch.utils.data import Dataset
from torchmetrics import Metric, MeanMetric
from torchvision.ops import MLP
from tqdm import tqdm
from transformers import GraphormerModel, get_linear_schedule_with_warmup, AdamW
from xztrainer import XZTrainable, ContextType, BaseContext, DataType, ModelOutputsType, XZTrainer, XZTrainerConfig, \
    SchedulerType
from xztrainer.logger.tensorboard import TensorboardLoggingEngineConfig
from xztrainer.setup_helper import set_seeds, enable_tf32

from attention import GraphTransEncoder, GraphTransConfig

EMBED_DIM = 256


class NodeEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self._opcode = nn.Embedding(120, EMBED_DIM)
        self._features = nn.Linear(140, EMBED_DIM)
        self._processor = MLP(
            in_channels=EMBED_DIM * 2,
            hidden_channels=[EMBED_DIM, EMBED_DIM],
            activation_layer=torch.nn.GELU,
        )

    def forward(self, features: torch.Tensor, opcodes: torch.Tensor):
        op_enc = self._opcode(opcodes)
        f_enc = self._features(features)
        return self._processor(torch.cat([op_enc, f_enc], dim=-1))


class NodeMutationEmbedder(nn.Module):
    def __init__(self):
        super().__init__()
        self._mutation = nn.Linear(24, EMBED_DIM)
        self._processor = MLP(
            in_channels=EMBED_DIM,
            hidden_channels=[EMBED_DIM, EMBED_DIM],
            activation_layer=torch.nn.GELU,
        )

    def forward(self, features: torch.Tensor):
        return self._processor(self._mutation(features))


class DistanceAttentionBias(nn.Module):
    def __init__(self, clamp_value: int = 30):
        super().__init__()
        mat = torch.empty([clamp_value + 2])
        for i in range(len(mat)):
            mat[i] = math.e ** (-i / (clamp_value ** 0.5)) - 1
        self._mat = nn.Parameter(mat)
        self._clamp_value = clamp_value

    def forward(self, distance_matrix: torch.Tensor):
        distance_matrix = distance_matrix.clone()
        distance_matrix[distance_matrix == -1] = self._clamp_value + 2
        distance_matrix[distance_matrix > self._clamp_value] = self._clamp_value + 1
        return self._mat[distance_matrix]


class GraphPreModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._distance_bias = DistanceAttentionBias(clamp_value=30)
        self._node = NodeEmbedder()
        self._cls = nn.Parameter(torch.empty([1, 1, EMBED_DIM]))
        nn.init.normal_(self._cls)
        self._transformer_encoder = GraphTransEncoder(GraphTransConfig(
            num_hidden_layers=6,
            hidden_size=EMBED_DIM,
            intermediate_size=EMBED_DIM * 4,
            hidden_act='gelu',
            attention_heads=8,
            attention_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            layer_norm_eps=1e-12
        ))

    def forward(self, node_feat: torch.Tensor, node_opcode: torch.Tensor, dist_matrix: torch.Tensor):
        cls = self._cls.repeat(node_feat.shape[0], 1, 1)
        nodes = self._node(node_feat, node_opcode)
        nodes = torch.cat([cls, nodes], dim=1)
        dist_bias = self._distance_bias(dist_matrix)
        out = self._transformer_encoder(nodes, dist_bias)
        return out


class GraphPostModelTile(nn.Module):
    def __init__(self):
        super().__init__()
        self._distance_bias = DistanceAttentionBias(clamp_value=30)
        self._mutator = NodeMutationEmbedder()
        self._transformer_encoder = GraphTransEncoder(GraphTransConfig(
            num_hidden_layers=4,
            hidden_size=EMBED_DIM,
            intermediate_size=EMBED_DIM * 4,
            hidden_act='gelu',
            attention_heads=8,
            attention_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            layer_norm_eps=1e-12
        ))
        self._transformer_encoder.gradient_checkpointing = True

    def forward(self, hidden_states: torch.Tensor, mutation_features: torch.Tensor, dist_matrix: torch.Tensor):
        mutation_add = self._mutator(mutation_features)
        dist_matrix = self._distance_bias(dist_matrix)
        bs = 128
        embeds = []
        for i in range(0, mutation_add.shape[0], bs):
            mut_local = mutation_add[i:i + bs]
            hs_local = hidden_states.repeat(mut_local.shape[0], 1, 1)
            hs_local[:, 0, :] += mut_local
            dist_local = dist_matrix.repeat(mut_local.shape[0], 1, 1)
            out_local = self._transformer_encoder(hs_local, dist_local)
            embeds.append(out_local[:, 0])

        return torch.cat(embeds, dim=0)


class WholeModelTile(nn.Module):
    def __init__(self):
        super().__init__()
        self._pre = GraphPreModel()
        self._post = GraphPostModelTile()
        self._head = nn.Sequential(nn.Linear(EMBED_DIM, EMBED_DIM), nn.GELU(), nn.Linear(EMBED_DIM, 1))

    def forward(self, node_feat: torch.Tensor, node_opcode: torch.Tensor, dist_matrix: torch.Tensor, config_feat: torch.Tensor):
        pre_graph_embed = self._pre(node_feat, node_opcode, dist_matrix)
        post_graph_embed = self._post(pre_graph_embed, config_feat, dist_matrix)
        headed = self._head(post_graph_embed)
        return headed.squeeze(1).unsqueeze(0)


class GraphTileDataset(Dataset):
    def __init__(self, data_dir: Path, split: str, normalize: Optional[str] = None, is_train: bool = True):
        if normalize:
            self._stats = torch.load(f'stats_{normalize}.pt')
        else:
            self._stats = {
                'node_mean': torch.scalar_tensor(0),
                'node_std': torch.scalar_tensor(1),
                'config_mean': torch.scalar_tensor(0),
                'config_std': torch.scalar_tensor(1)
            }
        self._data_dir = data_dir
        self._files = [x for x in (data_dir / 'tile').rglob(f'**/{split}/*.npz')]
        self._is_train = is_train

    def __getitem__(self, item):
        file = self._files[item]
        file_name = file.relative_to(self._data_dir)
        file_id = f'{file_name.parts[0]}:{file_name.parts[1]}:{file_name.stem}'
        file = dict(np.load(file))
        g = nx.DiGraph()
        for node in range(len(file['node_opcode'])):
            g.add_node(node + 1)
        g.add_node(0)  # cls node
        for link_to, link_from in file['edge_index']:
            g.add_edge(link_from + 1, link_to + 1)
        for i, fts in enumerate(file['node_feat']):
            if fts[0] == 1:
                g.add_edge(i + 1, 0)
        shortest_paths_matrix = torch.empty([len(file['node_opcode']) + 1, len(file['node_opcode']) + 1],
                                            dtype=torch.long).fill_(-1)
        for from_node, to_nodes in nx.shortest_path_length(g):
            for to_node, length in to_nodes.items():
                shortest_paths_matrix[to_node][from_node] = length
        return {
            'file_id': file_id,
            'node_feat': (torch.Tensor(file['node_feat']) - self._stats['node_mean']) / self._stats['node_std'],
            'node_opcode': torch.LongTensor(file['node_opcode']),
            'shortest_paths_matrix': shortest_paths_matrix,  # first is cls node
            'config_feat': (torch.Tensor(file['config_feat']) - self._stats['config_mean']) / self._stats['config_std'],
            'config_runtime': torch.Tensor(file['config_runtime']) / torch.Tensor(file['config_runtime_normalizers']) if self._is_train else None
        }

    def __len__(self):
        return len(self._files)


class MultiElementRankLoss(nn.Module):
    """
    Loss function that compares the output of the model with the output of the model with a permutation of the elements
    """

    def __init__(self, margin: float = 0.1, number_permutations: int = 30) -> None:
        super().__init__()
        self.loss_fn = torch.nn.MarginRankingLoss(margin=margin, reduction='none')
        self.number_permutations = number_permutations

    def forward(self,
                            outputs: torch.Tensor,
                            config_runtime: torch.Tensor,
                            config_idxs: torch.Tensor
                            ):
        """
        Generates a permutation of the predictions and targets and calculates the loss MarginRankingLoss against the permutation
        Args:
            outputs: Tensor of shape (bs, seq_len) with the outputs of the model
            config_runtime: Tensor of shape (bs, seq_len) with the runtime of the model
            config_mask: Tensor of shape (bs, seq_len) with 1 in the positions of the elements
            and 0 in the positions of the padding
        Returns:
            loss: Tensor of shape (bs, seq_len) with the loss for each element in the batch
        """
        bs, num_configs = outputs.shape
        permutation = torch.randperm(num_configs)
        permuted_idxs = config_idxs[:, permutation]
        # We mask those cases where we compare the same configuration
        config_mask = torch.where(config_idxs != permuted_idxs, 1, 0)
        permuted_runtime = config_runtime[:, permutation]
        labels = 2 * ((config_runtime - permuted_runtime) > 0) - 1
        permuted_output = outputs[:, permutation]
        loss = self.loss_fn(outputs.view(-1, 1), permuted_output.view(-1, 1), labels.view(-1, 1))
        loss = loss.view(bs, num_configs) * config_mask
        return loss.mean()


class TrainableTile(XZTrainable):
    def __init__(self):
        self._loss = MSELoss()
        self._ranking_loss = MultiElementRankLoss()

    def step(self, context: BaseContext, data: DataType) -> Tuple[Tensor, ModelOutputsType]:
        model_out = context.model(data['node_feat'], data['node_opcode'], data['shortest_paths_matrix'], data['config_feat'])
        if data['config_runtime'] is not None:
            loss_rank = self._ranking_loss(model_out, data['config_runtime'], torch.arange(0, data['config_runtime'].shape[1], device=model_out.device).unsqueeze(0))
            # loss = self._loss(model_out.flatten(), data['config_runtime'].flatten())
            loss = loss_rank
            return loss, {
                'loss': loss,
                'file_ids': data['file_id'],
                'model_outs': model_out,
                'targets': data['config_runtime']
            }
        else:
            return None, {
                'model_outs': model_out,
                'file_ids': data['file_id']
            }

    def create_metrics(self, context_type: ContextType) -> Dict[str, Metric]:
        return {
            'loss': MeanMetric(),
            'tile_lb': MeanMetric()
        }

    def update_metrics(self, context_type: ContextType, model_outputs: Dict[str, List], metrics: Dict[str, Metric]):
        metrics['loss'].update(model_outputs['loss'])
        best_tgt = torch.stack([x.min() for x in model_outputs['targets']])
        best_pred = torch.stack([tgt[x.sort().indices[:5]].min() for tgt, x in zip(model_outputs['targets'], model_outputs['model_outs'])])
        metric = 1 - ((best_pred / best_tgt) - 1)
        metrics['tile_lb'].update(metric)


def collate(inputs):
    x = inputs[0]
    return {
        'file_id': [x['file_id']],
        'node_feat': x['node_feat'].unsqueeze(0),
        'node_opcode': x['node_opcode'].unsqueeze(0),
        'shortest_paths_matrix': x['shortest_paths_matrix'].unsqueeze(0),
        'config_feat': x['config_feat'],
        'config_runtime': x['config_runtime'].unsqueeze(0) if x['config_runtime'] is not None else None
    }


if __name__ == '__main__':
    action = sys.argv[1]
    enable_tf32()
    set_seeds(0x11137)
    model_tile = WholeModelTile()
    trainable = TrainableTile()
    trainer = XZTrainer(XZTrainerConfig(
        batch_size=1,  # todo collation
        batch_size_eval=1,
        experiment_name='check-permutation',
        epochs=10,
        optimizer=lambda mod: AdamW(mod.parameters(), lr=1e-4),
        scheduler=lambda opt, steps: get_linear_schedule_with_warmup(opt, int(steps * 0.1), steps),
        scheduler_type=SchedulerType.STEP,
        dataloader_num_workers=4,
        accumulation_batches=16,
        print_steps=10,
        eval_steps=100,
        save_steps=100,
        save_keep_n=5,
        collate_fn=collate,
        logger=TensorboardLoggingEngineConfig()
    ), model_tile, trainable)
    if action == 'train':
        ds_train = GraphTileDataset(Path('data/npz_all/npz'), 'train', normalize='tile')
        ds_val = GraphTileDataset(Path('data/npz_all/npz'), 'valid', normalize='tile')
        trainer.train(ds_train, ds_val)
    else:
        ds_test = GraphTileDataset(Path('data/npz_all/npz'), 'test', normalize='tile', is_train=False)
        trainer.load_last_checkpoint()
        outs, _ = trainer.infer(ds_test, calculate_metrics=False)
        bests = [(id, ';'.join([str(i.item()) for i in x.sort().indices[:5]])) for x, id in zip(outs['model_outs'], outs['file_ids'])]
        sample_sub = pd.read_csv('sample_submission.csv').set_index('ID')
        tile_pred = pd.DataFrame(bests, columns=['ID', 'TopConfigs']).set_index('ID')
        out_df = tile_pred.combine_first(sample_sub)
        out_df.to_csv('submission.csv')