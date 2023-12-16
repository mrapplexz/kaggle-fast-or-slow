import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers import apply_chunking_to_forward
from transformers.activations import ACT2FN


@dataclass
class GraphTransConfig:
    num_hidden_layers: int
    hidden_size: int
    intermediate_size: int
    hidden_act: str
    attention_heads: int
    attention_dropout_prob: float
    hidden_dropout_prob: float
    layer_norm_eps: float


class SelfAttention(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.num_attention_heads = config.attention_heads
        self.attention_head_size = int(config.hidden_size / config.attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_dropout_prob)

    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.FloatTensor = None,
    ) -> Tuple[torch.Tensor]:
        mixed_query_layer = self.query(hidden_states)

        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask
        attention_scores = attention_scores + attention_mask.unsqueeze(1)

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return context_layer


class SelfOutput(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Attention(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.self = SelfAttention(config)
        self.output = SelfOutput(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.FloatTensor
    ) -> Tuple[torch.Tensor]:
        self_outputs = self.self(
            hidden_states,
            attention_mask
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        return attention_output


class Intermediate(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class Output(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states: torch.Tensor, input_tensor: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Layer(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: torch.FloatTensor,
    ) -> Tuple[torch.Tensor]:
        attention_output = self.attention(
            hidden_states,
            attention_mask
        )

        layer_output = self.output(self.intermediate(attention_output), attention_output)

        return layer_output


class GraphTransEncoder(nn.Module):
    def __init__(self, config: GraphTransConfig):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([Layer(config) for _ in range(config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.FloatTensor] = None
    ) -> torch.Tensor:
        for i, layer_module in enumerate(self.layer):
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)

                    return custom_forward

                layer_outputs = checkpoint(
                    create_custom_forward(layer_module),
                    hidden_states,
                    attention_mask
                )
            else:
                layer_outputs = layer_module(
                    hidden_states,
                    attention_mask
                )

            hidden_states = layer_outputs
        return hidden_states
