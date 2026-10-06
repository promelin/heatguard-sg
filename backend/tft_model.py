"""Inference-only implementation of the V4 Temporal Fusion Transformer."""

from __future__ import annotations

import math

import torch
from torch import nn


QUANTILES = (0.1, 0.5, 0.9)


class GatedLinearUnit(nn.Module):
    def __init__(self, input_size: int, output_size: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_size, output_size * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.projection(x).chunk(2, dim=-1)
        return value * torch.sigmoid(gate)


class GateAddNorm(nn.Module):
    def __init__(self, size: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(size, size)
        self.norm = nn.LayerNorm(size)

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return self.norm(residual + self.dropout(self.glu(x)))


class GatedResidualNetwork(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int | None = None,
        context_size: int | None = None, dropout: float = 0.1,
    ) -> None:
        super().__init__()
        output_size = output_size or input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.context = nn.Linear(context_size, hidden_size, bias=False) if context_size else None
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.glu = GatedLinearUnit(output_size, output_size)
        self.skip = nn.Identity() if input_size == output_size else nn.Linear(input_size, output_size)
        self.norm = nn.LayerNorm(output_size)

    def forward(self, x: torch.Tensor, context: torch.Tensor | None = None) -> torch.Tensor:
        hidden = self.fc1(x)
        if self.context is not None and context is not None:
            while context.ndim < hidden.ndim:
                context = context.unsqueeze(1)
            hidden = hidden + self.context(context)
        hidden = torch.nn.functional.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)
        return self.norm(self.skip(x) + self.glu(hidden))


class VariableSelectionNetwork(nn.Module):
    def __init__(self, variables: int, hidden_size: int, context_size: int | None, dropout: float) -> None:
        super().__init__()
        self.variables = variables
        self.weight_grn = GatedResidualNetwork(
            variables * hidden_size, hidden_size, variables, context_size, dropout
        )
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(hidden_size, hidden_size, hidden_size, None, dropout)
            for _ in range(variables)
        ])

    def forward(
        self, embedded: torch.Tensor, context: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        flat = embedded.flatten(start_dim=-2)
        weights = torch.softmax(self.weight_grn(flat, context), dim=-1)
        transformed = torch.stack(
            [network(embedded[..., i, :]) for i, network in enumerate(self.variable_grns)], dim=-2
        )
        return (transformed * weights.unsqueeze(-1)).sum(dim=-2), weights


class InterpretableMultiHeadAttention(nn.Module):
    def __init__(self, hidden_size: int, heads: int, dropout: float) -> None:
        super().__init__()
        if hidden_size % heads:
            raise ValueError("hidden size must be divisible by heads")
        self.heads = heads
        self.head_size = hidden_size // heads
        self.query = nn.Linear(hidden_size, heads * self.head_size, bias=False)
        self.key = nn.Linear(hidden_size, heads * self.head_size, bias=False)
        self.value = nn.Linear(hidden_size, self.head_size, bias=False)
        self.output = nn.Linear(self.head_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, query: torch.Tensor, key_value: torch.Tensor, allowed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, query_length, _ = query.shape
        key_length = key_value.shape[1]
        q = self.query(query).view(batch, query_length, self.heads, self.head_size).transpose(1, 2)
        k = self.key(key_value).view(batch, key_length, self.heads, self.head_size).transpose(1, 2)
        v = self.value(key_value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_size)
        scores = scores.masked_fill(~allowed[None, None], torch.finfo(scores.dtype).min)
        weights = self.dropout(torch.softmax(scores, dim=-1))
        attended = torch.einsum("bhqk,bkd->bhqd", weights, v).mean(dim=1)
        return self.output(attended), weights.mean(dim=1)


class FullTemporalFusionTransformer(nn.Module):
    def __init__(
        self, history_variables: int, future_real_variables: int,
        future_cat_cardinalities: list[int], stations: int,
        hidden_size: int = 64, heads: int = 4, dropout: float = 0.1,
        cooling_gate_enabled: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.history_variables = history_variables
        self.future_real_variables = future_real_variables
        self.future_cat_variables = len(future_cat_cardinalities)
        self.cooling_gate_enabled = cooling_gate_enabled
        self.history_weight = nn.Parameter(torch.randn(history_variables, hidden_size) * 0.02)
        self.history_bias = nn.Parameter(torch.zeros(history_variables, hidden_size))
        self.future_real_weight = nn.Parameter(torch.randn(future_real_variables, hidden_size) * 0.02)
        self.future_real_bias = nn.Parameter(torch.zeros(future_real_variables, hidden_size))
        self.future_cat_embeddings = nn.ModuleList([
            nn.Embedding(cardinality, hidden_size) for cardinality in future_cat_cardinalities
        ])
        self.station_embedding = nn.Embedding(stations, hidden_size)
        self.static_real_weight = nn.Parameter(torch.randn(2, hidden_size) * 0.02)
        self.static_real_bias = nn.Parameter(torch.zeros(2, hidden_size))
        self.static_selection = VariableSelectionNetwork(3, hidden_size, None, dropout)
        self.static_variable_context = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.static_enrichment_context = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.static_hidden_context = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.static_cell_context = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.history_selection = VariableSelectionNetwork(history_variables, hidden_size, hidden_size, dropout)
        self.future_selection = VariableSelectionNetwork(
            future_real_variables + len(future_cat_cardinalities), hidden_size, hidden_size, dropout
        )
        self.encoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.decoder = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.post_lstm_gate = GateAddNorm(hidden_size, dropout)
        self.static_enrichment = GatedResidualNetwork(
            hidden_size, hidden_size, hidden_size, hidden_size, dropout
        )
        self.attention = InterpretableMultiHeadAttention(hidden_size, heads, dropout)
        self.post_attention_gate = GateAddNorm(hidden_size, dropout)
        self.positionwise = GatedResidualNetwork(hidden_size, hidden_size, dropout=dropout)
        self.pre_output_gate = GateAddNorm(hidden_size, dropout)
        self.output = nn.Linear(hidden_size, len(QUANTILES))
        if cooling_gate_enabled:
            self.cooling_gate = nn.Linear(hidden_size, 1)
            self.cooling_correction = nn.Linear(hidden_size, 1)

    def forward(
        self, history: torch.Tensor, future_real: torch.Tensor, future_cat: torch.Tensor,
        station: torch.Tensor, station_static: torch.Tensor, return_interpretability: bool = False,
        return_auxiliary: bool = False,
    ):
        station_variable = self.station_embedding(station).unsqueeze(-2)
        static_real = station_static.unsqueeze(-1) * self.static_real_weight[None] + self.static_real_bias[None]
        static_variables = torch.cat([station_variable, static_real], dim=-2)
        static, static_weights = self.static_selection(static_variables)
        variable_context = self.static_variable_context(static)
        enrichment_context = self.static_enrichment_context(static)
        initial_h = self.static_hidden_context(static).unsqueeze(0)
        initial_c = self.static_cell_context(static).unsqueeze(0)
        history_embedded = history.unsqueeze(-1) * self.history_weight[None, None] + self.history_bias[None, None]
        selected_history, history_weights = self.history_selection(history_embedded, variable_context)
        future_real_embedded = (
            future_real.unsqueeze(-1) * self.future_real_weight[None, None]
            + self.future_real_bias[None, None]
        )
        if self.future_cat_embeddings:
            future_cat_embedded = torch.stack([
                embedding(future_cat[..., i]) for i, embedding in enumerate(self.future_cat_embeddings)
            ], dim=-2)
            future_embedded = torch.cat([future_real_embedded, future_cat_embedded], dim=-2)
        else:
            future_embedded = future_real_embedded
        selected_future, future_weights = self.future_selection(future_embedded, variable_context)
        encoder_output, recurrent_state = self.encoder(selected_history, (initial_h, initial_c))
        decoder_output, _ = self.decoder(selected_future, recurrent_state)
        recurrent = torch.cat([encoder_output, decoder_output], dim=1)
        selected = torch.cat([selected_history, selected_future], dim=1)
        post_lstm = self.post_lstm_gate(recurrent, selected)
        enriched = self.static_enrichment(post_lstm, enrichment_context)
        history_length = history.shape[1]
        future_length = future_real.shape[1]
        allowed = torch.zeros(
            future_length, history_length + future_length, dtype=torch.bool, device=history.device
        )
        allowed[:, :history_length] = True
        allowed[:, history_length:] = torch.tril(
            torch.ones(future_length, future_length, dtype=torch.bool, device=history.device)
        )
        attended, attention_weights = self.attention(enriched[:, history_length:], enriched, allowed)
        attention_layer = self.post_attention_gate(attended, enriched[:, history_length:])
        positionwise = self.positionwise(attention_layer)
        final = self.pre_output_gate(positionwise, post_lstm[:, history_length:])
        prediction = self.output(final)
        auxiliary = {}
        if self.cooling_gate_enabled:
            cooling_probability = torch.sigmoid(self.cooling_gate(final).squeeze(-1))
            cooling_correction = torch.nn.functional.softplus(self.cooling_correction(final).squeeze(-1))
            prediction = prediction - (cooling_probability * cooling_correction).unsqueeze(-1)
            auxiliary = {
                "cooling_probability": cooling_probability,
                "cooling_correction": cooling_correction,
            }
        if return_interpretability:
            auxiliary.update({
                "static_weights": static_weights,
                "history_weights": history_weights,
                "future_weights": future_weights,
                "attention_weights": attention_weights,
            })
            return prediction, auxiliary
        if return_auxiliary:
            auxiliary["decoder_representation"] = final
            return prediction, auxiliary
        return prediction
