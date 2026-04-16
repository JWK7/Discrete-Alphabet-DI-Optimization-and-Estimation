from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch
from torch import Tensor, nn


EstimatorType = Literal["di", "mi", "directed_information", "mutual_information"]
ArchitectureType = Literal["mlp", "lstm"]


def _activation(name: str) -> nn.Module:
	key = name.lower()
	if key == "relu":
		return nn.ReLU()
	if key == "gelu":
		return nn.GELU()
	if key == "tanh":
		return nn.Tanh()
	if key == "silu":
		return nn.SiLU()
	raise ValueError(f"Unsupported activation '{name}'.")


class MLPBackbone(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dims: Sequence[int] = (128, 128),
		output_dim: int = 1,
		activation: str = "relu",
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		if input_dim <= 0:
			raise ValueError("input_dim must be > 0.")
		if output_dim <= 0:
			raise ValueError("output_dim must be > 0.")

		layers: list[nn.Module] = []
		in_dim = input_dim
		act = _activation(activation)

		for hidden_dim in hidden_dims:
			if hidden_dim <= 0:
				raise ValueError("All hidden dimensions must be > 0.")
			layers.append(nn.Linear(in_dim, hidden_dim))
			layers.append(act.__class__())
			if dropout > 0:
				layers.append(nn.Dropout(dropout))
			in_dim = hidden_dim

		layers.append(nn.Linear(in_dim, output_dim))
		self.net = nn.Sequential(*layers)

	def forward(self, features: Tensor) -> Tensor:
		if features.dim() != 2:
			raise ValueError("MLPBackbone expects shape (batch, features).")
		return self.net(features)


class LSTMBackbone(nn.Module):
	def __init__(
		self,
		input_dim: int,
		hidden_dim: int = 128,
		num_layers: int = 1,
		output_dim: int = 1,
		dropout: float = 0.0,
	) -> None:
		super().__init__()
		if input_dim <= 0:
			raise ValueError("input_dim must be > 0.")
		if hidden_dim <= 0:
			raise ValueError("hidden_dim must be > 0.")
		if num_layers <= 0:
			raise ValueError("num_layers must be > 0.")
		if output_dim <= 0:
			raise ValueError("output_dim must be > 0.")

		effective_dropout = dropout if num_layers > 1 else 0.0
		self.lstm = nn.LSTM(
			input_size=input_dim,
			hidden_size=hidden_dim,
			num_layers=num_layers,
			batch_first=True,
			dropout=effective_dropout,
		)
		self.head = nn.Linear(hidden_dim, output_dim)

	def forward(self, sequence: Tensor) -> Tensor:
		if sequence.dim() != 3:
			raise ValueError("LSTMBackbone expects shape (batch, time, features).")
		_, (hidden_n, _) = self.lstm(sequence)
		final_hidden = hidden_n[-1]
		return self.head(final_hidden)


class PairScoringNetwork(nn.Module):
	def __init__(self, backbone: nn.Module, architecture: ArchitectureType) -> None:
		super().__init__()
		self.backbone = backbone
		self.architecture = architecture

	def forward(self, x: Tensor, y: Tensor) -> Tensor:
		if x.size(0) != y.size(0):
			raise ValueError("x and y must have the same batch size.")

		if self.architecture == "mlp":
			if x.dim() != 2 or y.dim() != 2:
				raise ValueError("For MLP, x and y must both have shape (batch, features).")
			features = torch.cat([x, y], dim=-1)
			return self.backbone(features)

		if self.architecture == "lstm":
			if x.dim() != 3 or y.dim() != 3:
				raise ValueError(
					"For LSTM, x and y must both have shape (batch, time, features)."
				)
			if x.size(1) != y.size(1):
				raise ValueError("For LSTM, x and y must have the same sequence length.")
			sequence = torch.cat([x, y], dim=-1)
			return self.backbone(sequence)

		raise ValueError(f"Unsupported architecture '{self.architecture}'.")


class DirectedInformationNetwork(nn.Module):
	"""DINE: scores causal sequences for directed information estimation.

	forward(x_ctx, y_ctx, y_curr) where:
	  x_ctx  : (batch, k) or (batch, k, x_dim)  -- X_{t-k+1} … X_t
	  y_ctx  : (batch, k) or (batch, k, y_dim)  -- Y_{t-k}   … Y_{t-1}
	  y_curr : (batch,)  or (batch, y_dim)       -- Y_t
	"""

	def __init__(self, backbone: nn.Module, architecture: ArchitectureType) -> None:
		super().__init__()
		self.backbone = backbone
		self.architecture = architecture

	def forward(self, x_ctx: Tensor, y_ctx: Tensor, y_curr: Tensor) -> Tensor:
		if not (x_ctx.size(0) == y_ctx.size(0) == y_curr.size(0)):
			raise ValueError("x_ctx, y_ctx, and y_curr must have the same batch size.")

		# Normalise to 3-D: (batch, k, dim)
		if x_ctx.dim() == 2:
			x_ctx = x_ctx.unsqueeze(-1)
		if y_ctx.dim() == 2:
			y_ctx = y_ctx.unsqueeze(-1)
		if y_curr.dim() == 1:
			y_curr = y_curr.unsqueeze(-1)

		if x_ctx.size(1) != y_ctx.size(1):
			raise ValueError("x_ctx and y_ctx must have the same context length.")

		if self.architecture == "mlp":
			batch = x_ctx.size(0)
			features = torch.cat(
				[x_ctx.reshape(batch, -1), y_ctx.reshape(batch, -1), y_curr],
				dim=-1,
			)
			return self.backbone(features)

		if self.architecture == "lstm":
			# Align y with x: drop oldest y, append y_curr so that position i
			# holds [X_{t-k+1+i}, Y_{t-k+1+i}] — both at the same time step.
			y_shifted = torch.cat([y_ctx[:, 1:], y_curr.unsqueeze(1)], dim=1)
			sequence = torch.cat([x_ctx, y_shifted], dim=-1)
			return self.backbone(sequence)

		raise ValueError(f"Unsupported architecture '{self.architecture}'.")


class MutualInformationNetwork(PairScoringNetwork):
	"""MINE: scores i.i.d. (x, y) pairs for mutual information estimation.

	forward(x, y) where:
	  x : (batch, x_dim)
	  y : (batch, y_dim)
	"""

	def __init__(self, backbone: nn.Module, architecture: ArchitectureType) -> None:
		if architecture == "lstm":
			raise ValueError("LSTM architecture is not supported for mutual information estimation.")
		super().__init__(backbone, architecture)


@dataclass(frozen=True)
class NetworkConfig:
	estimator: EstimatorType
	architecture: ArchitectureType
	x_dim: int
	y_dim: int
	context_len: int = 1  # number of time steps fed to DINE; ignored by MINE
	hidden_dims: Sequence[int] = (128, 128)
	hidden_dim: int = 128
	num_layers: int = 1
	output_dim: int = 1
	activation: str = "relu"
	dropout: float = 0.0


def build_estimator_network(config: NetworkConfig) -> nn.Module:
	architecture = config.architecture.lower()
	estimator = config.estimator.lower()

	if architecture not in {"mlp", "lstm"}:
		raise ValueError(f"Unsupported architecture '{config.architecture}'.")

	if estimator in {"di", "directed_information"}:
		if architecture == "mlp":
			# Flattened x_ctx + y_ctx + y_curr
			input_dim = config.context_len * (config.x_dim + config.y_dim) + config.y_dim
		else:
			# LSTM processes one [x_t, y_t] pair per time step
			input_dim = config.x_dim + config.y_dim
	elif estimator in {"mi", "mutual_information"}:
		input_dim = config.x_dim + config.y_dim
	else:
		raise ValueError(f"Unsupported estimator type '{config.estimator}'.")

	if architecture == "mlp":
		backbone = MLPBackbone(
			input_dim=input_dim,
			hidden_dims=config.hidden_dims,
			output_dim=config.output_dim,
			activation=config.activation,
			dropout=config.dropout,
		)
	else:
		backbone = LSTMBackbone(
			input_dim=input_dim,
			hidden_dim=config.hidden_dim,
			num_layers=config.num_layers,
			output_dim=config.output_dim,
			dropout=config.dropout,
		)

	if estimator in {"di", "directed_information"}:
		return DirectedInformationNetwork(backbone=backbone, architecture=architecture)

	if estimator in {"mi", "mutual_information"}:
		return MutualInformationNetwork(backbone=backbone, architecture=architecture)

	raise ValueError(f"Unsupported estimator type '{config.estimator}'.")


def build_di_network(
	architecture: ArchitectureType,
	x_dim: int,
	y_dim: int,
	context_len: int = 1,
	**kwargs,
) -> nn.Module:
	return build_estimator_network(
		NetworkConfig(
			estimator="di",
			architecture=architecture,
			x_dim=x_dim,
			y_dim=y_dim,
			context_len=context_len,
			**kwargs,
		)
	)


def build_mi_network(
	architecture: ArchitectureType,
	x_dim: int,
	y_dim: int,
	**kwargs,
) -> nn.Module:
	return build_estimator_network(
		NetworkConfig(
			estimator="mi",
			architecture=architecture,
			x_dim=x_dim,
			y_dim=y_dim,
			**kwargs,
		)
	)
