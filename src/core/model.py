"""
Recurrent Neural Network (RNN) models for brain-to-text decoding.
"""

import torch
from torch import nn
from typing import Optional, Tuple


class GRUDecoder(nn.Module):
    """
    GRU-based decoder for brain signals.

    This model combines day-specific input layers, a multi-layer GRU,
    and a linear output layer for phoneme classification.
    """

    def __init__(
        self,
        neural_dim: int,
        n_units: int,
        n_days: int,
        n_classes: int,
        rnn_dropout: float = 0.0,
        input_dropout: float = 0.0,
        n_layers: int = 5,
        patch_size: int = 0,
        patch_stride: int = 0,
    ):
        """
        Args:
            neural_dim (int): Number of input channels (e.g., 512).
            n_units (int): Number of hidden units per GRU layer.
            n_days (int): Number of recording days (for day-specific layers).
            n_classes (int): Number of output classes (phonemes).
            rnn_dropout (float): Dropout rate for GRU layers.
            input_dropout (float): Dropout rate for the input layer output.
            n_layers (int): Number of recurrent layers.
            patch_size (int): Number of timesteps to concatenate on input.
            patch_stride (int): Stride for time patching.
        """
        super().__init__()

        self.neural_dim = neural_dim
        self.n_units = n_units
        self.n_classes = n_classes
        self.n_layers = n_layers
        self.n_days = n_days
        self.rnn_dropout = rnn_dropout
        self.input_dropout = input_dropout
        self.patch_size = patch_size
        self.patch_stride = patch_stride

        # Day-specific input layers (projection to a common latent space)
        self.day_weights = nn.ParameterList(
            [nn.Parameter(torch.eye(self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_biases = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, self.neural_dim)) for _ in range(self.n_days)]
        )
        self.day_activation = nn.Softsign()
        self.day_dropout = nn.Dropout(input_dropout)

        self.input_size = self.neural_dim
        if self.patch_size > 0:
            self.input_size *= self.patch_size

        self.gru = nn.GRU(
            input_size=self.input_size,
            hidden_size=self.n_units,
            num_layers=self.n_layers,
            dropout=self.rnn_dropout,
            batch_first=True,
            bidirectional=False,
        )

        # Initialize weights
        for name, param in self.gru.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        self.out = nn.Linear(self.n_units, self.n_classes)
        nn.init.xavier_uniform_(self.out.weight)

        # Learnable initial hidden state
        self.h0 = nn.Parameter(
            nn.init.xavier_uniform_(torch.zeros(1, 1, self.n_units))
        )

    def forward(
        self,
        x: torch.Tensor,
        day_idx: torch.Tensor,
        states: Optional[torch.Tensor] = None,
        return_state: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs forward pass.

        Args:
            x (torch.Tensor): Batch of neural input [B, T, D].
            day_idx (torch.Tensor): Indices of recording days for each trial in batch.
            states (torch.Tensor, optional): Initial hidden states.
            return_state (bool): Whether to return the final hidden states.

        Returns:
            torch.Tensor: Logits [B, T, C].
            torch.Tensor (optional): Hidden states.
        """
        # Apply day-specific layers
        day_weights = torch.stack([self.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat(
            [self.day_biases[i] for i in day_idx], dim=0
        ).unsqueeze(1)

        x = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        x = self.day_activation(x)

        if self.input_dropout > 0:
            x = self.day_dropout(x)

        # Input patching (time concatenation)
        if self.patch_size > 0:
            x = x.unsqueeze(1).permute(0, 3, 1, 2)  # [B, D, 1, T]
            x_unfold = x.unfold(3, self.patch_size, self.patch_stride)
            x_unfold = x_unfold.squeeze(2).permute(0, 2, 3, 1)  # [B, T_new, P, D]
            x = x_unfold.reshape(x.size(0), x_unfold.size(1), -1)

        # RNN pass
        if states is None:
            states = self.h0.expand(
                self.n_layers, x.shape[0], self.n_units
            ).contiguous()

        output, hidden_states = self.gru(x, states)
        logits = self.out(output)

        if return_state:
            return logits, hidden_states

        return logits
