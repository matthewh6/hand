"""Transformer utility functions."""

import math

import numpy as np
import torch
from torch import Tensor, nn


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""
    def get_position_angle_vec(position):
        return [
            position / np.power(10000, 2 * (hid_j // 2) / dimension)
            for hid_j in range(dimension)
        ]

    sinusoid_table = np.array(
        [get_position_angle_vec(pos_i) for pos_i in range(num_positions)]
    )
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    return torch.from_numpy(sinusoid_table).float()


def get_pos_encoding(pos_enc_type: str, embedding_dim: int, max_len: int):
    """Get positional encoding - returns module with .weight attribute."""
    if pos_enc_type == "sine":
        pe = create_sinusoidal_pos_embedding(
            num_positions=max_len, dimension=embedding_dim
        )
        layer = nn.Embedding(max_len, embedding_dim)
        layer.weight.data = pe
        layer.weight.requires_grad = False
        return layer
    elif pos_enc_type == "learned":
        return nn.Embedding(max_len, embedding_dim)
    else:
        raise ValueError(f"Unknown pos_enc_type: {pos_enc_type}")
