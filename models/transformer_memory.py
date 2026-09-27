import torch
import torch.nn as nn


class TransformerMemory(nn.Module):

    def __init__(
        self,
        embedding_dim=256,
        num_heads=4,
        memory_length=8,
        num_layers=2,
        feedforward_dim=512
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.memory_length = memory_length

        # ----------------------------------------------------
        # Learnable positional embedding
        # ----------------------------------------------------

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                memory_length,
                embedding_dim
            )
        )

        nn.init.normal_(
            self.position_embedding,
            mean=0.0,
            std=0.02
        )

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        self.final_norm = nn.LayerNorm(
            embedding_dim
        )

    def forward(
        self,
        current_embedding,
        memory
    ):

        # ----------------------------------------------------
        # memory shape:
        #
        # [B, T, embedding_dim]
        # ----------------------------------------------------

        sequence_length = memory.size(1)

        # ----------------------------------------------------
        # Add positional information
        # ----------------------------------------------------

        position_embedding = (
            self.position_embedding[
                :, :sequence_length, :
            ]
        )

        x = memory + position_embedding

        # ----------------------------------------------------
        # Transformer Encoder
        # ----------------------------------------------------

        x = self.encoder(x)

        # ----------------------------------------------------
        # Last token represents current belief
        # ----------------------------------------------------

        belief_state = x[:, -1, :]

        # ----------------------------------------------------
        # Final normalization
        # ----------------------------------------------------

        belief_state = self.final_norm(
            belief_state
        )

        # ----------------------------------------------------
        # Attention weights
        #
        # nn.TransformerEncoder does not directly expose
        # attention weights.
        #
        # We return None for compatibility.
        # ----------------------------------------------------

        attention_weights = None

        return (
            belief_state,
            attention_weights
        )