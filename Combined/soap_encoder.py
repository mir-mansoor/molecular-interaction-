import torch
import torch.nn as nn
from typing import Optional


# -------------------------------------------------------------
# Token MLP (shallow, 3 layers with ReLU + Dropout)
# -------------------------------------------------------------
class TokenMLP(nn.Module):
    """
    Shallow per-token MLP:
    in_dim -> token_emb_dim -> hidden_dim -> token_emb_dim
    Applies ReLU + Dropout between layers.
    """
    def __init__(self, in_dim: int, token_emb_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        # self.fc1 = nn.Linear(in_dim, 256)
        # self.fc2 = nn.Linear(256, 128)
        # self.fc3 = nn.Linear(128, 64)

        self.fc1 = nn.Linear(in_dim, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.act = nn.ReLU()
        self.drop1 = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.act(self.fc3(x))
        return x


# -------------------------------------------------------------
# SoapEncoder with Flattening + Progressive Projection Head
# -------------------------------------------------------------
class SoapEncoder(nn.Module):
    """
    Encodes per-token SOAP descriptors with a shallow TokenMLP,
    applies masking, then flattens (B, N, d) -> (B, N·d),
    and finally projects to a fixed 256-dim drug embedding
    using progressive halving layers (dynamic).
    """
    def __init__(
        self,
        token_in_dim: int,
        token_emb_dim: int = 128, #128, 64
        hidden_dim: int = 256,
        dropout: float = 0.3,
        num_tokens: int = 30,             # must be fixed for flattening
        project_out_dim: int = 256        # final drug embedding size
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_emb_dim = token_emb_dim
        self.token_mlp = TokenMLP(token_in_dim, token_emb_dim, hidden_dim, dropout)

        flattened_dim = num_tokens * token_emb_dim

        # dynamically compute halved dimensions, ensuring they stay >= project_out_dim
        dim1 = max(flattened_dim // 2, project_out_dim)
        dim2 = max(flattened_dim // 4, project_out_dim)

        # projection head to drug embedding (progressive halving)
        self.out_proj = nn.Sequential(
            nn.Linear(flattened_dim, dim1),
            nn.BatchNorm1d(dim1),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(dim1, dim2),
            nn.BatchNorm1d(dim2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(dim2, project_out_dim),
            nn.BatchNorm1d(project_out_dim),
            # nn.ReLU(),
            # nn.Dropout(dropout),
        )
        self._out_dim = project_out_dim

    @property
    def out_dim(self) -> int:
        return self._out_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, in_dim)
        mask:   (B, N) boolean (True for valid tokens, False for padding)
        returns: (B, out_dim) fixed-size drug embedding
        """
        B, N, _ = tokens.shape
        assert N == self.num_tokens, f"Expected {self.num_tokens} tokens, got {N}"

        # Per-token embedding
        emb = self.token_mlp(tokens)  # (B, N, token_emb_dim)

        # Apply mask: set invalid tokens to 0
        emb = emb.masked_fill(~mask.unsqueeze(-1), 0)

        # Flatten all tokens into one vector
        flat = emb.view(B, -1)  # (B, N·token_emb_dim)

        # Project to fixed drug embedding
        out = self.out_proj(flat)  # (B, out_dim)
        return out
