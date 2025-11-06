# SRR-DDI/DrugBank/soap_io.py
import pickle
import numpy as np
import torch

def load_soap_tokens(path: str):
    with open(path, "rb") as f:
        d = pickle.load(f)  # dict[str DrugBankID] -> np.ndarray (T_max, D_soap)
    any_arr = next(iter(d.values()))
    T_max, D_soap = int(any_arr.shape[0]), int(any_arr.shape[1])
    return d, T_max, D_soap

def tokens_to_tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(arr.astype(np.float32))  # (T, D)

def make_mask(tokens_tensor: torch.Tensor) -> torch.Tensor:
    # Padded rows are zeros (as saved by PCA.py)
    return ~tokens_tensor.eq(0).all(dim=-1)         # (T,) bool
