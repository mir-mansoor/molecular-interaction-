# SOAP-Based 3D Molecular Representation Module for SRR-DDI
**Author:** *Mir Mansoor Ahmad*

This repository contains the **SOAP (Smooth Overlap of Atomic Positions)**–based modules developed to extend the **SRR-DDI** framework with **three-dimensional (3D)** atomic environment features.  
Only the original SOAP-related components are included here — the rest of the SRR-DDI source code is not redistributed.

---

## 🧩 Purpose
While SRR-DDI encodes molecules using **2D graph neural networks (GNNs)**, this module adds a **3D structural representation** by computing SOAP descriptors that describe each atom’s local spatial environment.  
These SOAP embeddings are later **combined (concatenated)** with SRR-DDI’s graph-based features to create a unified molecular representation for **drug–drug interaction (DDI) prediction**.

---

## 📁 Files Included

| File | Description |
|------|--------------|
| `non_PCA.py` | Generates per-molecule SOAP descriptors from SMILES using RDKit and DScribe. Outputs `soap_tokens.pkl` containing padded `(T_max × D_soap)` matrices. |
| `soap_io.py` | Handles loading of SOAP tokens and generation of boolean masks for padded entries. Used during dataset preprocessing. |
| `soap_encoder.py` | Defines the `SoapEncoder` class — a neural network that converts the SOAP token matrix into a fixed **256-dimensional molecular embedding**. |

---

## ⚙️ Integration Guide

### **Step 1 – Place the files**
Copy the following three files into your **SRR-DDI project directory** (for example, `SRR-DDI/src/`):
```
non_PCA.py
soap_io.py
soap_encoder.py
```

---

### **Step 2 – Generate SOAP descriptors**
Before training SRR-DDI, create the SOAP token file:
```bash
python non_PCA.py   --input-csv data/smiles.csv   --out-pkl data/soap_tokens.pkl   --r-cut 8.0 --n-max 8 --l-max 8 --sigma 0.4
```
This produces `data/soap_tokens.pkl`, where each entry corresponds to a molecule’s SOAP matrix.

---

### **Step 3 – Attach SOAP features to each drug**
In SRR-DDI’s data preprocessing component (`data_pre.py` or `dataset.py`),  
load and attach SOAP tensors to each molecular graph:
```python
from soap_io import load_soap_tokens, tokens_to_tensor, make_mask

soap_dict, T_max, D_soap = load_soap_tokens("data/soap_tokens.pkl")
graph.soap_tokens = tokens_to_tensor(soap_dict[drug_id]).unsqueeze(0)
graph.soap_mask = make_mask(graph.soap_tokens[0]).unsqueeze(0)
```
This ensures that every drug graph carries its SOAP matrix and mask during batching.

---

### **Step 4 – Combine SOAP with graph features**
In SRR-DDI’s model file (e.g., `model.py`), import and instantiate the encoder:
```python
from soap_encoder import SoapEncoder
self.soap_encoder = SoapEncoder(token_in_dim=soap_in_dim)
```

Then, in the fusion section of the model (where GNN embeddings are pooled),  
concatenate the SOAP embedding `s` with the graph embedding `d_g`:
```python
s = self.soap_encoder(soap_tokens, soap_mask)
d_g = torch.cat([
    global_max_pool(sub, batch),
    global_mean_pool(sub, batch)
], dim=-1)
fused = torch.cat([d_g, s], dim=-1)
```
This fused vector represents both **2D topological** and **3D geometric** molecular features.

---

## 🔬 Workflow Summary

```
SMILES → 3D geometry (RDKit/ASE)
      → SOAP descriptors (non_PCA.py)
      → SOAP tokens + mask (soap_io.py)
      → Encoded 3D embedding (soap_encoder.py)
      → Concatenation with SRR-DDI GNN features
      → DDI prediction
```

---

##  Notes
- **Best-performing SOAP configuration:** `l_max = 8`, `n_max = 8`, `r_cut = 8`, `σ = 0.4`  
- The SOAP branch outputs a **256-dimensional embedding** that complements SRR-DDI’s GNN representation.  
- You can disable or ablate this branch in SRR-DDI via model options (if supported).

---

## 🧩 Dependencies
- Python ≥ 3.8  
- PyTorch ≥ 1.12  
- PyTorch-Geometric ≥ 2.3  
- DScribe ≥ 2.0  
- RDKit ≥ 2022.09  
- ASE ≥ 3.22  
- NumPy, pandas, tqdm  

---

## 📖 Citation
If you use this module, please cite the associated publication on  
**SOAP-optimized 3D molecular representation for DDI prediction**.
