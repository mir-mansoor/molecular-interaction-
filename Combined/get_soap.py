#!/usr/bin/env python3
"""
SOAP_noPCA.py — Generate SOAP descriptors, pad to max tokens, and save.

Usage (defaults match your repo):
    python SOAP_noPCA.py \
      --input-csv data/smiles6.csv \
      --out-pkl  data/soap_tokens.pkl \
      --err-log  data/error_log.txt \
      --conform-pkl data/conformtion.pkl \
      --exclude-atoms "" \
      --r-cut 8.0 --n-max 8 --l-max 8 --rbf gto --sigma 0.4 \
      --periodic 0 --compression mu1nu1 --seed 30

Notes:
- Expects CSV with columns: "DrugBank ID" and "SMILES".
- No PCA is applied; SOAP descriptors are padded to the max token length observed.
"""

import os
import sys
import json
import pickle
import argparse
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

from dscribe.descriptors import SOAP
from ase import Atoms


def ensure_dir(path: str):
    d = os.path.dirname(os.path.abspath(path))
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def parse_exclude_atoms(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [tok.strip() for tok in s.split(",") if tok.strip()]


class SOAPDescriptorGenerator:
    """
    Generates SOAP descriptors for molecules given SMILES,
    then pads per-drug matrices to a common (T_max, soap_dim) shape and saves them.
    """

    def __init__(
        self,
        exclude_atoms: List[str],
        r_cut: float,
        n_max: int,
        l_max: int,
        rbf: str,
        sigma: float,
        periodic: bool,
        compression: Optional[str],
        seed: int = 30,
        conformations_pkl: Optional[str] = None,
    ):
        self.exclude_atoms = exclude_atoms
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.rbf = rbf
        self.sigma = sigma
        self.periodic = periodic
        self.compression = None if (compression in [None, "", "none", "None"]) else {"mode": compression}
        self.seed = seed
        self.conformations_pkl = conformations_pkl

        # quiet RDKit
        RDLogger.DisableLog("rdApp.*")

        # rdkit ETKDG seed
        self._embed_params = AllChem.ETKDG()
        self._embed_params.randomSeed = int(seed)

        # fallback conformations cache
        self._fallback_conformations = {}
        if self.conformations_pkl and os.path.exists(self.conformations_pkl):
            try:
                with open(self.conformations_pkl, "rb") as cf:
                    self._fallback_conformations = pickle.load(cf)
            except Exception:
                print(f"[WARN] Could not read fallback conformations: {self.conformations_pkl}", file=sys.stderr)

        self.soap = None  # created after species discovery

    def _discover_species(self, smiles_series: pd.Series) -> List[str]:
        species = set()
        for smi in smiles_series:
            mol = Chem.MolFromSmiles(smi)
            if not mol:
                continue
            mol = Chem.AddHs(mol)
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in self.exclude_atoms:
                    species.add(sym)
        return sorted(species)

    def _init_soap(self, species: List[str]):
        self.soap = SOAP(
            species=species,
            periodic=self.periodic,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            rbf=self.rbf,
            sigma=self.sigma,
            average="off",
            sparse=False,
            compression=self.compression,
        )

    def _smiles_to_ase(self, smiles: str, drug_id: str) -> Optional[Atoms]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # build 3D
        res = AllChem.EmbedMolecule(mol, self._embed_params)
        if res == -1:
            # fallback to cached conformations (if present)
            if self._fallback_conformations and drug_id in self._fallback_conformations:
                fb = self._fallback_conformations[drug_id]
                return Atoms(symbols=fb["atoms"], positions=fb["positions"])
            return None

        _ = AllChem.UFFOptimizeMolecule(mol)
        atoms = []
        positions = []
        conf = mol.GetConformer()
        for a in mol.GetAtoms():
            sym = a.GetSymbol()
            if sym in self.exclude_atoms:
                continue
            pos = conf.GetAtomPosition(a.GetIdx())
            atoms.append(sym)
            positions.append([pos.x, pos.y, pos.z])
        if not atoms:
            return None
        return Atoms(symbols=atoms, positions=positions)

    def generate_and_save(
        self,
        input_csv: str,
        output_pickle_file: str,
        error_log_file: Optional[str] = None,
        save_species_json: Optional[str] = None,
        save_meta_json: Optional[str] = None,
    ) -> Tuple[Dict[str, np.ndarray], List[str]]:
        """
        Returns (soap_dict, species). Also writes pickle and logs.
        """
        ensure_dir(output_pickle_file)
        if error_log_file:
            ensure_dir(error_log_file)
        if save_species_json:
            ensure_dir(save_species_json)
        if save_meta_json:
            ensure_dir(save_meta_json)

        # read input
        df = pd.read_csv(input_csv)
        if "DrugBank ID" not in df.columns or "SMILES" not in df.columns:
            raise ValueError('Input CSV must have columns "DrugBank ID" and "SMILES".')

        # species & SOAP init
        species = self._discover_species(df["SMILES"])
        if not species:
            raise RuntimeError("No species discovered; check exclude-atoms or input CSV.")
        self._init_soap(species)

        # SOAP per drug (UNPADDED)
        failed = []
        soap_data: Dict[str, np.ndarray] = {}
        new_conformations = {}

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating SOAP"):
            drug_id = str(row["DrugBank ID"])
            smi = row["SMILES"]
            try:
                ase_mol = self._smiles_to_ase(smi, drug_id)
                if ase_mol is None:
                    raise ValueError("3D embedding failed (and no fallback).")
                desc = self.soap.create(ase_mol)  # (tokens_k, soap_dim)
                soap_data[drug_id] = desc
                # Remember created conformations for potential reuse
                if self.conformations_pkl:
                    new_conformations[drug_id] = {
                        "atoms": ase_mol.get_chemical_symbols(),
                        "positions": ase_mol.get_positions().tolist(),
                    }
            except Exception as e:
                failed.append(f"{drug_id}\t{smi}\t{e}")

        if not soap_data:
            print("[WARN] No SOAP descriptors were generated; saving empty dict.")
            with open(output_pickle_file, "wb") as f:
                pickle.dump(soap_data, f)
            if error_log_file and failed:
                with open(error_log_file, "w") as f:
                    f.write("\n".join(failed))
            return soap_data, species

        # report max tokens and raw dim
        max_tokens_id = max(soap_data, key=lambda k: soap_data[k].shape[0])
        max_tokens = soap_data[max_tokens_id].shape[0]
        soap_dim = soap_data[max_tokens_id].shape[1]
        print("HERE")
        print(f"[INFO] Max tokens: {max_tokens} (drug {max_tokens_id}), raw SOAP dim: {soap_dim}")


        # ---------- PAD to MAX TOKENS ----------
        for k in list(soap_data.keys()):
            n_tok = soap_data[k].shape[0]
            if n_tok < max_tokens:
                pad = np.zeros((max_tokens - n_tok, soap_dim), dtype=soap_data[k].dtype)
                soap_data[k] = np.vstack([soap_data[k], pad])

        # save descriptors
        with open(output_pickle_file, "wb") as f:
            pickle.dump(soap_data, f)
        print(f"[OK] Saved SOAP descriptors to: {output_pickle_file}")

        # save species (optional)
        if save_species_json:
            with open(save_species_json, "w") as f:
                json.dump(species, f, indent=2)
            print(f"[OK] Saved species to: {save_species_json}")

        # save meta (optional)
        if save_meta_json:
            meta = {
                "max_tokens": max_tokens,
                "raw_soap_dim": soap_dim,
                "exclude_atoms": self.exclude_atoms,
                "params": {
                    "r_cut": self.r_cut, "n_max": self.n_max, "l_max": self.l_max,
                    "rbf": self.rbf, "sigma": self.sigma,
                    "periodic": self.periodic, "compression": self.compression,
                    "seed": self.seed,
                },
                "input_csv": os.path.abspath(input_csv),
                "output_pickle_file": os.path.abspath(output_pickle_file),
            }
            with open(save_meta_json, "w") as f:
                json.dump(meta, f, indent=2)
            print(f"[OK] Saved meta to: {save_meta_json}")

        # save/merge conformations cache (optional)
        if self.conformations_pkl and new_conformations:
            try:
                merged = dict(self._fallback_conformations)
                merged.update(new_conformations)
                ensure_dir(self.conformations_pkl)
                with open(self.conformations_pkl, "wb") as cf:
                    pickle.dump(merged, cf)
                print(f"[OK] Saved conformations cache to: {self.conformations_pkl}")
            except Exception as e:
                print(f"[WARN] Could not save conformations cache: {e}", file=sys.stderr)

        # write failures
        if error_log_file and failed:
            with open(error_log_file, "w") as f:
                f.write("\n".join(failed))
            print(f"[INFO] Logged {len(failed)} failures to: {error_log_file}")

        return soap_data, species


def main():
    ap = argparse.ArgumentParser(description="Generate SOAP descriptors (no PCA) and save.")
    ap.add_argument("--input-csv", default="data/smiles6.csv", help="CSV with columns: DrugBank ID, SMILES")
    ap.add_argument("--out-pkl", default="data/soap_tokens.pkl", help="Output pickle path for SOAP descriptors")
    ap.add_argument("--err-log", default="data/error_log.txt", help="Where to write failures log")
    ap.add_argument("--conform-pkl", default="data/conformtion.pkl",
                    help="Cache for conformations (created if missing)")

    ap.add_argument("--exclude-atoms", default="", help="Comma-separated list to exclude (e.g., 'H')")
    ap.add_argument("--r-cut", type=float, default=8.0)
    ap.add_argument("--n-max", type=int, default=8)
    ap.add_argument("--l-max", type=int, default=8)
    ap.add_argument("--rbf", choices=["gto", "polynomial"], default="gto")
    ap.add_argument("--sigma", type=float, default=0.4)
    ap.add_argument("--periodic", type=int, default=0, help="0/1")
    ap.add_argument("--compression", default="mu2", help="'none' to disable, or dscribe mode like 'mu1nu1'")
    ap.add_argument("--seed", type=int, default=20)
    ap.add_argument("--species-json", default="", help="Optional: save species list JSON")
    ap.add_argument("--meta-json", default="", help="Optional: save meta JSON")

    args = ap.parse_args()

    exclude_atoms = parse_exclude_atoms(args.exclude_atoms)
    compression = None if str(args.compression).lower() in ["none", "null", ""] else args.compression
    periodic_bool = bool(int(args.periodic))

    gen = SOAPDescriptorGenerator(
        exclude_atoms=exclude_atoms,
        r_cut=args.r_cut,
        n_max=args.n_max,
        l_max=args.l_max,
        rbf=args.rbf,
        sigma=args.sigma,
        periodic=periodic_bool,
        compression=compression,
        seed=args.seed,
        conformations_pkl=args.conform_pkl,
    )

    soap_dict, species = gen.generate_and_save(
        input_csv=args.input_csv,
        output_pickle_file=args.out_pkl,
        error_log_file=args.err_log,
        save_species_json=(args.species_json or None),
        save_meta_json=(args.meta_json or None),
    )

    print(f"[DONE] {len(soap_dict)} drugs processed; example shape: {next(iter(soap_dict.values())).shape}")


if __name__ == "__main__":
    main()
