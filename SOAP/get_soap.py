import os
import pickle
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from dscribe.descriptors import SOAP
from ase import Atoms
from tqdm import tqdm

class SOAPDescriptorGenerator:
    def __init__(self, exclude_atoms, r_cut, n_max, l_max, rbf, sigma, periodic, compression):
        """
        Initialize the SOAP descriptor generator with the given parameters.
        """
        self.exclude_atoms = exclude_atoms
        self.r_cut = r_cut
        self.n_max = n_max
        self.l_max = l_max
        self.rbf = rbf
        self.sigma = sigma
        self.periodic = periodic
        self.compression = compression

    def generate_descriptors(self, input_file, output_pickle_file, error_log_file):
        # Suppress RDKit warnings
        RDLogger.DisableLog('rdApp.*')

        # Attempt to load fallback conformations from file, if available
        conform_file = "data/conformtion.pkl"
        fallback_conformations = {}
        if os.path.exists(conform_file):
            with open(conform_file, "rb") as cf:
                fallback_conformations = pickle.load(cf)
            #print(f"Loaded fallback conformations from: {conform_file}")
        else:
            print(f"No existing conformations file found at: {conform_file}")

        # Load the SMILES CSV file
        df = pd.read_csv(input_file)

        # Extract unique atomic species (excluding defined atoms)
        unique_elements = set()
        for smiles in df["SMILES"]:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mol = Chem.AddHs(mol)
                for atom in mol.GetAtoms():
                    if atom.GetSymbol() not in self.exclude_atoms:
                        unique_elements.add(atom.GetSymbol())
        species = sorted(list(unique_elements))

        # Initialize the SOAP descriptor generator with the species list and parameters
        self.soap_generator = SOAP(
            species=species,
            periodic=self.periodic,
            r_cut=self.r_cut,
            n_max=self.n_max,
            l_max=self.l_max,
            rbf=self.rbf,
            sigma=self.sigma,
            average="off",
            sparse=False,
            compression=self.compression
        )

        failed_smiles = []
        soap_data = {}
        # This dictionary will collect conformations generated during this run.
        new_conformations = {}

        # Define a helper function to process a single SMILES string.
        def smiles_to_soap(smiles, drug_id):
            try:
                mol = Chem.MolFromSmiles(smiles)

                if mol is None:
                    raise ValueError("Invalid SMILES format")
                #result = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
                #below is to remove randomness
                mol = Chem.AddHs(mol)

                params = AllChem.ETKDG()
                params.randomSeed = 30
                result = AllChem.EmbedMolecule(mol, params)

                if result == -1:
                    raise ValueError("3D embedding failed")
                opt_result = AllChem.UFFOptimizeMolecule(mol)
                if opt_result == -1:
                    raise ValueError("UFF optimization failed")
                atoms = []
                positions = []
                for atom in mol.GetAtoms():
                    symbol = atom.GetSymbol()
                    pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
                    if symbol not in self.exclude_atoms:
                        atoms.append(symbol)
                        positions.append([pos.x, pos.y, pos.z])
                if not atoms:
                    raise ValueError("All atoms were excluded")
                # Save the new conformation
                new_conformations[drug_id] = {"atoms": atoms, "positions": positions}
                ase_molecule = Atoms(symbols=atoms, positions=positions)
                soap_desc = self.soap_generator.create(ase_molecule)
                return soap_desc
            except Exception as e:
                # Attempt fallback if available.
                if fallback_conformations and drug_id in fallback_conformations:
                    fb_data = fallback_conformations[drug_id]
                    #####print(f"Using fallback conformation for {drug_id} due to error: {e}")
                    ase_molecule = Atoms(symbols=fb_data["atoms"], positions=fb_data["positions"])
                    soap_desc = self.soap_generator.create(ase_molecule)
                    return soap_desc
                else:
                    failed_smiles.append(f"{smiles} -> {e}")
                    return None

        # Process each SMILES entry and build SOAP descriptors along with conformations.
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing SMILES"):
            drug_id = row["DrugBank ID"]
            smiles = row["SMILES"]
            soap_descriptor = smiles_to_soap(smiles, drug_id)
            if soap_descriptor is not None:
                soap_data[drug_id] = soap_descriptor

        # Find the maximum number of tokens across descriptors
        if soap_data:
            max_tokens_id = max(soap_data, key=lambda k: soap_data[k].shape[0])
            max_tokens = soap_data[max_tokens_id].shape[0]
            soap_dim = soap_data[max_tokens_id].shape[1]
            # print(f"DrugBank ID with the highest number of tokens: {max_tokens_id} ({max_tokens} tokens)")
            print(f"DrugBank ID with the highest number of tokens: {max_tokens_id} ({max_tokens} tokens, {soap_dim}-dimensional descriptors)")

            # Pad all SOAP descriptors to have the same number of tokens
            for drug_id in soap_data:
                num_tokens = soap_data[drug_id].shape[0]
                if num_tokens < max_tokens:
                    pad_size = max_tokens - num_tokens
                    padding = np.zeros((pad_size, soap_dim))
                    soap_data[drug_id] = np.vstack([soap_data[drug_id], padding])
        else:
            print("No SOAP descriptors were generated.")

        # Save the SOAP descriptor dictionary
        with open(output_pickle_file, "wb") as f:
            pickle.dump(soap_data, f)
        if failed_smiles:
            with open(error_log_file, "w") as f:
                f.write("\n".join(failed_smiles))
        # print(f"SOAP descriptor pickle file saved to: {output_pickle_file}")
        # print(f"Failed SMILES count: {len(failed_smiles)} (see {error_log_file})")
        #-----------print(f"Extracted Species (excluding {self.exclude_atoms}): {species}")


        #print just for debugging
        # Print SOAP descriptor for a specific DrugBank ID
        # Convert to list to avoid truncation
        # full_matrix = soap_data["DB00945"].tolist()
        # # Print the full matrix
        # for row in full_matrix:
        #     print(row)
        # exit()
        # Save the newly generated conformations if the file does not exist.
        if not os.path.exists(conform_file) and new_conformations:
            with open(conform_file, "wb") as cf:
                pickle.dump(new_conformations, cf)
            #print(f"New conformations saved to: {conform_file}")
        # else:
        #     if os.path.exists(conform_file):
        #         print(f"Conformations file '{conform_file}' already exists. Skipping save of new conformations.")
        #     else:
        #         print("No new conformations to save.")

        return soap_data, species
