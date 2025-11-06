import os
import pickle
import random
import math
from collections import deque
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, \
    average_precision_score
from tqdm import tqdm
# Import the SOAP descriptor class from the separate file
from get_soap import SOAPDescriptorGenerator
# Import required chemistry packages (if needed elsewhere)
#results:
#Test ACC: 0.8992 | Test AUROC: 0.9391 | Test F1: 0.9039 | Test precision: 0.8632 | Test recall: 0.9487 | Test aupr: 0.9102
#Test ACC: 0.9029 | Test AUROC: 0.9434 | Test F1: 0.9075 | Test precision: 0.8663 | Test recall: 0.9528 | Test aupr: 0.9176

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
from ase import Atoms
#history = deque(maxlen=5)
#3rd run now
seed=20 #Final runs
dropout_rate=0.5
# Completed ones: First seed: 30 is A, 20 is C and 10 is B
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# -------------------- Dataset and Collate Function --------------------
class DDIDataset(Dataset):
    def __init__(self, df, soap_dict):
        #print("Inside DDIDataset class")
        self.drug1_ids = df.iloc[:, 0].astype(str).values
        self.drug2_ids = df.iloc[:, 1].astype(str).values
        self.labels = df.iloc[:, 2].values.astype(np.float32)
        self.soap_dict = soap_dict

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        drug1_id = self.drug1_ids[idx]
        drug2_id = self.drug2_ids[idx]
        if drug1_id not in self.soap_dict or drug2_id not in self.soap_dict:
            raise ValueError(f"SOAP descriptor missing for {drug1_id} or {drug2_id}")
        tokens1 = torch.tensor(self.soap_dict[drug1_id], dtype=torch.float32)
        tokens2 = torch.tensor(self.soap_dict[drug2_id], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tokens1, tokens2, label

def collate_fn(batch):
    #print("Inside collate_fn method")
    tokens1_list, tokens2_list, labels = zip(*batch)
    tokens1 = torch.stack(tokens1_list)
    tokens2 = torch.stack(tokens2_list)
    labels = torch.stack(labels)
    mask1 = (tokens1.abs().sum(dim=-1) != 0)
    mask2 = (tokens2.abs().sum(dim=-1) != 0)
    return tokens1.to(device), tokens2.to(device), labels.to(device), mask1.to(device), mask2.to(device)



# -------------------- Training and Evaluation Functions --------------------
def init_weights(m):
    #print("Inside init_weights metho")
    #print(f"Applying to: {m.__class__.__name__}")
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for drug1, drug2, labels, mask1, mask2 in tqdm(loader, desc="Evaluating"):
            outputs = model(drug1, drug2, mask1, mask2)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(np.float32)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, zero_division=1)
    precision = precision_score(all_labels, all_preds, zero_division=1)
    recall = recall_score(all_labels, all_preds, zero_division=1)
    aupr = average_precision_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    return acc, auc, f1, precision,recall,aupr

def train(model, train_loader, valid_loader, criterion, optimizer, epochs=200, patience=20):
    best_auc = 0.0
    no_improve_epochs = 0
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
        for drug1, drug2, labels, mask1, mask2 in progress_bar:
            optimizer.zero_grad()
            outputs = model(drug1, drug2, mask1, mask2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
        avg_loss = total_loss / len(train_loader)
        val_acc, val_auc, val_f1, val_pre, val_recall, val_apur = evaluate(model, valid_loader)
        print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} | Val ACC: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Pre: {val_pre:.4f} | Val recall: {val_recall:.4f} | Val aupr: {val_apur:.4f}")
        #val_acc, val_auc, val_f1, val_pre = evaluate(model, valid_loader)
        #####print(f"Epoch {epoch + 1} - Loss: {avg_loss:.4f} | Val ACC: {val_acc:.4f} | Val AUC: {val_auc:.4f} | Val F1: {val_f1:.4f} | Val Pre: {val_pre:.4f}")
        if val_acc > best_auc: # actually changed from auc to val_acc
            best_auc = val_acc # actually changes from auc to val_acc
            no_improve_epochs = 0
            torch.save(model.state_dict(), best_model_path)
            #####print(f"✅ Best model saved for current candidate with AUC: {best_auc:.4f}")
        else:
            no_improve_epochs += 1
        if no_improve_epochs >= patience:
            print("Early stopping triggered after no improvement in the last {} epochs.".format(patience))
            break
    return best_auc # this is changed to accuracy above. note that AUC in many places are actually acc.

# -------------------- Global Data Loading and Setup --------------------
# CSV files should have columns: [drug1_id, drug2_id, label]
train_df = pd.read_csv("data/trainDB.csv")
valid_df = pd.read_csv("data/validDB.csv")
test_df = pd.read_csv("data/testDB.csv")

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#####print(f"Using device: {device}")
#####print(torch.cuda.get_device_name(device))

# Default SOAP parameters (controlled from the main model code)
soap_params = {
    "input_file":  "data/smiles6.csv",
    "output_pickle_file": "data/soap_tokens.pkl",
    "error_log_file": "data/error_log.txt",
    "exclude_atoms": ['H'],  # Example: exclude Hydrogen
    "r_cut": 6.0,
    "n_max": 6,
    "l_max": 6,
    "rbf": "gto",
    "sigma": 0.7,
    "periodic": False,
    "compression": {'mode': 'mu2'}
    #"pca_dim": 1024
}

#Below is the actual code of soap_params
# soap_params = {
#     "input_file": "data/smiles6.csv",
#     "output_pickle_file": "data/soap_tokens.pkl",
#     "error_log_file": "data/error_log.txt",
#     "exclude_atoms": ['H'],  # Example: exclude Hydrogen
#     "r_cut": 8.0,
#     "n_max": 8,
#     "l_max": 8,
#     "rbf": "gto",
#     "sigma": 0.4,
#     "periodic": False,
#     "compression": {'mode': 'mu2'}
#     #"pca_dim": 1024
# }
#NEW: Test ACC: 0.9303 | Test AUROC: 0.9805 | Test F1: 0.9294 | Test precision: 0.9309 | Test recall: 0.9280 | Test aupr: 0.9810
#Test ACC: 0.9319 | Test AUROC: 0.9815 | Test F1: 0.9311 | Test precision: 0.9329 | Test recall: 0.9292 | Test aupr: 0.9821
best_model_path = "best_local_single_run.pth"
best_global_model_path = "best_global_single_run.pth"
batch_size = 1024 #1024
print("setting Loss function")
criterion = nn.BCEWithLogitsLoss()

# -------------------- Simulated Annealing with Sliding Windows --------------------
def sa_optimization():
    # SA hyperparameters
    T = 1.0
    T_min = 0.001
    cooling_rate = 0.95
    max_SA_iterations =1
    early_stopping_patience = 20

    global_best_auc = 0.0
    global_best_params = None
    global_best_history = []  # Store the best 30 parameter combinations

    # Initialize current parameters (only tuning these four)
    current_params = {
        'l_max': soap_params['l_max'],
        'n_max': soap_params['n_max'],
        'r_cut': soap_params['r_cut'],
        'sigma': soap_params['sigma']
    }

    current_auc = 0.0

    # Define step sizes and parameter bounds
    param_steps = {'l_max': 1, 'n_max': 1, 'r_cut': 0.5, 'sigma': 0.05}
    param_bounds = {
        'l_max': (3, 13),
        'n_max': (3, 13),
        'r_cut': (3.0, 10.0),
        'sigma': (0.1, 1.0)
    }

    # Initialize sliding windows for each parameter and direction
    sliding_windows = {
        'l_max': {'increase': [1], 'decrease': [1]},
        'n_max': {'increase': [1], 'decrease': [1]},
        'r_cut': {'increase': [1], 'decrease': [1]},
        'sigma': {'increase': [1], 'decrease': [1]}
    }

    sa_history = []  # To keep last 10 SA iterations

    for sa_iter in range(max_SA_iterations):
        #print("------------------------------------------------first iteration")
        print(f"\n--------------------- SA Iteration {sa_iter + 1} ---------------------")
        #####print(f"Current Temperature: {T:.4f}")
        #####print(f"Current Parameters: {current_params}, Current AUC: {current_auc:.4f}")

        #####print("cleaned")

        # Decide which parameter and direction to perturb based on sliding windows (oppostite to random permutation)
        candidate_scores = {}
        for param in current_params:
            lower_bound, upper_bound = param_bounds[param]
            for direction in ['increase', 'decrease']:
                if direction == 'increase' and current_params[param] >= upper_bound:
                    #avg_score=np.random.rand()
                    avg_score = 0.0
                elif direction == 'decrease' and current_params[param] <= lower_bound:
                    #avg_score=np.random.rand()
                    avg_score = 0.0
                else:
                    avg_score = np.mean(sliding_windows[param][direction])
                candidate_scores[(param, direction)] = avg_score
        #
        # This gives the list of candidate score i.e. average/mean
        #####print("Sliding Window :  ",sliding_windows)

        #####print("Candidate Scores (Parameter, Direction):")
        #####for k, v in candidate_scores.items():
            #####print(f"  {k}: {v:.4f}")
        max_score = max(candidate_scores.values())
        best_candidates = [k for k, v in candidate_scores.items() if v == max_score]
        chosen_param, chosen_direction = random.choice(best_candidates)
        #print(f"Chosen for perturbation: Parameter '{chosen_param}' in '{chosen_direction}' direction (score: {max_score:.4f})")

        #Add the chosen selection to history #not needed
        # history.append((chosen_param, chosen_direction))
        # if len(history) == 5 and len(set(history)) == 1:
        #     min_score = min(candidate_scores.values())
        #     best_candidates = [k for k, v in candidate_scores.items() if v == min_score]
        #     chosen_param, chosen_direction = random.choice(best_candidates)
        # print(history)
        # print("ABOVE IS HISTORY DEQUEUE")

        #Propose new candidate value
        new_candidate_params = current_params.copy()

        step = param_steps[chosen_param]
        if chosen_direction == 'increase':
            #new_candidate_params[chosen_param] += step
            print()
        else:
            #new_candidate_params[chosen_param] -= step
            print()
        # Clamp candidate value to its bounds
        lower_bound, upper_bound = param_bounds[chosen_param]

        new_candidate_params[chosen_param] = max(lower_bound, min(new_candidate_params[chosen_param], upper_bound))
        # if new_candidate_params[chosen_param] == upper_bound and chosen_direction == 'increase':
        #     print(f"Note: {chosen_param} reached its upper bound.")
        # if new_candidate_params[chosen_param] == lower_bound and chosen_direction == 'decrease':
        #     print(f"Note: {chosen_param} reached its lower bound.")
        #####print(f"Proposed New Parameters: {new_candidate_params}")
        # Update soap_params with candidate values
        soap_params['l_max'] = new_candidate_params['l_max']
        soap_params['n_max'] = new_candidate_params['n_max']
        soap_params['r_cut'] = new_candidate_params['r_cut']
        soap_params['sigma'] = new_candidate_params['sigma']

        # Create an instance of the SOAPDescriptorGenerator using current parameters
        soap_generator_instance = SOAPDescriptorGenerator(
            exclude_atoms=soap_params['exclude_atoms'],
            r_cut=soap_params['r_cut'],
            n_max=soap_params['n_max'],
            l_max=soap_params['l_max'],
            rbf=soap_params['rbf'],
            sigma=soap_params['sigma'],
            periodic=soap_params['periodic'],
            compression=soap_params['compression']
            #pca_dim = soap_params['pca_dim']
        )
        # Generate SOAP descriptors using candidate parameters
        soap_dict, species = soap_generator_instance.generate_descriptors(
            input_file=soap_params['input_file'],
            output_pickle_file=soap_params['output_pickle_file'],
            error_log_file=soap_params['error_log_file']
        )
        # Here we take the dimensions of from the SOAP shape
        any_drug = next(iter(soap_dict.values()))
        num_tokens, token_in_dim = any_drug.shape
        token_out_dim = token_in_dim // 2 if token_in_dim > 1 else token_in_dim
        flattened_dim = num_tokens * (token_out_dim // 4)
        fc_hidden_dim = flattened_dim // 2

        # Completed ones: First seed: 30, 20
        #random.seed(seed)
        #np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        # Initialize model and optimizer for candidate
        model = DDI_Predictor(token_in_dim, token_out_dim, num_tokens, fc_hidden_dim).to(device)
        #print(model)

        model.apply(init_weights)

        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00003) #lr=0.001, weight_decay=0.00001)
        #optimizer = optim.Adam(model.parameters(), lr=0.001)
        import torch_optimizer as optim_advanced
        #print("Create DataLoaders ")
        # Create DataLoaders
        train_loader = DataLoader(DDIDataset(train_df, soap_dict), batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(DDIDataset(valid_df, soap_dict), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(DDIDataset(test_df, soap_dict), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        # Train candidate model
        candidate_best_auc = train(model, train_loader, valid_loader, criterion, optimizer, epochs=200, patience=early_stopping_patience)
        #####print(f"Candidate's Best Validation AUC: {candidate_best_auc:.4f}")

        # Update sliding window for chosen parameter/direction
        window = sliding_windows[chosen_param][chosen_direction]
        if len(window) < 1: #keep window size 1,3,5 etc
            window.append(candidate_best_auc)
        else:
            window.pop(0)
            window.append(candidate_best_auc)
        updated_avg = np.mean(window)
        #print(f"Updated sliding window for '{chosen_param}' ({chosen_direction}): {window} -> Average: {updated_avg:.4f}")

        # SA Acceptance Criterion
        delta = candidate_best_auc - current_auc
        if delta >= 0:
            accept = True
            #print("Candidate improves or equals the current AUC. Accepting candidate.")
        else:
            accept_prob = math.exp(delta / T)
            r=random.random()
            accept = (r < accept_prob)
            #print(f"Candidate is worse by Δ={delta:.4f}. Acceptance probability: {accept_prob:.4f}. Random draw yields {'accept' if accept else 'reject'}.")
        if accept:
            current_params = new_candidate_params.copy()
            current_auc = candidate_best_auc
            #print("Candidate accepted. Current parameters updated.")
        # else:
        #     print("Candidate rejected. Retaining previous parameters.")

        # Global Best Update
        if candidate_best_auc > global_best_auc:

            global_best_auc = candidate_best_auc
            global_best_params = current_params.copy()
            torch.save(model.state_dict(), best_global_model_path)

            #####print(">>> Global best updated!")
            global_best_history.append((global_best_auc, global_best_params.copy()))
            # we remove the oldest if want to keep the list of fized size. also note that oldest is the lowest score.
            #if len(global_best_history) > 250:
                #global_best_history.pop(0)
        else:
            if len(global_best_history) < 250:
                global_best_history.append((candidate_best_auc, new_candidate_params.copy()))
            # else:
            #     print("Candidate did not beat the global best.")

        # Record SA iteration history (last 10)
        sa_history.append((current_params.copy(), current_auc))
        if len(sa_history) > 250:
            sa_history.pop(0)


        # Cooling
        T *= cooling_rate
        #####print(f"Cooling: New Temperature = {T:.4f}")

        #reset the priority queue
        # if (sa_iter + 1) % 30 == 0:
        #     sliding_windows = {
        #         'l_max': {'increase': [1], 'decrease': [1]},
        #         'n_max': {'increase': [1], 'decrease': [1]},
        #         'r_cut': {'increase': [1], 'decrease': [1]},
        #         'sigma': {'increase': [1], 'decrease': [1]}
        #     }
            #####print("Sliding windows reset to initial values after 30 iterations.")

        if T < T_min:
            #####print("Temperature threshold reached. Terminating SA optimization.")
            break

    #####print("\n================== SA Optimization Completed ==================")
    #####print("Global Best Parameters:", global_best_params, "with AUC:", global_best_auc)
    #####print("\nLast 200 Global Best Parameter Combinations and their Scores:")
    for idx, (score, params) in enumerate(global_best_history):
        print(f"  {idx + 1}. Score: {score:.4f}, Parameters: {params}")

    # Final Test Evaluation with Global Best Model
    #####print("\nRunning final evaluation on Test Set using the Global Best Model...")
    best_model = DDI_Predictor(token_in_dim, token_out_dim, num_tokens, fc_hidden_dim).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device))
    best_model.to(device)
    # acc, auroc, f1, precision,recall,aupr
    test_acc, test_auc, test_f1, test_pre, test_recall, test_aupr = evaluate(best_model, test_loader)
    #####print("\n================== Final Test Set Evaluation ==================")
    print(f"Test ACC: {test_acc:.4f} | Test AUROC: {test_auc:.4f} | Test F1: {test_f1:.4f} | Test precision: {test_pre:.4f} | Test recall: {test_recall:.4f} | Test aupr: {test_aupr:.4f}")
    #####print("\nGlobal Best History (Last 200):")
    for idx, (score, params) in enumerate(global_best_history):
        print(f"  {idx + 1}. Score: {score:.4f}, Parameters: {params}")
     #save to txt file the list of SOAP optimziations
    # Define the file name
    file_name = "list_single_run_DS2_dataset.txt"

    # Save the global_best_history into the file
    with open(file_name, "w") as f:
        for idx, (score, params) in enumerate(global_best_history):
            f.write(f"{idx + 1}. Score: {score:.4f}, Parameters: {params}\n")

    #####print(f"Saved global_best_history to {file_name}")


# -------------------- Run SA Optimization --------------------
if __name__ == "__main__":
    print("model_1_priority_reset_SA---Before close _FINAL")
    sa_optimization()

