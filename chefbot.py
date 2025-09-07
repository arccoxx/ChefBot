import numpy as np
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
import os
import urllib.request
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import copy

# ============================================================================
# 0. Configuration and Setup
# ============================================================================

SMOKE_TEST = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# --- Data Loading ---
DATASET_FILE = "simplified-recipes-1M.npz"
if not os.path.exists(DATASET_FILE):
    print("Downloading dataset..."); urllib.request.urlretrieve("https://github.com/schmidtdominik/RecipeNet/raw/master/simplified-recipes-1M.npz", DATASET_FILE)
simplified_recipes_data = np.load(DATASET_FILE, allow_pickle=True)
recipes, ingredients = simplified_recipes_data['recipes'], simplified_recipes_data['ingredients']
N_INGREDIENTS = len(ingredients)
print(f"Loaded {len(recipes)} recipes with {N_INGREDIENTS} unique ingredients.")

# --- Dataset Classes ---
class GenerativeRecipeDataset(torch.utils.data.Dataset):
    def __init__(self, recipes_array, n_total_ingredients):
        self.recipes = recipes_array; self.n_ingredients = n_total_ingredients
    def __getitem__(self, idx):
        recipe_vector = np.zeros(self.n_ingredients, dtype=np.float32)
        recipe_indices = self.recipes[idx].astype(np.int64)
        recipe_vector[recipe_indices] = 1.0
        return torch.from_numpy(recipe_vector)
    def __len__(self): return len(self.recipes)

class FineTuningRecipeDataset(torch.utils.data.Dataset):
    def __init__(self, recipes_array, n_total_ingredients, target_idx):
        self.recipes = recipes_array; self.n_ingredients = n_total_ingredients; self.target_idx = target_idx
    def __getitem__(self, idx):
        recipe_indices = self.recipes[idx].astype(np.int64)
        recipe_vector = np.zeros(self.n_ingredients, dtype=np.float32)
        recipe_vector[recipe_indices] = 1.0; 
        label = torch.tensor(recipe_vector[self.target_idx], dtype=torch.float32)
        recipe_vector[self.target_idx] = 0.0
        return torch.from_numpy(recipe_vector), label
    def __len__(self): return len(self.recipes)

# --- Model Definitions ---
class RBM(nn.Module):
    def __init__(self, n_vis, n_hid):
        super(RBM, self).__init__(); self.W = nn.Parameter(torch.randn(n_vis, n_hid, device=DEVICE) * 0.01); self.v_bias = nn.Parameter(torch.zeros(n_vis, device=DEVICE)); self.h_bias = nn.Parameter(torch.zeros(n_hid, device=DEVICE))
    def energy(self, v, h): v_term = torch.einsum('...v,v->...', v, self.v_bias); h_term = torch.einsum('...h,h->...', h, self.h_bias); w_term = torch.sum(torch.matmul(v, self.W) * h, dim=-1); return -(v_term + h_term + w_term)
    def sample_h(self, v): h_prob = torch.sigmoid(torch.matmul(v, self.W) + self.h_bias); return (torch.rand_like(h_prob) < h_prob).float()
    def free_energy(self, v): return -torch.sum(self.v_bias * v, dim=-1) - torch.sum(nn.functional.softplus(torch.matmul(v, self.W) + self.h_bias), dim=-1)

class DBN(nn.Module):
    def __init__(self, rbm_layers):
        super(DBN, self).__init__(); self.rbm_layers = nn.ModuleList(rbm_layers)

# ============================================================================
# 2. ACA-like Solver and ACA-Enhanced AIS Evaluator
# ============================================================================

class ParallelTemperingRBMSolver:
    def __init__(self, rbm_model, batch_size, num_replicas=48, T_min=0.1, T_max=2.0):
        self.rbm=rbm_model; self.num_replicas=num_replicas; self.temps=torch.logspace(np.log10(T_min),np.log10(T_max),num_replicas).to(DEVICE); self.reset_states(batch_size)
    def reset_states(self, batch_size): n_vis, n_hid = self.rbm.W.shape; self.v_replicas = (torch.rand(self.num_replicas, batch_size, n_vis, device=DEVICE) > 0.5).float(); self.h_replicas = (torch.rand(self.num_replicas, batch_size, n_hid, device=DEVICE) > 0.5).float()
    def solve(self, iterations=50):
        with torch.no_grad():
            for _ in range(iterations):
                self.h_replicas = self.rbm.sample_h(self.v_replicas)
                v_prob = torch.sigmoid(torch.einsum('rbh,hv->rbv',self.h_replicas,self.rbm.W.T)+self.rbm.v_bias); self.v_replicas=(torch.rand_like(v_prob)<v_prob).float()
                energies = torch.mean(self.rbm.energy(self.v_replicas, self.h_replicas), dim=1)
                E1, E2 = energies[:-1], energies[1:]; beta1, beta2 = 1.0/self.temps[:-1], 1.0/self.temps[1:]
                swap_mask = torch.rand(self.num_replicas - 1, device=DEVICE) < torch.exp((beta1 - beta2) * (E1 - E2))
                v_temp = self.v_replicas[:-1][swap_mask].clone(); self.v_replicas[:-1][swap_mask] = self.v_replicas[1:][swap_mask]; self.v_replicas[1:][swap_mask] = v_temp
                h_temp = self.h_replicas[:-1][swap_mask].clone(); self.h_replicas[:-1][swap_mask] = self.h_replicas[1:][swap_mask]; self.h_replicas[1:][swap_mask] = h_temp
        return self.v_replicas[0], self.h_replicas[0]

class AIS_for_RBM:
    def __init__(self, rbm_model, n_chains=50, n_steps=500):
        self.rbm = rbm_model; self.n_chains = n_chains; self.betas = torch.linspace(0.0, 1.0, n_steps).to(DEVICE); self.n_vis = rbm_model.W.shape[0]
    def _get_model_for_beta(self, beta):
        temp_rbm = RBM(self.rbm.W.shape[0], self.rbm.W.shape[1]).to(DEVICE)
        with torch.no_grad(): temp_rbm.W.data = self.rbm.W.data * beta; temp_rbm.v_bias.data = self.rbm.v_bias.data * beta; temp_rbm.h_bias.data = self.rbm.h_bias.data * beta
        return temp_rbm
    def estimate_log_z(self):
        log_z0 = self.n_vis * np.log(2.0); v = (torch.rand(self.n_chains, self.n_vis, device=DEVICE) > 0.5).float()
        log_weights = torch.zeros(self.n_chains, device=DEVICE)
        for i in tqdm(range(len(self.betas) - 1), desc="AIS Evaluation", leave=False):
            beta_curr, beta_next = self.betas[i], self.betas[i+1]
            with torch.no_grad():
                h = self.rbm.sample_h(v); energy = self.rbm.energy(v, h)
                log_weights += -(beta_next - beta_curr) * energy
                temp_model = self._get_model_for_beta(beta_next); solver = ParallelTemperingRBMSolver(temp_model, batch_size=self.n_chains, num_replicas=4)
                solver.v_replicas[0] = v; v, _ = solver.solve(iterations=5)
        log_z_ratio = torch.logsumexp(log_weights, dim=0) - np.log(self.n_chains)
        return log_z0 + log_z_ratio.item()

def calculate_rbm_log_likelihood(rbm, data_loader, log_z):
    total_free_energy = 0; num_samples = 0
    with torch.no_grad():
        for batch in data_loader:
            v = (batch[0] if isinstance(batch, list) else batch).to(DEVICE)
            total_free_energy += torch.sum(-rbm.free_energy(v)).item()
            num_samples += v.size(0)
    return (total_free_energy / num_samples) - log_z

# ============================================================================
# 3. Training and Evaluation Pipeline
# ============================================================================

def determine_dbn_architecture(n_input):
    layer_sizes = [2048, 1024, 512]
    valid_layers = [size for size in layer_sizes if size < n_input]
    return valid_layers

def main(is_smoke_test=False):
    # --- 1. Data Setup ---
    num_recipes = 2000 if is_smoke_test else len(recipes)
    train_indices_full, test_indices = train_test_split(np.arange(num_recipes), test_size=0.1, random_state=42)
    train_indices, val_indices = train_test_split(train_indices_full, test_size=0.1, random_state=42)
    
    # --- 2. DBN Architecture and Pre-training ---
    if is_smoke_test:
        dbn_hidden_sizes, max_epochs, patience, batch_size, lr_pretrain = [1024, 512], 2, 1, 256, 0.01
    else:
        dbn_hidden_sizes = determine_dbn_architecture(N_INGREDIENTS); max_epochs, patience, batch_size, lr_pretrain = 10, 2, 1024, 0.04
    
    print(f"\nDetermined DBN Architecture: {N_INGREDIENTS} -> {' -> '.join(map(str, dbn_hidden_sizes))}")
    trained_rbm_layers = []; current_input_dim = N_INGREDIENTS
    current_full_train_dataset = GenerativeRecipeDataset(recipes[train_indices_full], N_INGREDIENTS)

    for i, h_size in enumerate(dbn_hidden_sizes):
        print(f"\n--- Pre-training Layer {i+1} (RBM: {current_input_dim} -> {h_size}) ---")
        rbm = RBM(current_input_dim, h_size).to(DEVICE)
        sub_train_dataset, val_dataset = torch.utils.data.random_split(current_full_train_dataset, [len(train_indices), len(val_indices)], generator=torch.Generator().manual_seed(42))
        train_loader = torch.utils.data.DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size*2)
        solver = ParallelTemperingRBMSolver(rbm, batch_size=batch_size)
        best_log_likelihood = -np.inf; epochs_without_improvement = 0; best_state_dict = None

        for epoch in range(max_epochs):
            print(f"\nLayer {i+1} Epoch {epoch+1}/{max_epochs}"); rbm.train()
            for batch in tqdm(train_loader, desc="Training"):
                v_pos = (batch[0] if isinstance(batch, list) else batch).to(DEVICE)
                if v_pos.size(0) != solver.v_replicas.size(1): solver.reset_states(v_pos.size(0))
                h_pos_prob=torch.sigmoid(torch.matmul(v_pos,rbm.W)+rbm.h_bias);v_neg,_=solver.solve()
                with torch.no_grad():
                    pos_assoc=torch.matmul(v_pos.T,h_pos_prob);neg_assoc=torch.matmul(v_neg.T,torch.sigmoid(torch.matmul(v_neg,rbm.W)+rbm.h_bias))
                    # --- THIS IS THE FIX ---
                    # Use the lr_pretrain variable instead of a hardcoded value.
                    rbm.W+=lr_pretrain*(pos_assoc-neg_assoc)/v_pos.size(0); rbm.v_bias+=lr_pretrain*torch.mean(v_pos-v_neg,dim=0); rbm.h_bias+=lr_pretrain*torch.mean(h_pos_prob-torch.sigmoid(torch.matmul(v_neg,rbm.W)+rbm.h_bias),dim=0)
            
            rbm.eval(); print("  Evaluating model fit with AIS...")
            ais_evaluator = AIS_for_RBM(rbm, n_chains=20, n_steps=200)
            log_z = ais_evaluator.estimate_log_z()
            if is_smoke_test: assert np.isfinite(log_z), "AIS returned non-finite Log Z!"; print("  [Smoke Test] Log Z is finite.")
            log_likelihood = calculate_rbm_log_likelihood(rbm, val_loader, log_z)
            if is_smoke_test: assert np.isfinite(log_likelihood), "Log Likelihood is non-finite!"; print("  [Smoke Test] Log Likelihood is finite.")
            print(f"  Validation Log-Likelihood: {log_likelihood:.4f}")
            if log_likelihood > best_log_likelihood:
                best_log_likelihood = log_likelihood; epochs_without_improvement = 0; best_state_dict = copy.deepcopy(rbm.state_dict()); print(f"  New best model found!")
            else:
                epochs_without_improvement += 1; print(f"  No improvement for {epochs_without_improvement} epoch(s).")
            if epochs_without_improvement >= patience: print(f"  Stopping early. Best Log-Likelihood: {best_log_likelihood:.4f}"); break
        
        if is_smoke_test: assert best_state_dict is not None, "No best model was saved!"
        rbm.load_state_dict(best_state_dict); trained_rbm_layers.append(rbm)
        with torch.no_grad():
            propagated_data = []; prop_loader = torch.utils.data.DataLoader(current_full_train_dataset, batch_size=512)
            for batch in prop_loader: v = (batch[0] if isinstance(batch, list) else batch).to(DEVICE); h = torch.sigmoid(torch.matmul(v, rbm.W) + rbm.h_bias); propagated_data.append(h.cpu())
            current_full_train_dataset = torch.utils.data.TensorDataset(torch.cat(propagated_data, dim=0)); current_input_dim = h_size
    
    print("\nDBN pre-training complete.")
    
    # --- 3. Supervised Fine-tuning ---
    print("\n" + "="*50); print("--- Starting Supervised Fine-tuning ---"); print("="*50)
    target_ingredient_idx = np.where(ingredients == 'salt')[0][0]
    finetune_train_dataset = FineTuningRecipeDataset(recipes[train_indices_full], N_INGREDIENTS, target_ingredient_idx)
    finetune_test_dataset = FineTuningRecipeDataset(recipes[test_indices], N_INGREDIENTS, target_ingredient_idx)
    train_loader = torch.utils.data.DataLoader(finetune_train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(finetune_test_dataset, batch_size=batch_size)
    classifier = nn.Sequential()
    for i, rbm_layer in enumerate(trained_rbm_layers):
        linear_layer = nn.Linear(rbm_layer.W.shape[0], rbm_layer.W.shape[1]); linear_layer.weight.data = rbm_layer.W.data.T; linear_layer.bias.data = rbm_layer.h_bias.data
        classifier.add_module(f"layer_{i}", linear_layer); classifier.add_module(f"activation_{i}", nn.Sigmoid())
    classifier.add_module("output_layer", nn.Linear(dbn_hidden_sizes[-1], 1)); classifier.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(); optimizer = optim.Adam(classifier.parameters(), lr=0.001)
    epochs_finetune = 1 if is_smoke_test else 5
    for epoch in range(epochs_finetune):
        classifier.train()
        for data, target in tqdm(train_loader, desc=f"Fine-tune Epoch {epoch+1}/{epochs_finetune}"):
            optimizer.zero_grad(); output = classifier(data.to(DEVICE)); loss = criterion(output, target.to(DEVICE).unsqueeze(1)); loss.backward(); optimizer.step()
    print("\n--- Final Evaluation of Fine-tuned Model ---")
    classifier.eval(); correct, total = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            output = classifier(data.to(DEVICE)); predicted = (torch.sigmoid(output) > 0.5).float()
            total += target.size(0); correct += (predicted == target.to(DEVICE).unsqueeze(1)).sum().item()
    print(f"Final Test Accuracy on 'salt' prediction: {100 * correct / total:.2f}%")
    torch.save(classifier.state_dict(), 'dbn_finetuned_classifier.pth')
    print("\n✅ Final fine-tuned model saved to dbn_finetuned_classifier.pth")
    return classifier

if __name__ == '__main__':
    if SMOKE_TEST:
        print("#"*70); print("# RUNNING IN SMOKE TEST MODE - Using a small subset of data and fewer epochs."); print("#"*70)
        trained_model = main(is_smoke_test=True)
        print("\n✅ Smoke test completed successfully.")
    else:
        trained_model = main(is_smoke_test=False)
    print("\nVariable 'trained_model' now holds the trained classifier.")
