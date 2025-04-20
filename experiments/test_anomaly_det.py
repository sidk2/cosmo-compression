import os
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt
import optuna

from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.downstream import anomaly_det_model as ad

torch.manual_seed(90)

# Define file paths
save_dir = "../../../monolith/global_data/astro_compression/CAMELS/images"
os.makedirs(save_dir, exist_ok=True)
cdm_latents_path = os.path.join(save_dir, "cdm_latents.npy")
cdm_params_path = os.path.join(save_dir, "cdm_params.npy")
wdm_latents_path = os.path.join(save_dir, "wdm_latents.npy")
wdm_params_path = os.path.join(save_dir, "wdm_params.npy")

use_latents = False

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Load model
fm = represent.Represent.load_from_checkpoint("masked_flow_matching/step=step=25100-val_loss=0.263.ckpt")
fm.encoder = fm.encoder.cuda()
for p in fm.encoder.parameters():
    p.requires_grad = False
fm.eval()

# Set sampling parameters
n_sampling_steps = 30

# Function to compute latents
def compute_latents(dataset, is_cdm=True):
    diff, params = [], []
    for i, (data, cosmo) in tqdm.tqdm(enumerate(dataset)):
        with torch.no_grad():
            data = torch.tensor(data).unsqueeze(0).cuda()
            spatial, vec = fm.encoder(data)
            latent = spatial, vec.unsqueeze(0)
            # out = fm.decoder.predict(x0=torch.randn_like(data), h=latent, n_sampling_steps=n_sampling_steps)
            params.append(cosmo)
            diff.append(latent[1].cpu().numpy())
            # diff.append((data - out).cpu().numpy()[0, 0, :, :])
            
    return np.array(diff), np.array(params)

cdm_data = data.CAMELS(idx_list=range(15000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","A_AGN2"], suite="IllustrisTNG", dataset="LH", map_type="Mcdm")
wdm_data = data.CAMELS(idx_list=range(15000), parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_AGN2', 'WDM'], suite="IllustrisTNG", dataset="WDM", map_type="Mcdm")

# Load or compute CDM latents
if os.path.exists(cdm_latents_path) and os.path.exists(cdm_params_path):
    cdm_latents = np.load(cdm_latents_path)
    cdm_params = np.load(cdm_params_path)
    # print("Loaded CDM latents from cache.")
else:
    print("Computing CDM latents")
    cdm_latents, cdm_params = compute_latents(cdm_data)
    np.save(cdm_latents_path, cdm_latents)
    np.save(cdm_params_path, cdm_params)
    print("Saved CDM latents.")

# Load or compute WDM latents
if os.path.exists(wdm_latents_path) and os.path.exists(wdm_params_path):
    wdm_latents = np.load(wdm_latents_path)
    wdm_params = np.load(wdm_params_path)
    # print("Loaded WDM latents from cache.")
else:
    print("Computing WDM latents")
    wdm_latents, wdm_params = compute_latents(wdm_data, is_cdm=False)
    np.save(wdm_latents_path, wdm_latents)
    np.save(wdm_params_path, wdm_params)
    print("Saved WDM latents.")

# Prepare dataset
cdm_labels = np.zeros(len(cdm_latents))  # CDM labeled as 0
wdm_labels = np.ones(len(wdm_latents))   # WDM labeled as 1

# Compute dataset sizes
cdm_train_size = int(0.6 * len(cdm_data))
cdm_val_size = int(0.2 * len(cdm_data))
cdm_test_size = len(cdm_data) - cdm_train_size - cdm_val_size

wdm_train_size = int(0.6 * len(wdm_data))
wdm_val_size = int(0.2 * len(wdm_data))
wdm_test_size = len(wdm_data) - wdm_train_size - wdm_val_size

# Select subsets
cdm_train = cdm_data.y[:cdm_train_size].squeeze() if use_latents else cdm_latents[:cdm_train_size]
cdm_val = cdm_data.y[cdm_train_size:cdm_train_size + cdm_val_size].squeeze() if use_latents else cdm_latents[cdm_train_size:cdm_train_size + cdm_val_size]
cdm_test = cdm_data.y[cdm_train_size + cdm_val_size:].squeeze() if use_latents else cdm_latents[cdm_train_size + cdm_val_size:]

wdm_train = wdm_data.y[:wdm_train_size].squeeze() if use_latents else wdm_latents[:wdm_train_size]
wdm_val = wdm_data.y[wdm_train_size:wdm_train_size + wdm_val_size].squeeze() if use_latents else wdm_latents[wdm_train_size:wdm_train_size + wdm_val_size]
wdm_test = wdm_data.y[wdm_train_size + wdm_val_size:].squeeze() if use_latents else wdm_latents[wdm_train_size + wdm_val_size:]

# Prepare dataset
X_train = np.concatenate((cdm_train, wdm_train), axis=0)
y_train = np.concatenate((np.zeros(len(cdm_train)), np.ones(len(wdm_train))), axis=0)

X_val = np.concatenate((cdm_val, wdm_val), axis=0)
y_val = np.concatenate((np.zeros(len(cdm_val)), np.ones(len(wdm_val))), axis=0)

X_test = np.concatenate((cdm_test, wdm_test), axis=0)
y_test = np.concatenate((np.zeros(len(cdm_test)), np.ones(len(wdm_test))), axis=0)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)

X_val_tensor = torch.tensor(X_val, dtype=torch.float32).unsqueeze(1)
y_val_tensor = torch.tensor(y_val, dtype=torch.long)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Create datasets
dataset_train = TensorDataset(X_train_tensor, y_train_tensor)
dataset_val = TensorDataset(X_val_tensor, y_val_tensor)
dataset_test = TensorDataset(X_test_tensor, y_test_tensor)

# Create data loaders
train_loader = DataLoader(dataset_train, batch_size=512, shuffle=True)
val_loader = DataLoader(dataset_val, batch_size=512, shuffle=False)
test_loader = DataLoader(dataset_test, batch_size=512, shuffle=False)

# # Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ad.ADVec(hidden_dim=200, num_hiddens=3, in_dim=126, output_size=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1.0062357803767319e-05)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-7)

# # Training loop
num_epochs = 300
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).squeeze(0), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), labels.float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).squeeze(0), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
    
    scheduler.step()
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")


# Save best model state
torch.save(best_model_state, f"cosmo_compression/downstream/ad_params_latent_{use_latents}.pt")
# Evaluate model on test set
model.load_state_dict(torch.load(f"cosmo_compression/downstream/ad_params_latent_{use_latents}.pt"))

        
# Track incorrectly classified WDM samples
incorrect_wdm_params = []

# Evaluate model on test set
model.eval()
correct = 0
with torch.no_grad():
    for i, (inputs, labels) in enumerate(test_loader):
        inputs, labels = inputs.to(device).squeeze(0), labels.to(device)
        outputs = model(inputs)
        predicted = (torch.sigmoid(outputs.squeeze()) > 0.5).long()
        correct += torch.sum(predicted == labels)

print(incorrect_wdm_params)
# Save incorrect WDM parameters
np.save("incorrect_wdm_params.npy", np.array(incorrect_wdm_params))
        
print(f"Test Accuracy of Anomaly Detection on Latent: {100 * correct / len(test_loader.dataset):.2f}%")
