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
save_dir = "cosmo_compression/data"
os.makedirs(save_dir, exist_ok=True)
cdm_latents_path = os.path.join(save_dir, "cdm_latents.npy")
cdm_params_path = os.path.join(save_dir, "cdm_params.npy")
wdm_latents_path = os.path.join(save_dir, "wdm_latents.npy")
wdm_params_path = os.path.join(save_dir, "wdm_params.npy")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Load model
fm = represent.Represent.load_from_checkpoint("dropout_128/step=step=21300-val_loss=0.250.ckpt")
fm.encoder = fm.encoder.cuda()
for p in fm.encoder.parameters():
    p.requires_grad = False
fm.eval()

# Set sampling parameters
n_sampling_steps = 30

# Function to compute latents
def compute_latents(dataset, is_cdm=True):
    latents, params = [], []
    for i, (data, cosmo) in tqdm.tqdm(enumerate(dataset)):
        with torch.no_grad():
            data = torch.tensor(data).unsqueeze(0).cuda()
            latent = fm.encoder(data)
            latents.append(latent.cpu().numpy())
            params.append(np.append(cosmo, 0.0 if is_cdm else 1.0))
    return np.array(latents), np.array(params)

cdm_data = data.CAMELS(idx_list=range(10000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","A_AGN2"], suite="IllustrisTNG", dataset="LH", map_type="Mcdm")
wdm_data = data.CAMELS(idx_list=range(10000), parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_AGN2', 'WDM'], suite="IllustrisTNG", dataset="WDM", map_type="Mcdm")

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

X = np.concatenate((cdm_latents, wdm_latents), axis=0)
y = np.concatenate((cdm_labels, wdm_labels), axis=0)

# Convert to tensors
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # Add channel dim
y_tensor = torch.tensor(y, dtype=torch.long)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = int(0.1 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# # Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ad.AnomalyDetectorImg(hidden=4, dr = 0.1, channels=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005522306480291232, weight_decay=1.0062357803767319e-05)

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    hidden = trial.suggest_int('hidden', 1, 10)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ad.AnomalyDetectorImg(hidden=hidden, dr = 0.1, channels=1).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop (reduced epochs for faster optimization)
    num_epochs = 50  # Set lower for Optuna trials
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device).squeeze(1), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device).squeeze(1), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs.squeeze(), labels.float())
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

# Run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=20)

# Get best hyperparameters
best_params = study.best_params
print("Best hyperparameters:", best_params)
# Training loop
num_epochs = 30
best_val_loss = float('inf')
best_model_state = None

model = ad.AnomalyDetectorImg(hidden=best_params['hidden'], dr = 0.1, channels=1).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).squeeze(1), labels.to(device)
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
            inputs, labels = inputs.to(device).squeeze(1), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels.float())
            val_loss += loss.item()
    
    val_loss /= len(val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_state = model.state_dict()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}")



# Evaluate model on test set
model.load_state_dict(best_model_state)
model.eval()
correct, total = 0, 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device).squeeze(1), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).long()
        total += labels[labels == 1].size(0)
        correct += (predicted[labels == 1].squeeze() == labels[labels == 1]).nonzero().size(0)
print(f"Test Accuracy of Anomaly Detection on Latent: {100 * correct / total:.2f}%")