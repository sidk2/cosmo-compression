import os
import numpy as np
import torch
import tqdm
from cosmo_compression.data import data
from cosmo_compression.model import represent
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

from cosmo_compression.downstream import param_est_model as pe
import optuna

from torchvision import transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

def pct_error_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

wdm = True
latent = True

print("Performing parameter estimation with WDM = ", wdm, " and Latent = ", latent)

cdm_data = data.CAMELS(idx_list=range(0, 14000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","Wdm",], suite="IllustrisTNG" if wdm else "IllustrisTNG", dataset="WDM" if wdm else "LH", map_type="Mcdm")
val_data = data.CAMELS(idx_list=range(14000, 15000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","Wdm",], suite="IllustrisTNG" if wdm else "IllustrisTNG", dataset="WDM" if wdm else "LH", map_type="Mcdm")

fm = represent.Represent.load_from_checkpoint("dropout_128/step=step=21300-val_loss=0.250.ckpt")
# fm = represent.Represent.load_from_checkpoint("64_hier/step=step=60600-val_loss=0.268.ckpt")
fm.encoder = fm.encoder.cuda()
for p in fm.encoder.parameters():
    p.requires_grad = False
fm.eval()

train_loader = DataLoader(
    cdm_data,
    batch_size=128,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

test_loader = DataLoader(
    val_data,
    batch_size=128,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)
encoded_images = []
n_sampling_steps = 20
with torch.no_grad():
    for images, _ in tqdm.tqdm(train_loader):
        images = images.cuda()
        if latent:
            images = fm.encoder(images)
        encoded_images.append(images.cpu())

# Concatenate the encoded images
encoded_images = torch.cat(encoded_images, dim=0)

# Create a new dataset with the encoded images
train_dataset = TensorDataset(encoded_images, torch.tensor(cdm_data.x))

encoded_images = []
n_sampling_steps = 20
with torch.no_grad():
    for images, _ in tqdm.tqdm(test_loader):
        images = images.cuda()
        if latent:
            images = fm.encoder(images)
        encoded_images.append(images.cpu())

# Concatenate the encoded images
encoded_images = torch.cat(encoded_images, dim=0)
test_dataset = TensorDataset(encoded_images, torch.tensor(val_data.x))


# Create data loaders for the new dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Loaded data")

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if latent:
    model = pe.ParamEstimatorLat(hidden=5, dr = 0.1, channels=128, output_size=(1 if wdm else 2)).to(device)
else:
    model = pe.ParamEstimatorImg(hidden=5, dr = 0.1, channels=1 if latent else 1, output_size=(1 if wdm else 2)).to(device)

criterion = nn.MSELoss()

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    hidden = trial.suggest_int('hidden', 1, 10)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if latent:
        model = pe.ParamEstimatorLat(hidden=5, dr = 0.1, channels=128, output_size=(1 if wdm else 2)).to(device)
    else:
        model = pe.ParamEstimatorImg(hidden=5, dr = 0.1, channels=1, output_size=(1 if wdm else 2)).to(device)
    criterion = pct_error_loss
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop (reduced epochs for faster optimization)
    num_epochs = 10  # Set lower for Optuna trials
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), (labels.to(device)[:, 0:2] if not wdm else labels.to(device)[:, -1].reshape(labels.shape[0], 1))
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), (labels.to(device)[:, 0:2] if not wdm else labels.to(device)[:, -1].reshape(labels.shape[0], 1))
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

# Run Optuna optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)

# # Get best hyperparameters
# best_params = study.best_params
# print("Best hyperparameters:", best_params)

# # Train final model with best hyperparameters
if latent:
    model = pe.ParamEstimatorLat(hidden=5, dr = 0.1, channels=128, output_size=(1 if wdm else 2)).to(device)
else:
    model = pe.ParamEstimatorImg(hidden=5, dr = 0.1, channels=1, output_size=(1 if wdm else 2)).to(device)

final_optimizer = optim.Adam(model.parameters(), lr=0.0009425213281205723, weight_decay=4.3833691213012614e-05)
scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=20, gamma=0.5)

best_loss = float('inf')
# Training Loop
num_epochs = 50
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), (labels.to(device)[:, 0:2] if not wdm else  1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1))
        final_optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    for images, labels in test_loader:
        images, labels = images.to(device), (labels.to(device)[:, 0:2] if not wdm else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1))
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), f'pe_params_wdm_{wdm}_latent_{latent}.pt')
    
    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    scheduler.step()


model.load_state_dict(torch.load(f'pe_params_wdm_{wdm}_latent_{latent}.pt'))

# Evaluate model and plot results
model.eval()
true_params = []
pred_params = []

l1_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), (labels.to(device)[:, 0:2] if not wdm else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1))
        outputs = model(images)
        l1_loss += (torch.mean(torch.abs(outputs - 1 / labels) / (1 / labels))).item()
        true_params.extend(labels.cpu().numpy())
        pred_params.extend(outputs.cpu().numpy())

true_params = np.array(true_params)
pred_params = np.array(pred_params)

plt.figure(figsize=(10, 5))

labels = ['Omega_m', 'sigma_8'] if not wdm else ['WDM']

for i, param_name in enumerate(labels):
    l1_loss = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), (labels.to(device)[:, 0:2] if not wdm else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1))
            outputs = model(images)
            l1_loss += (torch.mean(torch.abs(outputs[:, i] - 1 / labels[:, i]) / (1 / labels[:, i]))).item()
        print(f"Average Relative Accuracy for WDM {wdm} with Latent {latent}, for {param_name}: {l1_loss/len(test_loader)}")
    
    x = true_params[:, i] if not wdm else true_params
    y = pred_params[:, i] if not wdm else pred_params
    plt.subplot(1, 2, i+1)
    
    print(x.shape, y.shape)
    plt.scatter(x, y, alpha=0.1)
    
    # Compute and plot line of best fit
    slope, intercept = np.polyfit(x[:, 0], y[:, 0], 1)
    best_fit_line = slope * x + intercept
    plt.plot(x, best_fit_line, 'b-', label=f'Fit: y={slope:.2f}x')
    print(f"Slope for {param_name}: {slope:.2f}")
    
    # Plot y=x reference line
    min_val, max_val = x.min(), x.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    
    # Set equal axis limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name}")
    plt.title(f"True vs Predicted {param_name}")
    plt.legend()

plt.tight_layout()
plt.savefig(f"cosmo_compression/results/param_est_wdm_{wdm}_latent_{latent}.png")
