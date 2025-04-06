import os
import numpy as np
import torch
import tqdm
from cosmo_compression.data import data
from cosmo_compression.model import represent
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from scipy import stats as scistats


from cosmo_compression.downstream import param_est_model as pe
import optuna

from torchvision import transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def pct_error_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))

wdm = True
latent = False

print("Performing parameter estimation with WDM =", wdm, "and Latent =", latent)

if wdm:
    if latent:
        cdm_data = data.CAMELS(
            idx_list=range(0, 14000),
            parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1", "Wdm"],
            suite="IllustrisTNG",
            dataset="WDM" if wdm else "LH",
            map_type="Mcdm"
        )
        val_data = data.CAMELS(
            idx_list=range(14000, 15000),
            parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1", "Wdm"],
            suite="IllustrisTNG",
            dataset="WDM" if wdm else "LH",
            map_type="Mcdm"
        )
    else:
        cdm_data = np.load("../../../monolith/global_data/astro_compression/CAMELS/images/wdm_latents.npy")[:14000]
        val_data = np.load("../../../monolith/global_data/astro_compression/CAMELS/images/wdm_latents.npy")[14000:]
        full_wdm = data.CAMELS(
            idx_list=range(0, 14000),
            parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1", "Wdm"],
            suite="IllustrisTNG",
            dataset="WDM" if wdm else "LH",
            map_type="Mcdm"
        )
        full_val = data.CAMELS(
            idx_list=range(14000, 15000),
            parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1", "Wdm"],
            suite="IllustrisTNG",
            dataset="WDM" if wdm else "LH",
            map_type="Mcdm"
        )
        
else:
    cdm_data = data.CAMELS(
        idx_list=range(0, 14600),
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
        suite="IllustrisTNG",
        dataset="WDM" if wdm else "LH",
        map_type="Mcdm"
    )
    val_data = data.CAMELS(
        idx_list=range(14600, 15000),
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2'],
        suite="IllustrisTNG",
        dataset="WDM" if wdm else "LH",
        map_type="Mcdm"
    )

# fm = represent.Represent.load_from_checkpoint("reversion_2_126lat/step=step=60600-val_loss=0.232.ckpt")
# fm.encoder = fm.encoder.cuda()
# for p in fm.encoder.parameters():
#     p.requires_grad = False
# fm.eval()

# train_loader = DataLoader(
#     cdm_data,
#     batch_size=126,
#     shuffle=False,
#     num_workers=1,
#     pin_memory=True,
# )

# test_loader = DataLoader(
#     val_data,
#     batch_size=126,
#     shuffle=False,
#     num_workers=1,
#     pin_memory=True,
# )

# # Build the encoded dataset using the latent vector output from the encoder
# encoded_images = []
# with torch.no_grad():
#     for images, _ in tqdm.tqdm(train_loader):
#         images = images.cuda()
#         if latent:
#             # Assume encoder returns (spatial_latent, latent_vector)
#             _, latent_vector = fm.encoder(images)
#             images = latent_vector
#         encoded_images.append(images.cpu())

# encoded_images = torch.cat(encoded_images, dim=0)
print(cdm_data.shape, np.array(full_wdm.x).shape)
train_dataset = TensorDataset(cdm_data, torch.tensor(np.array(full_wdm.x)))


# encoded_images = []
# with torch.no_grad():
#     for images, _ in tqdm.tqdm(test_loader):
#         images = images.cuda()
#         if latent:
#             _, latent_vector = fm.encoder(images)
#             images = latent_vector
#         encoded_images.append(images.cpu())
# encoded_images = torch.cat(encoded_images, dim=0)
test_dataset = TensorDataset(val_data, torch.tensor(full_val.x))

# Create new data loaders from the latent dataset
train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

print("Loaded data")

# Set up training: use ParamEstVec for latent vector inference
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if latent:
    model = pe.ParamEstVec(hidden_dim=5, num_hiddens=5, in_dim=126, output_size=(1 if wdm else 2)).to(device)
else:
    model = pe.ParamEstimatorImg(hidden=5, dr=0.1, channels=1, output_size=(1 if wdm else 2)).to(device)

criterion = nn.MSELoss()

def objective(trial):
    # Hyperparameter search space
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    hidden_dim = trial.suggest_int('hidden_dim', 1, 2048)
    hidden = trial.suggest_int('hidden', 1, 5)

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if latent:
        model = pe.ParamEstVec(hidden_dim=hidden_dim, num_hiddens=hidden, in_dim=126, output_size=(1 if wdm else 2)).to(device)
    else:
        model = pe.ParamEstimatorImg(hidden=hidden, dr=0.1, channels=1, output_size=(1 if wdm else 2)).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    num_epochs = 100  # Reduced epochs for faster Optuna trials
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), (
                labels.to(device)[:, 0:2] if not wdm 
                else labels.to(device)[:, -1].reshape(labels.shape[0], 1)
            )
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), (
                    labels.to(device)[:, 0:2] if not wdm 
                    else labels.to(device)[:, -1].reshape(labels.shape[0], 1)
                )
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

# Uncomment below to run Optuna optimization
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)
best_params = study.best_params
print("Best hyperparameters:", best_params)

# Train final model using ParamEstVec with chosen hyperparameters
if latent:
    model = pe.ParamEstVec(hidden_dim=best_params['hidden_dim'], num_hiddens=best_params['hidden'], in_dim=126, output_size=(1 if wdm else 2)).to(device)
else:
    model = pe.ParamEstimatorImg(hidden=5, dr=0.1, channels=1, output_size=(1 if wdm else 2)).to(device)

final_optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(final_optimizer, T_max=500, eta_min=1e-7)

best_loss = float('inf')
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), (
            labels.to(device)[:, 0:2] if not wdm 
            else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1)
        )
        final_optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    for images, labels in test_loader:
        images, labels = images.to(device), (
            labels.to(device)[:, 0:2] if not wdm 
            else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1)
        )
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

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), (
            labels.to(device)[:, 0:2] if not wdm 
            else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1)
        )
        outputs = model(images)
        true_params.extend(labels.cpu().numpy())
        pred_params.extend(outputs.cpu().numpy())

true_params = np.array(true_params)
pred_params = np.array(pred_params)

plt.figure(figsize=(10, 5))
plot_labels = ['Omega_m', 'sigma_8'] if not wdm else ['WDM']

for i, param_name in enumerate(plot_labels):
    l1_loss_param = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), (
                labels.to(device)[:, 0:2] if not wdm 
                else 1 / labels.to(device)[:, -1].reshape(labels.shape[0], 1)
            )
            outputs = model(images)
            l1_loss_param += (torch.mean(torch.abs((outputs[:, i] - labels[:, i]) / labels[:, i]))).item()
        print(f"Average Relative Error for WDM {wdm} with Latent {latent}, for {param_name}: {l1_loss_param/len(test_loader)}")
    
    if wdm:
        x = true_params
        y = pred_params
    else:
        x = true_params[:, i]
        y = pred_params[:, i]
        
    print(x.shape, y.shape)
        
    plt.subplot(1, 2, i+1)
    plt.scatter(x, y, alpha=0.2)
    
    print(scistats.pearsonr(true_params[:, i], pred_params[:, i]))

    
    # Compute and plot line of best fit
    if not wdm:
        slope, intercept = np.polyfit(x, y, 1)
        print(param_name, " slope ", slope, "intercept ", intercept)
        best_fit_line = slope * x + intercept
        plt.plot(x, best_fit_line, 'b-', label=f'Fit: y={slope:.2f}x')
    else:
        slope, intercept = np.polyfit(x, y, 1)
        best_fit_line = slope * x + intercept
        plt.plot(x, best_fit_line, 'b-', label=f'Fit: y={slope:.2f}x')
    
    min_val, max_val = x.min(), x.max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name}")
    plt.title(f"True vs Predicted {param_name}")
    plt.legend()

plt.tight_layout()
plt.savefig(f"cosmo_compression/results/param_est_wdm_{wdm}_latent_{latent}.png")
