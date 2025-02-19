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

# Define file paths
save_dir = "cosmo_compression/data"
os.makedirs(save_dir, exist_ok=True)
cdm_latents_path = os.path.join(save_dir, "cdm_latents.npy")
cdm_params_path = os.path.join(save_dir, "cdm_params.npy")
wdm_latents_path = os.path.join(save_dir, "wdm_latents.npy")
wdm_params_path = os.path.join(save_dir, "wdm_params.npy")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


# Load model
# fm = represent.Represent.load_from_checkpoint("basic_latent_64/step=step=8000-val_loss=0.305.ckpt").cuda()
# fm.eval()

# Set sampling parameters
# n_sampling_steps = 50

# Function to compute latents
def compute_latents(dataset, is_cdm=True):
    latents, params = [], []
    for i, (data, cosmo) in tqdm.tqdm(enumerate(dataset)):
        with torch.no_grad():
            data = torch.tensor(data).unsqueeze(0).cuda()
            # latent = fm.encoder(data)
            latents.append(data.cpu().numpy())
            params.append(np.append(cosmo, 0.0 if is_cdm else 1.0))
    return np.array(latents), np.array(params)

# Load or compute CDM latents
if os.path.exists(cdm_latents_path) and os.path.exists(cdm_params_path):
    cdm_latents = np.load(cdm_latents_path)
    cdm_params = np.load(cdm_params_path)
    print("Loaded CDM latents from cache.")
else:
    print("Computing CDM latents")
    cdm_data = data.CAMELS(idx_list=range(10000), parameters=['Omega_m', 'sigma_8'], suite="Astrid", dataset="LH", map_type="Mcdm")
    cdm_latents, cdm_params = compute_latents(cdm_data)
    np.save(cdm_latents_path, cdm_latents)
    np.save(cdm_params_path, cdm_params)
    print("Saved CDM latents.")

# Load or compute WDM latents
if os.path.exists(wdm_latents_path) and os.path.exists(wdm_params_path):
    wdm_latents = np.load(wdm_latents_path)
    wdm_params = np.load(wdm_params_path)
    print("Loaded WDM latents from cache.")
else:
    print("Computing WDM latents")
    wdm_data = data.CAMELS(idx_list=range(10000), parameters=['Omega_m', 'sigma_8', 'SN1', 'AGN1', 'foo', 'WDM'], suite="IllustrisTNG", dataset="WDM", map_type="Mcdm")
    wdm_latents, wdm_params = compute_latents(wdm_data, is_cdm=False)
    np.save(wdm_latents_path, wdm_latents)
    np.save(wdm_params_path, wdm_params)
    print("Saved WDM latents.")

# Define the CNN classifier
class CNNClassifier(nn.Module):
    def __init__(self):
        super(CNNClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 256, kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        # self.conv3 = nn.Conv2d(64, 256, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Sequential(
            nn.Linear(256, 2304),
            nn.ReLU(),
            nn.Linear(2304, 2304),
            nn.ReLU(),
            nn.Linear(2304, 2304),
            nn.ReLU(),
            nn.Linear(2304, 1)
        )
    
    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        return x

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

# Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CNNClassifier().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-6)

# Training loop
num_epochs = 100
best_val_loss = float('inf')
best_model_state = None

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).squeeze(1).squeeze(1), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1).float())
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device).squeeze(1).squeeze(1), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels.unsqueeze(1).float())
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
        inputs, labels = inputs.to(device).squeeze(1).squeeze(1), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).long()
        total += labels.size(0)
        correct += (predicted.squeeze() == labels).nonzero().size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")

# Load best model
print("Loaded best model based on validation loss.")


class ParameterEstimator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ParameterEstimator, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    
    def forward(self, x):
        x = self.pool(x)
        x = x.view(x.shape[0], -1)  # Flatten
        x = self.fc(x)
        return x

# Prepare dataset for parameter estimation
X_param = cdm_latents
y_param = cdm_params[:, 0:2]

# Convert to tensors
X_param_tensor = torch.tensor(X_param, dtype=torch.float32).unsqueeze(1)  # Add channel dim
y_param_tensor = torch.tensor(y_param, dtype=torch.float32)

param_dataset = TensorDataset(X_param_tensor, y_param_tensor)
param_train_size = int(0.8 * len(param_dataset))
param_val_size = int(0.1 * len(param_dataset))
param_test_size = len(param_dataset) - param_train_size - param_val_size
param_train_dataset, param_val_dataset, param_test_dataset = random_split(param_dataset, [param_train_size, param_val_size, param_test_size])

param_train_loader = DataLoader(param_train_dataset, batch_size=256, shuffle=True)
param_val_loader = DataLoader(param_val_dataset, batch_size=256, shuffle=False)
param_test_loader = DataLoader(param_test_dataset, batch_size=256, shuffle=False)

# Initialize parameter estimation model, loss, and optimizer
param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
param_model = ParameterEstimator(input_dim=64, output_dim=y_param.shape[1]).to(param_device)
param_criterion = nn.MSELoss()
param_optimizer = optim.Adam(param_model.parameters(), lr=1e-5, weight_decay=1e-6)

# Training loop
num_epochs = 100
best_val_loss = float('inf')
best_param_model_state = None

for epoch in range(num_epochs):
    param_model.train()
    running_loss = 0.0
    for inputs, labels in param_train_loader:
        inputs, labels = inputs.to(param_device), labels.to(param_device)
        param_optimizer.zero_grad()
        outputs = param_model(inputs.squeeze())
        loss = param_criterion(outputs, labels)
        loss.backward()
        param_optimizer.step()
        running_loss += loss.item()
    
    param_model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in param_val_loader:
            inputs, labels = inputs.to(param_device), labels.to(param_device)
            outputs = param_model(inputs.squeeze())
            loss = param_criterion(outputs, labels)
            val_loss += loss.item()
    
    val_loss /= len(param_val_loader)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_param_model_state = param_model.state_dict()
    
    print(f"Epoch {epoch+1}, Loss: {running_loss / len(param_train_loader)}, Val Loss: {val_loss}")
    
import matplotlib.pyplot as plt

# Get the predicted and true parameter values
param_model.eval()
predicted_params = []
true_params = []
with torch.no_grad():
    for inputs, labels in param_test_loader:
        inputs, labels = inputs.to(param_device), labels.to(param_device)
        outputs = param_model(inputs)
        predicted_params.extend(outputs.cpu().numpy())
        true_params.extend(labels.cpu().numpy())

# Create a figure with multiple subplots
fig, axs = plt.subplots(nrows=len(true_params[0]), figsize=(8, 6*len(true_params[0])))

# Plot the predicted vs. true parameter value for each parameter
for i in range(len(true_params[0])):
    axs[i].scatter(true_params, predicted_params, label=f'Parameter {i+1}')
    axs[i].set_xlabel('True parameter value')
    axs[i].set_ylabel('Predicted parameter value')
    axs[i].legend()

# Layout so plots do not overlap
fig.tight_layout()

# Save the figure
plt.savefig('cosmo_compression/results/predicted_vs_true_params.png')