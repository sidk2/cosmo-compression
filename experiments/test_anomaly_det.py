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
import optuna

# Define file paths
save_dir = "cosmo_compression/data"
os.makedirs(save_dir, exist_ok=True)
cdm_latents_path = os.path.join(save_dir, "cdm_latents.npy")
cdm_params_path = os.path.join(save_dir, "cdm_params.npy")
wdm_latents_path = os.path.join(save_dir, "wdm_latents.npy")
wdm_params_path = os.path.join(save_dir, "wdm_params.npy")

os.environ["CUDA_VISIBLE_DEVICES"] = "5"


# Load model
fm = represent.Represent.load_from_checkpoint("64_hier/step=step=60600-val_loss=0.268.ckpt")
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

# Load or compute CDM latents
if os.path.exists(cdm_latents_path) and os.path.exists(cdm_params_path):
    cdm_latents = np.load(cdm_latents_path)
    cdm_params = np.load(cdm_params_path)
    print("Loaded CDM latents from cache.")
else:
    print("Computing CDM latents")
    cdm_data = data.CAMELS(idx_list=range(10000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","A_AGN2"], suite="IllustrisTNG", dataset="LH", map_type="Mcdm")
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
    wdm_data = data.CAMELS(idx_list=range(10000), parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_AGN2', 'WDM'], suite="IllustrisTNG", dataset="WDM", map_type="Mcdm")
    wdm_latents, wdm_params = compute_latents(wdm_data, is_cdm=False)
    np.save(wdm_latents_path, wdm_latents)
    np.save(wdm_params_path, wdm_params)
    print("Saved WDM latents.")

# Define the CNN classifier
class model_o3_err(nn.Module):
    def __init__(self, hidden, dr, channels):
        super(model_o3_err, self).__init__()
        
        # input: 1x256x256 ---------------> output: 2*hiddenx128x128
        self.C01 = nn.Conv2d(channels,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C02 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=3, stride=1, padding=1, 
                            padding_mode='circular', bias=True)
        self.C03 = nn.Conv2d(2*hidden,  2*hidden, kernel_size=2, stride=2, padding=0, 
                            padding_mode='circular', bias=True)
        self.B01 = nn.BatchNorm2d(2*hidden)
        self.B02 = nn.BatchNorm2d(2*hidden)
        self.B03 = nn.BatchNorm2d(2*hidden)
        
        # input: 2*hiddenx128x128 ----------> output: 4*hiddenx64x64
        self.C11 = nn.Conv2d(2*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C12 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C13 = nn.Conv2d(4*hidden, 4*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B11 = nn.BatchNorm2d(4*hidden)
        self.B12 = nn.BatchNorm2d(4*hidden)
        self.B13 = nn.BatchNorm2d(4*hidden)
        
        # input: 4*hiddenx64x64 --------> output: 8*hiddenx32x32
        self.C21 = nn.Conv2d(4*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C22 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=2, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.B41 = nn.BatchNorm2d(32*hidden)
        self.B42 = nn.BatchNorm2d(32*hidden)
        self.B43 = nn.BatchNorm2d(32*hidden)
        
        # input: 32*hiddenx8x8 ----------> output:64*hiddenx4x4
        self.C51 = nn.Conv2d(32*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C52 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C53 = nn.Conv2d(64*hidden, 64*hidden, kernel_size=2, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B51 = nn.BatchNorm2d(64*hidden)
        self.B52 = nn.BatchNorm2d(64*hidden)
        self.B53 = nn.BatchNorm2d(64*hidden)

        # input: 64*hiddenx4x4 ----------> output: 128*hiddenx1x1
        self.C61 = nn.Conv2d(64*hidden, 128*hidden, kernel_size=1, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B61 = nn.BatchNorm2d(128*hidden)

        self.P0  = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)

        self.FC1  = nn.Linear(16*16*64, 256*hidden)  
        self.FC2  = nn.Linear(256*hidden, 128*hidden) 
        self.FC3  = nn.Linear(128*hidden, 64*hidden)  
        self.FC4  = nn.Linear(64*hidden, 64*hidden)  
        self.FC5  = nn.Linear(64*hidden, 64*hidden)  
        self.FC6  = nn.Linear(64*hidden, 64*hidden)  
        self.FC7  = nn.Linear(64*hidden,  1)  

        self.dropout   = nn.Dropout(p=dr)
        self.ReLU      = nn.ReLU()
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.tanh      = nn.Tanh()

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)


    def forward(self, image):
        x = self.LeakyReLU(self.FC1(image.reshape(-1, 16*16*64)))
        x = self.LeakyReLU(self.FC2(x))
        x = self.LeakyReLU(self.FC3(x))
        x = self.LeakyReLU(self.FC4(x))
        x = self.LeakyReLU(self.FC5(x))
        x = self.LeakyReLU(self.FC6(x))
        x = self.FC7(x)

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

# # Initialize model, loss, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_o3_err(hidden=4, dr = 0.1, channels=64).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0005522306480291232, weight_decay=1.0062357803767319e-05)



def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    hidden = trial.suggest_int('hidden', 1, 10)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_o3_err(hidden=hidden, dr = 0.1, channels=64).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop (reduced epochs for faster optimization)
    num_epochs = 30  # Set lower for Optuna trials
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs.squeeze(), labels.float())
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
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

model = model_o3_err(hidden=best_params['hidden'], dr = 0.1, channels=64).to(device)
optimizer = optim.Adam(model.parameters(), lr=best_params['lr'], weight_decay=best_params['weight_decay'])

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device).squeeze(1).squeeze(1), labels.to(device)
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
            inputs, labels = inputs.to(device).squeeze(1).squeeze(1), labels.to(device)
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
        inputs, labels = inputs.to(device).squeeze(1).squeeze(1), labels.to(device)
        outputs = model(inputs)
        predicted = (outputs > 0.5).long()
        total += labels[labels == 1].size(0)
        correct += (predicted[labels == 1].squeeze() == labels[labels == 1]).nonzero().size(0)
print(f"Test Accuracy: {100 * correct / total:.2f}%")


# class ParameterEstimator(nn.Module):
#     def __init__(self, input_dim, output_dim):
#         super(ParameterEstimator, self).__init__()
#         self.fc = nn.Sequential(
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, 1024),
#             nn.ReLU(),
#             nn.Linear(1024, output_dim)
#         )
    
#     def forward(self, x):
#         x = x.view(x.shape[0], -1)  # Flatten
#         x = self.fc(x)
#         return x

# # Prepare dataset for parameter estimation
# X_param = cdm_latents
# y_param = cdm_params[:, 0:2]

# # Convert to tensors
# X_param_tensor = torch.tensor(X_param, dtype=torch.float32).unsqueeze(1)  # Add channel dim
# y_param_tensor = torch.tensor(y_param, dtype=torch.float32)

# param_dataset = TensorDataset(X_param_tensor, y_param_tensor)
# param_train_size = int(0.8 * len(param_dataset))
# param_val_size = int(0.1 * len(param_dataset))
# param_test_size = len(param_dataset) - param_train_size - param_val_size
# param_train_dataset, param_val_dataset, param_test_dataset = random_split(param_dataset, [param_train_size, param_val_size, param_test_size])

# param_train_loader = DataLoader(param_train_dataset, batch_size=256, shuffle=True)
# param_val_loader = DataLoader(param_val_dataset, batch_size=256, shuffle=False)
# param_test_loader = DataLoader(param_test_dataset, batch_size=256, shuffle=False)

# # Initialize parameter estimation model, loss, and optimizer
# param_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# param_model = ParameterEstimator(input_dim=4, output_dim=y_param.shape[1]).to(param_device)
# param_criterion = nn.MSELoss()
# param_optimizer = optim.Adam(param_model.parameters(), lr=1e-6, weight_decay=1e-8)

# # Training loop
# num_epochs = 100
# best_val_loss = float('inf')
# best_param_model_state = None

# for epoch in range(num_epochs):
#     param_model.train()
#     running_loss = 0.0
#     for inputs, labels in param_train_loader:
#         inputs, labels = inputs.to(param_device).squeeze(), labels.to(param_device)
#         param_optimizer.zero_grad()
#         outputs = param_model(inputs.squeeze())
#         loss = param_criterion(outputs, labels)
#         loss.backward()
#         param_optimizer.step()
#         running_loss += loss.item()
    
#     param_model.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for inputs, labels in param_val_loader:
#             inputs, labels = inputs.to(param_device).squeeze(), labels.to(param_device)
#             outputs = param_model(inputs.squeeze())
#             loss = param_criterion(outputs, labels)
#             val_loss += loss.item()
    
#     val_loss /= len(param_val_loader)
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         best_param_model_state = param_model.state_dict()
    
#     print(f"Epoch {epoch+1}, Loss: {running_loss / len(param_train_loader)}, Val Loss: {val_loss}")
    
# import matplotlib.pyplot as plt

# # Get the predicted and true parameter values
# param_model.eval()
# predicted_params = []
# true_params = []
# with torch.no_grad():
#     for inputs, labels in param_test_loader:
#         inputs, labels = inputs.to(param_device).squeeze(), labels.to(param_device)
#         outputs = param_model(inputs)
#         predicted_params.extend(outputs.cpu().numpy())
#         true_params.extend(labels.cpu().numpy())