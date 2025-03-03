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

from torchvision import transforms as T

os.environ["CUDA_VISIBLE_DEVICES"] = "4"

cdm_data = data.CAMELS(idx_list=range(0, 14000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","A_AGN2",], suite="IllustrisTNG", dataset="LH", map_type="Mcdm")
val_data = data.CAMELS(idx_list=range(14000, 15000), parameters=['Omega_m', 'sigma_8', "A_SN1", "A_SN2", "A_AGN1","A_AGN2",], suite="IllustrisTNG", dataset="LH", map_type="Mcdm")

fm = represent.Represent.load_from_checkpoint("64_hier/step=step=60600-val_loss=0.268.ckpt")
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
        # x0 = torch.randn_like(images).cuda()
        images = fm.encoder(images)
        # encoded_image = fm.decoder.predict(x0.cuda(), h=encoded_image, n_sampling_steps=n_sampling_steps)
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
        x0 = torch.randn_like(images).cuda()
        images = fm.encoder(images)
        # encoded_image = fm.decoder.predict(x0.cuda(), h=encoded_image, n_sampling_steps=n_sampling_steps)
        encoded_images.append(images.cpu())

# Concatenate the encoded images
encoded_images = torch.cat(encoded_images, dim=0)
test_dataset = TensorDataset(encoded_images, torch.tensor(val_data.x))


# Create data loaders for the new dataset
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

print("Loaded data")
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
        self.C23 = nn.Conv2d(8*hidden, 8*hidden, kernel_size=2, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B21 = nn.BatchNorm2d(8*hidden)
        self.B22 = nn.BatchNorm2d(8*hidden)
        self.B23 = nn.BatchNorm2d(8*hidden)
        
        # input: 8*hiddenx32x32 ----------> output: 16*hiddenx16x16
        self.C31 = nn.Conv2d(8*hidden,  16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C32 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C33 = nn.Conv2d(16*hidden, 16*hidden, kernel_size=2, stride=1, padding=0,
                            padding_mode='circular', bias=True)
        self.B31 = nn.BatchNorm2d(16*hidden)
        self.B32 = nn.BatchNorm2d(16*hidden)
        self.B33 = nn.BatchNorm2d(16*hidden)
        
        # input: 16*hiddenx16x16 ----------> output: 32*hiddenx8x8
        self.C41 = nn.Conv2d(16*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C42 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=3, stride=1, padding=1,
                            padding_mode='circular', bias=True)
        self.C43 = nn.Conv2d(32*hidden, 32*hidden, kernel_size=2, stride=1, padding=0,
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

        self.FC1  = nn.Linear(128*hidden, 64*hidden)  
        self.FC2  = nn.Linear(64*hidden,  2)  

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

        x = self.LeakyReLU(self.C01(image))
        x = self.LeakyReLU(self.B02(self.C02(x)))
        x = self.LeakyReLU(self.B03(self.C03(x)))

        x = self.LeakyReLU(self.B11(self.C11(x)))
        x = self.LeakyReLU(self.B12(self.C12(x)))
        x = self.LeakyReLU(self.B13(self.C13(x)))

        x = self.LeakyReLU(self.B21(self.C21(x)))
        x = self.LeakyReLU(self.B22(self.C22(x)))
        x = self.LeakyReLU(self.B23(self.C23(x)))

        x = self.LeakyReLU(self.B31(self.C31(x)))
        x = self.LeakyReLU(self.B32(self.C32(x)))
        x = self.LeakyReLU(self.B33(self.C33(x)))

        x = self.LeakyReLU(self.B41(self.C41(x)))
        x = self.LeakyReLU(self.B42(self.C42(x)))
        x = self.LeakyReLU(self.B43(self.C43(x)))

        x = self.LeakyReLU(self.B51(self.C51(x)))
        x = self.LeakyReLU(self.B52(self.C52(x)))
        x = self.LeakyReLU(self.B53(self.C53(x)))

        x = self.LeakyReLU(self.B61(self.C61(x)))

        x = x.view(image.shape[0],-1)
        x = self.dropout(x)
        x = self.dropout(self.LeakyReLU(self.FC1(x)))
        x = self.FC2(x)

        return x

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model_o3_err(hidden=5, dr = 0.1, channels=64).to(device)
criterion = nn.MSELoss()

def objective(trial):
    # Define hyperparameters to optimize
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-3)
    weight_decay = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    
    # Initialize model, loss, and optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_o3_err(hidden=5, dr = 0.1, channels=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # Training loop (reduced epochs for faster optimization)
    num_epochs = 10  # Set lower for Optuna trials
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)[:, 0:2]
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)[:, 0:2]
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        val_loss /= len(test_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    return best_val_loss

# # Run Optuna optimization
# study = optuna.create_study(direction='minimize')
# study.optimize(objective, n_trials=20)

# # Get best hyperparameters
# best_params = study.best_params
# print("Best hyperparameters:", best_params)

# Train final model with best hyperparameters
model = model_o3_err(hidden=5, dr = 0.1, channels=1).to(device)
final_optimizer = optim.Adam(model.parameters(), lr=0.0005522306480291232, weight_decay=1.0062357803767319e-05)
scheduler = optim.lr_scheduler.StepLR(final_optimizer, step_size=20, gamma=0.5)

best_loss = float('inf')
# Training Loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)[:, 0:2]
        final_optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        final_optimizer.step()
        running_loss += loss.item()
    
    model.eval()
    val_loss = 0.0
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)[:, 0:2]
        outputs = model(images)
        loss = criterion(outputs, labels)
        val_loss += loss.item()
        
    if val_loss < best_loss:
        best_loss = val_loss
        torch.save(model.state_dict(), 'best_model_params.pt')
    
    val_loss /= len(test_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
    scheduler.step()
model.load_state_dict(torch.load('best_model_params.pt'))
# Evaluate model and plot results
model.eval()
true_params = []
pred_params = []

l1_loss = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)[:, 0:2]
        outputs = model(images)
        l1_loss += (torch.mean(torch.abs(outputs - labels) / labels)).item()
        true_params.extend(labels.cpu().numpy())
        pred_params.extend(outputs.cpu().numpy())
true_params = np.array(true_params)
pred_params = np.array(pred_params)

print(f"Average Relative Accuracy: {l1_loss/len(test_loader)}")
plt.figure()
plt.hist(true_params[:, 0]-pred_params[:, 0], bins=20, alpha=0.5, label='Omega_m Error')
plt.savefig("cosmo_compression/results/omega_m_error.png")

plt.figure(figsize=(10, 5))

for i, param_name in enumerate(['Omega_m', 'sigma_8']):
    plt.subplot(1, 2, i+1)
    plt.scatter(true_params[:, i], pred_params[:, i], alpha=0.1)
    
    # Compute and plot line of best fit
    slope, intercept = np.polyfit(true_params[:, i], pred_params[:, i], 1)
    best_fit_line = slope * true_params[:, i] + intercept
    plt.plot(true_params[:, i], best_fit_line, 'b-', label=f'Fit: y={slope:.2f}x+{intercept:.2f}')
    print(f"Slope for {param_name}: {slope:.2f}")
    
    # Plot y=x reference line
    min_val, max_val = true_params[:, i].min(), true_params[:, i].max()
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='y=x')
    
    # Set equal axis limits
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    
    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name}")
    plt.title(f"True vs Predicted {param_name}")
    plt.legend()

plt.tight_layout()
plt.savefig("cosmo_compression/results/param_est.png")
