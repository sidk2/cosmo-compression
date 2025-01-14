from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader,Dataset, random_split
from torch.optim.lr_scheduler import CosineAnnealingLR
from scipy import stats as scistats
# Custom Imports


# Set CUDA device
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set random seed for reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

class ParamMLP(nn.Module):
    def __init__(self, input_dim, hidden_widths, output_dim):
        super(ParamMLP, self).__init__()
        self.layers = nn.Sequential()
        prev = input_dim
        for width in hidden_widths:
            self.layers.append(nn.Linear(in_features=prev, out_features=width))
            self.layers.append(nn.ReLU())
            prev = width
        self.layers.append(nn.Linear(in_features=prev, out_features=output_dim))
    
    def forward(self, x,):
        return self.layers(x)

 
class CompressionDataset(Dataset):
    def __init__(self, idx_list: List[int]):
        super().__init__()
        self.x = torch.tensor(np.load('cosmo_compression/parameter_estimation/data/cdm_latents.npy'))[idx_list[0]: idx_list[-1]]
        self.y = torch.tensor(np.load('cosmo_compression/parameter_estimation/data/cdm_params.npy'))[idx_list[0]: idx_list[-1]]
    
    def __len__(self):
        return len(self.x)
        
    def __getitem__(self, index):
        return self.x[index], self.y[index][0:2]

if __name__ == '__main__':
    # Hyperparameters
    num_dps = 15000
    batch_size = 100
    learning_rate = 1e-6
    num_epochs = 50  # Adjust based on your training needs

    train_data = CompressionDataset(idx_list=[0, 14600])
    val_data = CompressionDataset(idx_list=[14600, 15000])

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_data,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # # Define model, optimizer, and loss function
    model = ParamMLP(input_dim=2304, hidden_widths=[1000,1000,256], output_dim=2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Adjust based on the specific task and model requirements
    sched = CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-7)

    # # Training and validation loops
    best_val_loss = float('inf')
    checkpoint_path = "cosmo_compression/parameter_estimation/data/best_model_full_img.pth"

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            # print(targets, outputs)
            
            if torch.isnan(outputs).any():
                exit(0)
            loss = criterion(outputs, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
        sched.step()

        # Average train loss over all batches
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)

                # Forward pass
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)

        # Average validation loss over all batches
        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save checkpoint if validation loss improves
        if val_loss < best_val_loss:
            print(f"Validation loss improved from {best_val_loss:.4f} to {val_loss:.4f}. Saving model...")
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

    print("Training complete. Best model saved as 'best_model.pth'")

    # --- Validation and Plotting ---

    # Load the best model
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in val_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            
            # Store predictions and targets for plotting
            all_preds.append(outputs[0].cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    # Convert lists to numpy arrays
    all_preds = np.array(all_preds)
    all_targets = np.concatenate(all_targets, axis=0)

    print(all_preds.shape, all_targets.shape)

    # Plotting scatter plots for each column with line of best fit
    params = ['$\Omega_m$', '$\sigma_8$',]
    # params = ['A_AGN1', 'A_AGN2']
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    for i in range(2):
        axes[i].scatter(all_targets[:, i], all_preds[:, i], alpha=0.5, color='skyblue', edgecolor='black')
        
        # Line of best fit
        m, b = np.polyfit(all_targets[:, i], all_preds[:, i], 1)
        print(scistats.pearsonr(all_targets[:, i], all_preds[:, i]))
        axes[i].plot(all_targets[:, i], m * all_targets[:, i] + b, color='red')
        
        # Titles and labels
        axes[i].set_title(rf'{params[i]}')
        axes[i].set_xlabel('Ground Truth')
        axes[i].set_ylabel('Prediction')

    plt.suptitle("Predictions using Latent Space of Model")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("cosmo_compression/results/scatter_est_cdm.png")
