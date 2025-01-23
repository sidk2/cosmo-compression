from cosmo_compression.data import data
from cosmo_compression.model import represent

import numpy as np
import torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Load data
wdm_data = data.CAMELS(
    idx_list=range(10000),
        parameters=['Omega_m', 'sigma_8', 'SN1', 'AGN1', 'foo', 'WDM'],
        suite="IllustrisTNG",
        dataset="WDM",
        map_type="Mcdm",
    )

cdm_data = data.CAMELS(
    idx_list=range(10000),
        parameters=['Omega_m', 'sigma_8'],
        suite="Astrid",
        dataset="LH",
        map_type="Mcdm",
    )

fm = represent.Represent.load_from_checkpoint("img-lat-64ch/step=step=44500-val_loss=0.282.ckpt").cuda()
fm.eval()

# Compute CDM latents
cdm_latents = []
cdm_params = []

print("Computing CDM latents")
for i, (cosmo, data) in enumerate(cdm_data):
    with torch.no_grad():
        data = torch.tensor(data)
        data = data.cuda()
        latent, img = fm.encoder(data.unsqueeze(0))
        data = data.cpu()
        latent = latent.cpu()

        cdm_latents.append(latent)
        cdm_params.append(cosmo)

cdm_latents = np.array(cdm_latents).squeeze()
cdm_params = np.array(cdm_params).squeeze()

# Compute total matter latents
print("Computing total matter latents")
wdm_latents = []
wdm_params = []

for i, (cosmo, data) in enumerate(wdm_data):
    with torch.no_grad():
        data = torch.tensor(data)
        data = data.cuda()
        latent, img = fm.encoder(data.unsqueeze(0))
        data = data.cpu()
        latent = latent.cpu()

        wdm_latents.append(latent)
        wdm_params.append(cosmo)

wdm_latents = np.array(wdm_latents).squeeze()
wdm_params = np.array(wdm_params).squeeze()


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch.nn as nn
import torch.optim as optim

# Combine latents and labels
latents = np.concatenate([cdm_latents, wdm_latents], axis=0)
labels = np.concatenate([np.zeros(len(cdm_latents)), np.ones(len(wdm_latents))], axis=0)  # 0 for CDM, 1 for WDM

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(latents, labels, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

# Define the classifier
class LatentClassifier(nn.Module):
    def __init__(self, input_dim):
        super(LatentClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 1000),
            nn.ReLU(),
            nn.Linear(1000, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Output layer for binary classification
        )

    def forward(self, x):
        return self.fc(x)

# Initialize model, loss, and optimizer
input_dim = latents.shape[1]
model = LatentClassifier(input_dim).cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
epochs = 2000
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    outputs = model(X_train.cuda())
    loss = criterion(outputs, y_train.cuda())
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

# Evaluate the model
model.eval()
with torch.no_grad():
    y_pred_train = torch.argmax(model(X_train.cuda()), dim=1).cpu()
    y_pred_test = torch.argmax(model(X_test.cuda()), dim=1).cpu()

# Calculate accuracy
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print("Train Accuracy: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy: {:.2f}%".format(test_accuracy * 100))
print("Classification Report:\n", classification_report(y_test, y_pred_test))