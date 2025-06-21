import os
import argparse
import random

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy import stats as scistats
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
import torchvision
import tqdm

from cosmo_compression.data import data as data
from cosmo_compression.downstream import param_est_model as pe
from cosmo_compression.model import represent

class CNNProjector(nn.Module):
    def __init__(self, backbone='resnet18', output_dim=128):
        super(CNNProjector, self).__init__()

        # Load a pretrained CNN backbone from torchvision
        if backbone == 'resnet18':
            self.backbone = torchvision.models.resnet18(pretrained=False)
            self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            num_feats = self.backbone.fc.out_features
            print(f"Loaded ResNet18 with {num_feats} features")
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Projection to 128D
        self.project_128 = nn.Linear(num_feats, output_dim)
        self.dropout = nn.Dropout(p=0.3)
        # Final projection to 2D
        self.project_2d = nn.Linear(output_dim, 2)

    def forward(self, x):
        x = self.backbone(x)              # Output shape: (B, num_feats)
        x = self.dropout(F.relu(self.project_128(x)))
        x = self.project_2d(x)            # Output shape: (B, 2)
        return x

def pct_error_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter estimation from latent representations"
    )
    parser.add_argument(
        '--model-checkpoint', '-m',
        type=str, required=False, default=None,
        help='Path to the pretrained representer checkpoint (.ckpt)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str, required=True,
        help='Directory to save results and trained models'
    )
    parser.add_argument(
        '--fig-dir', '-f',
        type=str,
        default=None,
        help='Directory to save evaluation figures (default: <output-dir>/figures)'
    )
    parser.add_argument(
        '--wdm', action='store_true',
        help='Use WDM dataset instead of CDM'
    )
    parser.add_argument(
        '--no-latent', action='store_true',
        help='Disable latent encoding, use direct image features'
    )
    parser.add_argument(
        '-c', '--channel', type=int, default=None,
        help='Channel to use for latent encoding'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    wdm = args.wdm
    use_latent = not args.no_latent

    # Prepare directories
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = args.fig_dir or os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Performing parameter estimation with WDM={wdm}, Latent={use_latent}")

    # Data loading
    param_list = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2']
    if wdm:
        param_list = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'Wdm']

    train_data = data.CAMELS(
        idx_list=range(14000),
        parameters=param_list,
        suite="IllustrisTNG" if not wdm else "IllustrisTNG",
        dataset="WDM" if wdm else "LH",
        map_type="Mcdm"
    )
    val_data = data.CAMELS(
        idx_list=range(14000, 15000),
        parameters=param_list,
        suite="IllustrisTNG" if not wdm else "IllustrisTNG",
        dataset="WDM" if wdm else "LH",
        map_type="Mcdm"
    )

    # 1) Load representer (once)
    fm = None
    if args.model_checkpoint is not None:
        fm = represent.Represent.load_from_checkpoint(args.model_checkpoint)
        fm.encoder = fm.encoder.to(device)
        for p in fm.encoder.parameters():
            p.requires_grad = False
        fm.eval()
    else:
        ...

    # 2) Helper to encode dataset (same for training and validation)
    def encode_dataset(dataset):
        loader = DataLoader(dataset, batch_size=126, shuffle=False, num_workers=5, pin_memory=True)
        features, labels = [], []
        with torch.no_grad():
            for imgs, cosmo in tqdm.tqdm(loader):
                imgs = imgs.to(device)
                if use_latent:
                    latent = fm.encoder(imgs)
                    if args.channel is not None:
                        tmp = torch.zeros_like(latent)
                        tmp[:, 6:] = latent[:, 6:]
                        latent = tmp
                    feats = fm.decoder.velocity_model.pool(latent).squeeze()
                else:
                    feats = imgs.detach()
                features.append(feats.cpu())
                labels.append(cosmo[:, :2])  # only Omega_m and sigma_8
        return torch.cat(features), torch.cat(labels)

    # 3) Encode both splits once
    x_train, y_train = encode_dataset(train_data)
    x_val, y_val = encode_dataset(val_data)

    # 4) Compute per‐parameter mean/std on the training labels (once)
    # label_mean = y_train.mean(dim=0, keepdim=True)
    # label_std = y_train.std(dim=0, keepdim=True)
    # label_std[label_std == 0] = 1.0  # avoid division by zero
    label_mean = torch.tensor([0.0, 0.0])
    label_std = torch.tensor([1.0, 1.0])

    # 5) Normalize labels (once)
    y_train_n = (y_train - label_mean) / label_std
    y_val_n = (y_val - label_mean) / label_std

    # Wrap into dataloaders (shuffle only during training)
    train_ds = TensorDataset(x_train, y_train_n)
    val_ds = TensorDataset(x_val, y_val_n)

    # 6) OPTUNA hyperparameter search (once)
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
        wd = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
        hidden = trial.suggest_int('hidden', 1, 5)
        
        if use_latent:
            hidden_dim = trial.suggest_int('hidden_dim', 1, 2048)
        else:
            hidden_dim = None
        
        # Build model according to “use_latent”
        if use_latent:
            model = pe.ParamEstVec(
                hidden_dim=hidden_dim,
                num_hiddens=hidden,
                in_dim=x_train.shape[1],
                output_size=2
            ).to(device)
        else:
            model = CNNProjector().to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()

        # Fixed seed for Optuna objective
        torch.manual_seed(0)
        np.random.seed(0)

        train_loader_local = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=8, pin_memory=True)
        val_loader_local = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=8, pin_memory=True)

        for _ in tqdm.tqdm(range(10)):
            model.train()
            for xb, yb in train_loader_local:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()

            model.eval()
            val_losses = []
            with torch.no_grad():
                for xb, yb in val_loader_local:
                    xb, yb = xb.to(device), yb.to(device)
                    out = model(xb)
                    val_losses.append(loss_fn(out, yb).item())
            vl = np.mean(val_losses)

        return vl

    # print("Starting hyperparameter search (Optuna)...")
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best = study.best_params
    print("Best hyperparameters found:", best)

    # 7) Train a single model with the best hyperparameters
    def init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    # Instantiate the model once
    if use_latent:
        model = pe.ParamEstVec(
            hidden_dim=best['hidden_dim'],
            num_hiddens=best['hidden'],
            in_dim=x_train.shape[1],
            output_size=2
        ).to(device)
        model.apply(init_weights)
    else:
        model = CNNProjector().to(device)
        
    lr         = 2e-4
    eta_min    = lr/100
    epochs     = 200
    patience   = 20

    optimizer = torch.optim.AdamW(model.parameters(), lr=best['lr'], weight_decay=best['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=epochs, 
                                                       eta_min=eta_min, verbose=True)
    best_val_loss = float('inf')
    best_model_path = os.path.join(args.output_dir, 'best_model_single.pt')

    train_loader = DataLoader(train_ds, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=50, shuffle=False)

    epochs_since_improvement = 0
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = nn.MSELoss()(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = model(xb)
                val_losses.append(nn.MSELoss()(out, yb).item())
        val_loss = np.mean(val_losses)

        # Save best checkpoint
        if val_loss < best_val_loss:
            epochs_since_improvement = 0
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1:3d} | Train Loss {running_loss/len(train_loader):.4f} | Val Loss {val_loss:.4f}")

        scheduler.step()

    # 8) Load best model and evaluate on validation set
    model.load_state_dict(torch.load(best_model_path))
    model.eval()

    true_n_list, pred_n_list = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            pred_n_list.append(out)
            true_n_list.append(yb.numpy())

    true_n = np.vstack(true_n_list)  # shape (n_val, 2)
    pred_n = np.vstack(pred_n_list)  # shape (n_val, 2)

    # De‐normalize
    true = true_n * label_std.numpy() + label_mean.numpy()
    pred = pred_n * label_std.numpy() + label_mean.numpy()

    # Compute percent‐relative‐error per sample and parameter
    rel_err = np.abs((true - pred) / true)
    rel_err = np.where(true != 0, rel_err, np.nan)  # avoid division by zero

    # Compute mean and std of relative error over samples
    mean_per_param = np.nanmean(rel_err, axis=0)   # [mean_Om, mean_sigma8]
    std_per_param = np.nanstd(rel_err, axis=0)     # [std_Om, std_sigma8]
    overall_mean = np.nanmean(rel_err)             # single scalar
    overall_std = np.nanstd(rel_err)               # single scalar

    # Compute quartiles (25th, 50th, 75th) for each parameter and overall
    q25_per_param = np.nanpercentile(rel_err, 25, axis=0)
    q50_per_param = np.nanpercentile(rel_err, 50, axis=0)
    q75_per_param = np.nanpercentile(rel_err, 75, axis=0)

    q25_overall = np.nanpercentile(rel_err, 25)
    q50_overall = np.nanpercentile(rel_err, 50)
    q75_overall = np.nanpercentile(rel_err, 75)

    # Save summary (mean, std, and quartiles)
    summary_txt = os.path.join(fig_dir, 'parameter_estimation_summary.txt')
    with open(summary_txt, 'w') as f:
        f.write("=== Summary (Single Model) ===\n")
        for i, name in enumerate(['Omega_m', 'sigma_8']):
            f.write(f"{name}: mean rel-error = {mean_per_param[i]:.4f}, "
                    f"std = {std_per_param[i]:.4f}, "
                    f"25th={q25_per_param[i]:.4f}, "
                    f"median={q50_per_param[i]:.4f}, "
                    f"75th={q75_per_param[i]:.4f}\n")
        f.write(f"\nOverall percent‐relative‐error:\n")
        f.write(f"  mean = {overall_mean:.4f}, std = {overall_std:.4f}\n")
        f.write(f"  25th = {q25_overall:.4f}, median = {q50_overall:.4f}, 75th = {q75_overall:.4f}\n")
    print(f"\nSaved summary (mean, std, quartiles) at {summary_txt}")

    # Print to console
    print("\n=== Mean ± Std Dev over validation samples ===")
    for i, name in enumerate(['Omega_m', 'sigma_8']):
        print(f"{name}: mean = {mean_per_param[i]:.4f}, std = {std_per_param[i]:.4f}, "
              f"25th={q25_per_param[i]:.4f}, median={q50_per_param[i]:.4f}, 75th={q75_per_param[i]:.4f}")
    print(f"Overall percent‐relative‐error: mean = {overall_mean:.4f}, std = {overall_std:.4f}")
    print(f"  25th = {q25_overall:.4f}, median = {q50_overall:.4f}, 75th = {q75_overall:.4f}")

    # End of main()


if __name__ == '__main__':
    main()
