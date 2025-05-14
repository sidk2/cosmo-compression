import os
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import matplotlib.pyplot as plt
import numpy as np
import optuna
from scipy import stats as scistats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms as T
import tqdm

from cosmo_compression.data import data
from cosmo_compression.downstream import param_est_model as pe
from cosmo_compression.model import represent

def pct_error_loss(y_pred, y_true):
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Parameter estimation from latent representations"
    )
    parser.add_argument(
        '--model-checkpoint', '-m',
        type=str, required=True,
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
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

    print(f"Performing parameter estimation with WDM={wdm}, Latent={use_latent}")

    # Data loading
    param_list = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2']
    if wdm:
        param_list = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'Wdm']

    train_idx = range(0, 14000) if wdm else range(0, 14600)
    val_idx   = range(train_idx.stop, train_idx.stop +  (1000 if wdm else 400))

    train_data = data.CAMELS(
        idx_list=train_idx,
        parameters=param_list,
        suite="IllustrisTNG",
        dataset="WDM" if wdm else "LH",
        map_type="Mcdm"
    )
    val_data = data.CAMELS(
        idx_list=val_idx,
        parameters=param_list,
        suite="IllustrisTNG",
        dataset="WDM" if wdm else "LH",
        map_type="Mcdm"
    )

    # Load representer
    fm = represent.Represent.load_from_checkpoint(args.model_checkpoint)
    fm.encoder = fm.encoder.to(device)
    for p in fm.encoder.parameters(): p.requires_grad = False
    fm.eval()

    # Helper to encode dataset
    def encode_dataset(dataset):
        loader = DataLoader(dataset, batch_size=126, shuffle=False, num_workers=1, pin_memory=True)
        features, labels = [], []
        with torch.no_grad():
            for imgs, cosmo in tqdm.tqdm(loader):
                imgs = imgs.to(device)
                if use_latent:
                    latent = fm.encoder(imgs)
                    feats = fm.decoder.velocity_model.fc(
                        fm.decoder.velocity_model.pool(latent).squeeze()
                    )
                else:
                    feats = imgs.view(imgs.size(0), -1)
                features.append(feats.cpu())
                labels.append(cosmo[:,:2])
        return torch.cat(features), torch.cat(labels)

    x_train, y_train = encode_dataset(train_data)
    x_val,   y_val   = encode_dataset(val_data)

    train_ds = TensorDataset(x_train, y_train)
    val_ds   = TensorDataset(x_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=512, shuffle=False)

    # Optuna hyperparameter search
    def objective(trial):
        lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
        wd = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
        hidden_dim = trial.suggest_int('hidden_dim', 1, 2048)
        hidden = trial.suggest_int('hidden', 1, 5)

        if use_latent:
            model = pe.ParamEstVec(
                hidden_dim=hidden_dim,
                num_hiddens=hidden,
                in_dim=x_train.shape[1],
                output_size=(1 if wdm else 2)
            ).to(device)
        else:
            model = pe.ParamEstimatorImg(
                hidden=hidden,
                dr=0.1,
                channels=1,
                output_size=(1 if wdm else 2)
            ).to(device)
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
        loss_fn = nn.MSELoss()

        for _ in range(100):
            model.train()
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)[:, : (1 if wdm else 2)]
                opt.zero_grad()
                out = model(xb)
                l = loss_fn(out, yb)
                l.backward()
                opt.step()
            model.eval()
            vl = np.mean([loss_fn(model(xb.to(device)), yb.to(device)[:, :(1 if wdm else 2)]).item()
                          for xb, yb in val_loader])
        return vl

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=30)
    best = study.best_params
    print("Best hyperparameters:", best)

    # Final training
    if use_latent:
        model = pe.ParamEstVec(
            hidden_dim=best['hidden_dim'], num_hiddens=best['hidden'],
            in_dim=x_train.shape[1], output_size=(1 if wdm else 2)
        ).to(device)
    else:
        model = pe.ParamEstimatorImg(
            hidden=best['hidden'], dr=0.1, channels=1,
            output_size=(1 if wdm else 2)
        ).to(device)
    opt = optim.Adam(model.parameters(), lr=best['lr'], weight_decay=best['weight_decay'])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=500, eta_min=1e-7)
    best_loss = float('inf')

    for epoch in range(500):
        model.train()
        run_loss = 0.0
        
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)[:, :(1 if wdm else 2)]
            opt.zero_grad()
            out = model(xb)
            l = nn.MSELoss()(out, yb)
            l.backward()
            opt.step()
            run_loss += l.item()
        
        val_loss = np.sum([nn.MSELoss()(model(xb.to(device)), yb.to(device)[:, :(1 if wdm else 2)]).item()
                           for xb, yb in val_loader])
        val_loss /= len(val_loader)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
        print(f"Epoch {epoch+1}: Train {run_loss/len(train_loader):.4f}, Val {val_loss:.4f}")
        sched.step()

    # Evaluation & plotting
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pt')))
    model.eval()
    true, pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            out = model(xb)
            true.append(yb.numpy())
            pred.append(out.cpu().numpy())
    true = np.vstack(true)
    pred = np.vstack(pred)

    plot_labels = ['WDM'] if wdm else ['Omega_m', 'sigma_8']
    fig, axes = plt.subplots(1, len(plot_labels), figsize=(10, 5))
    for i, name in enumerate(plot_labels):
        ax = axes[i] if len(plot_labels)>1 else axes
        x, y = true[:, i], pred[:, i]
        ax.scatter(x, y, alpha=0.2)
        slope, inter = np.polyfit(x, y, 1)
        ax.plot(x, slope*x + inter, '-', label=f'y={slope:.2f}x')
        ax.plot([x.min(), x.max()], [x.min(), x.max()], '--', label='y=x')
        ax.set(
            xlabel=f'True {name}', ylabel=f'Predicted {name}',
            title=f'{name} (r={scistats.pearsonr(x, y)[0]:.2f})'
        )
        ax.legend()
    plt.tight_layout()
    # Save figure to specified fig_dir
    fig_path = os.path.join(fig_dir, 'param_est_results.png')
    plt.savefig(fig_path)
    print(f"Saved evaluation figure at {fig_path}")

    # Compute and save validation percent relative error
    # true and pred are numpy arrays
    rel_err = np.abs((true - pred) / true)
    # handle zeros in true to avoid division by zero
    rel_err = np.where(true != 0, rel_err, np.nan)
    per_param = np.nanmean(rel_err, axis=0)
    overall = np.nanmean(rel_err)
    txt_path = os.path.join(fig_dir, 'val_pct_error.txt')
    with open(txt_path, 'w') as f:
        f.write("Validation Percent Relative Error\n")
        for name, err in zip(plot_labels, per_param):
            f.write(f"{name}: {err:.4f}\n")
        f.write(f"Overall: {overall:.4f}\n")
    print(f"Saved validation percent relative error at {txt_path}")

if __name__ == '__main__':
    main()
