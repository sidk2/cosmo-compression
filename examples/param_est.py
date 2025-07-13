import os
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
import tqdm

from cosmo_compression.data import data as data_module
from cosmo_compression.downstream import param_est_model as pe
from cosmo_compression.model import represent


def pct_error_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Mean percent-relative error loss."""
    return torch.mean(torch.abs((y_true - y_pred) / y_true))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parameter estimation from latent representations"
    )
    parser.add_argument(
        '-m', '--model-checkpoint',
        type=str,
        default=None,
        help='Path to pretrained representer checkpoint (.ckpt)'
    )
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='Directory to save results and trained models'
    )
    parser.add_argument(
        '-f', '--fig-dir',
        type=str,
        default=None,
        help='Directory to save evaluation figures (default: <output-dir>/figures)'
    )
    parser.add_argument(
        '--wdm',
        action='store_true',
        help='Use WDM dataset instead of CDM'
    )
    parser.add_argument(
        '--no-latent',
        action='store_true',
        help='Disable latent encoding, use direct image features'
    )
    parser.add_argument(
        '-c', '--channel',
        type=int,
        default=None,
        help='Channel to use for latent encoding'
    )
    return parser.parse_args()


def setup_directories(output_dir: str, fig_dir: str = None) -> str:
    os.makedirs(output_dir, exist_ok=True)
    figures = fig_dir or os.path.join(output_dir, 'figures')
    os.makedirs(figures, exist_ok=True)
    return figures


def load_representer(checkpoint: str, device: torch.device):
    if checkpoint:
        rep = represent.Represent.load_from_checkpoint(checkpoint)
        rep.encoder = rep.encoder.to(device)
        for param in rep.encoder.parameters():
            param.requires_grad = False
        rep.eval()
        return rep
    return None


def encode_dataset(
    dataset,
    encoder_model,
    device: torch.device,
    use_latent: bool,
    channel: int = None,
    batch_size: int = 126,
) -> tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=5,
        pin_memory=True,
    )
    features, labels = [], []

    with torch.no_grad():
        for imgs, cosmo in tqdm.tqdm(loader):
            imgs = imgs.to(device)

            if use_latent and encoder_model is not None:
                latent = encoder_model.encoder(imgs)
                if channel is not None:
                    mask = torch.zeros_like(latent)
                    mask[:, channel:] = latent[:, channel:]
                    latent = mask
                feats = encoder_model.decoder.velocity_model.pool(latent).squeeze()
            else:
                feats = imgs.detach()

            features.append(feats.cpu())
            labels.append(cosmo[:, :2])

    return torch.cat(features), torch.cat(labels)


def objective(
    trial,
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_val: torch.Tensor,
    y_val: torch.Tensor,
    device: torch.device,
    use_latent: bool,
) -> float:
    lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    wd = trial.suggest_loguniform('weight_decay', 1e-8, 1e-4)
    num_hiddens = trial.suggest_int('num_hiddens', 1, 5)
    hidden_dim = (
        trial.suggest_int('hidden_dim', 1, 2048) if use_latent else None
    )

    model = pe.ParamEstVec(
        hidden_dim=hidden_dim,
        num_hiddens=num_hiddens,
        in_dim=x_train.shape[1],
        output_size=2,
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    torch.manual_seed(0)
    np.random.seed(0)

    train_ds = TensorDataset(x_train, y_train)
    val_ds = TensorDataset(x_val, y_val)

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    for _ in range(10):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(loss_fn(model(xb), yb).item())

    return float(np.mean(val_losses))


def init_weights(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def train_and_evaluate(args: argparse.Namespace) -> None:
    # Directories and device
    fig_dir = setup_directories(args.output_dir, args.fig_dir)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Data
    params = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'Wdm'] if args.wdm else ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2']
    suite = 'IllustrisTNG'
    dataset_type = 'WDM' if args.wdm else 'LH'

    train_data = data_module.CAMELS(range(14000), params, suite, dataset_type, 'Mcdm')
    val_data = data_module.CAMELS(range(14000, 15000), params, suite, dataset_type, 'Mcdm')

    rep_model = load_representer(args.model_checkpoint, device)
    use_latent = not args.no_latent

    x_train, y_train = encode_dataset(train_data, rep_model, device, use_latent, args.channel)
    x_val, y_val = encode_dataset(val_data, rep_model, device, use_latent, args.channel)

    # Normalize labels
    label_mean = torch.zeros(2)
    label_std = torch.ones(2)
    y_train_n = (y_train - label_mean) / label_std
    y_val_n = (y_val - label_mean) / label_std

    # Hyperparameter search
    study = optuna.create_study(direction='minimize')
    study.optimize(
        lambda t: objective(t, x_train, y_train_n, x_val, y_val_n, device, use_latent),
        n_trials=30,
    )
    best_params = study.best_params
    print("Best hyperparameters:", best_params)

    # Final training
    model = pe.ParamEstVec(
        hidden_dim=best_params.get('hidden_dim'),
        num_hiddens=best_params['num_hiddens'],
        in_dim=x_train.shape[1],
        output_size=2,
    ).to(device)
    model.apply(init_weights)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=200, eta_min=best_params['lr'] / 100, verbose=True
    )

    best_loss = float('inf')
    best_path = os.path.join(args.output_dir, 'best_model.pt')

    train_ds = TensorDataset(x_train, y_train_n)
    val_ds = TensorDataset(x_val, y_val_n)
    train_loader = DataLoader(train_ds, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=50, shuffle=False)

    for epoch in range(200):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = nn.MSELoss()(model(xb), yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                val_losses.append(nn.MSELoss()(model(xb), yb).item())

        val_loss = float(np.mean(val_losses))
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), best_path)
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= 20:
                print(f"Early stopping at epoch {epoch+1}")
                break

        print(f"Epoch {epoch+1:3d} | Train {running_loss/len(train_loader):.4f} | Val {val_loss:.4f}")
        scheduler.step()

    # Evaluation
    model.load_state_dict(torch.load(best_path))
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            preds = model(xb.to(device)).cpu().numpy()
            all_pred.append(preds)
            all_true.append(yb.numpy())

    true_arr = np.vstack(all_true)
    pred_arr = np.vstack(all_pred)

    # De-normalize and compute statistics
    true_vals = true_arr * label_std.numpy() + label_mean.numpy()
    pred_vals = pred_arr * label_std.numpy() + label_mean.numpy()
    rel_err = np.abs((true_vals - pred_vals) / true_vals)
    rel_err[true_vals == 0] = np.nan

    stats = {
        'mean_per_param': np.nanmean(rel_err, axis=0),
        'std_per_param': np.nanstd(rel_err, axis=0),
        'overall_mean': np.nanmean(rel_err),
        'overall_std': np.nanstd(rel_err),
        'quartiles_per_param': np.nanpercentile(rel_err, [25, 50, 75], axis=0),
        'quartiles_overall': np.nanpercentile(rel_err, [25, 50, 75]),
    }

    # Save summary
    summary_path = os.path.join(fig_dir, 'parameter_estimation_summary.txt')
    with open(summary_path, 'w') as out:
        out.write("=== Summary ===\n")
        for i, name in enumerate(['Omega_m', 'sigma_8']):
            m, s = stats['mean_per_param'][i], stats['std_per_param'][i]
            q25, q50, q75 = stats['quartiles_per_param'][:, i]
            out.write(f"{name}: mean={m:.4f}, std={s:.4f}, 25={q25:.4f}, med={q50:.4f}, 75={q75:.4f}\n")
        out.write(f"Overall mean={stats['overall_mean']:.4f}, std={stats['overall_std']:.4f}\n")
        q25o, q50o, q75o = stats['quartiles_overall']
        out.write(f"Overall quartiles: 25={q25o:.4f}, med={q50o:.4f}, 75={q75o:.4f}\n")

    print(f"Summary saved to {summary_path}")


if __name__ == '__main__':
    args = parse_args()
    train_and_evaluate(args)
