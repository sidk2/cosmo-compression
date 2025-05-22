import os
import argparse
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

import optuna

from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.downstream import anomaly_det_model as ad

np.random.seed(42)
torch.manual_seed(42)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def parse_args():
    parser = argparse.ArgumentParser(
        description="Anomaly detection on CAMELS latents or image reconstruction errors"
    )
    parser.add_argument(
        '-c', '--checkpoint', type=str, required=True,
        help='Path to pretrained representer checkpoint (.ckpt)'
    )
    parser.add_argument(
        '-o', '--output-dir', type=str, required=True,
        help='Directory to save outputs (models, metrics, figures)'
    )
    parser.add_argument(
        '--use-latents', action='store_true',
        help='Use latent vectors as features instead of reconstruction errors'
    )
    parser.add_argument(
        '--wdm', action='store_true',
        help='Include WDM data alongside CDM'
    )
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument(
        '-n', '--n-steps', type=int, default=30,
        help='Number of sampling steps for reconstruction'
    )
    parser.add_argument(
        '-f', '--fig-dir', type=str, default=None,
        help='Directory to save figures (default: <output-dir>/figures)'
    )
    parser.add_argument(
        '-l', '--latent-save-dir', type=str, default=None,
        help='Directory to save feature arrays'
    )
    parser.add_argument(
        '-s', '--subdir', type=str, default=None,
        help='Subdirectory to save outputs (default: <output-dir>/<subdir>)'
    )
    return parser.parse_args()

def compute_latents(fm, dataset, batch_size):
    latents = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for imgs, _ in tqdm.tqdm(loader, desc="Computing latents"):
        imgs = imgs.cuda()
        with torch.no_grad():
            spatial = fm.encoder(imgs)
            vec = fm.decoder.velocity_model.fc(
                fm.decoder.velocity_model.pool(spatial).squeeze()
            )
        latents.append(vec.cpu().numpy())
    return np.vstack(latents)

def compute_diffs(fm, dataset, batch_size, n_steps, args):
    all_diffs  = []
    all_orig   = []
    all_rec    = []
    all_labels = []

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for imgs, labels in tqdm.tqdm(loader, desc="Computing diffs"):
        imgs = imgs.cuda()
        with torch.no_grad():
            spatial = fm.encoder(imgs)
            recon   = fm.decoder.predict(
                x0=torch.randn_like(imgs),
                h=spatial,
                n_sampling_steps=n_steps
            )
        diff = (imgs - recon).cpu().numpy()
        
        all_diffs.append(diff)
        all_orig.append(imgs.cpu().numpy())
        all_rec.append(recon.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Concatenate across batches
    diffs  = np.concatenate(all_diffs,  axis=0)  # (N, C, H, W)
    origs  = np.concatenate(all_orig,   axis=0)
    recs   = np.concatenate(all_rec,    axis=0)
    labels = np.concatenate(all_labels, axis=0)  # (N, L) or more dims

    # Extract the last label for each sample
    # If labels have extra dims, we flatten except first
    last_lbl = labels.reshape(labels.shape[0], -1)[:, -1]  # shape (N,)

    # Find the index of the sample with the largest last-label value
    sample_idx = int(np.argmax(last_lbl))

    # Pull out that sample
    orig_img  = origs[sample_idx]
    recon_img = recs[sample_idx]
    lbl_value = last_lbl[sample_idx]

    # Helper to shape for imshow
    def prep(img):
        if img.ndim == 3 and img.shape[0] == 1:
            return img[0]
        if img.ndim == 3:
            return np.transpose(img, (1,2,0))
        return img

    orig_plot  = prep(orig_img)
    recon_plot = prep(recon_img)

    # Plot & save
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(orig_plot,  cmap='viridis' if orig_plot.ndim==2 else None)
    axes[0].set_title(f"Original (idx={sample_idx}, last_label={lbl_value})")
    axes[0].axis('off')
    axes[1].imshow(recon_plot, cmap='viridis' if recon_plot.ndim==2 else None)
    axes[1].set_title("Reconstruction")
    axes[1].axis('off')

    plt.tight_layout()
    plt.savefig("max_last_label_reconstruction.png")
    plt.close(fig)

    print(f"Saved reconstruction for sample with max last-label (idx={sample_idx}) "
          f"to 'max_last_label_reconstruction.png'")

    return diffs

def load_or_compute_features(fm, dataset, args, mode: str, label: str):
    """
    mode: 'latents' or 'diffs'
    label: 'cdm' or 'wdm' (used in filename)
    """
    save_folder = os.path.join(args.latent_save_dir, args.subdir)
    os.makedirs(save_folder, exist_ok=True)
    fname = f"{mode}_{label}.npy"
    fpath = os.path.join(save_folder, fname)
    if os.path.exists(fpath):
        X = np.load(fpath)
        print(f"Loaded {mode} from {fpath}")
    else:
        if mode == 'latents':
            X = compute_latents(fm, dataset, args.batch_size)
        else:
            X = compute_diffs(fm, dataset, args.batch_size, args.n_steps, args)
        np.save(fpath, X)
    return X

def split_data(X, y, ratios=(0.6, 0.2, 0.2)):
    n = len(X)
    i1 = int(ratios[0] * n)
    i2 = i1 + int(ratios[1] * n)
    return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])

def make_dataloader(X, y, batch_size, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = args.fig_dir or os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(args.latent_save_dir, exist_ok=True)

    # load representer
    fm = represent.Represent.load_from_checkpoint(args.checkpoint).cuda()
    fm.encoder.eval()

    # prepare datasets
    cdm_params = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2']
    wdm_params = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_AGN2', 'WDM']
    cdm_ds = data.CAMELS(
        idx_list=range(5000), parameters=cdm_params,
        suite="IllustrisTNG", dataset="LH", map_type="Mcdm"
    )
    if args.wdm:
        wdm_ds = data.CAMELS(
            idx_list=range(5000), parameters=wdm_params,
            suite="IllustrisTNG", dataset="WDM", map_type="Mcdm"
        )

    mode = 'latents' if args.use_latents else 'diffs'

    # load or compute features for CDM
    X_cdm = load_or_compute_features(fm, cdm_ds, args, mode, 'cdm')
    y_cdm = np.zeros(len(X_cdm))

    # optionally include WDM
    if args.wdm:
        X_wdm = load_or_compute_features(fm, wdm_ds, args, mode, 'wdm')
        y_wdm = np.ones(len(X_wdm))
        X = np.vstack([X_cdm, X_wdm])
        y = np.concatenate([y_cdm, y_wdm])
    else:
        X, y = X_cdm, y_cdm

    # shuffle to ensure class balance
    perm = np.random.permutation(len(y))
    X = X[perm]
    y = y[perm]

    # split into train/val/test
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_data(X, y)
    
    mu   = X_tr.mean(axis=0, keepdims=True)     # shape (1, C, H, W) or (1, D)
    sigma = X_tr.std(axis=0, keepdims=True) + 1e-8

    # apply to train / val / test
    X_tr = (X_tr - mu) / sigma
    X_val = (X_val - mu) / sigma
    X_te  = (X_te  - mu) / sigma

    # create loaders
    tr_loader = make_dataloader(X_tr, y_tr, args.batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, args.batch_size)
    te_loader = make_dataloader(X_te, y_te, args.batch_size)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # def objective(trial):
    #     lr = trial.suggest_loguniform('lr', 1e-6, 1e-1)
    #     wd = trial.suggest_loguniform('weight_decay', 1e-8, 1e-2)
    #     hidden_dim = trial.suggest_int('hidden', 1, 32)

    #     model = ad.AnomalyDetectorImg(hidden=hidden_dim, dr=0.3, channels=1).to(device)
    #     opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    #     loss_fn = nn.BCEWithLogitsLoss()

    #     for _ in range(20):
    #         model.train()
    #         for xb, yb in tr_loader:
    #             xb, yb = xb.to(device), yb.to(device)
    #             opt.zero_grad()
    #             out = model(xb).squeeze()
    #             l = loss_fn(out, yb)
    #             l.backward()
    #             opt.step()
    #         model.eval()
    #         vl = np.mean([loss_fn(model(xb.to(device)).squeeze(), yb.to(device)).item()
    #                       for xb, yb in val_loader])
    #     return vl

    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective, n_trials=30)
    # best = study.best_params

    # build and train model
    channels = X.shape[1] if mode == 'latents' else 1
    model = ad.AnomalyDetectorImg(hidden=2, dr=0.3, channels=channels).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-7
    )

    patience = 10
    no_improve = 0
    best_val = float('inf')
    best_state = None
    for ep in range(args.epochs):
        model.train()
        train_loss = 0.0
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(Xb).squeeze()
            loss = crit(out, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item()
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(device), yb.to(device)
                val_loss += crit(model(Xb).squeeze(), yb).item()
        val_loss /= len(val_loader)
        sched.step()
        if val_loss < best_val:
            best_val, best_state = val_loss, model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= patience:
            print(f"Stopping early at epoch {ep+1}")
            break
        print(f"Epoch {ep + 1}: train={train_loss/len(tr_loader):.4f}, val={val_loss:.4f}")

    # save best model
    model_path = os.path.join(args.output_dir, 'best_ad_model.pt')
    torch.save(best_state, model_path)

    # test accuracy
    model.load_state_dict(best_state)
    model.eval()
    correct, total = 0, 0
    for Xb, yb in te_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = (torch.sigmoid(model(Xb).squeeze()) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    acc = correct / total * 100

    # save metrics
    metrics_dir = os.path.join(args.latent_save_dir, args.subdir)
    os.makedirs(metrics_dir, exist_ok=True)
    with open(os.path.join(metrics_dir, 'metrics.txt'), 'a+') as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_dir}/metrics.txt")

if __name__ == '__main__':
    main()
