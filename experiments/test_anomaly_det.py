import os
import argparse
import numpy as np
import torch
import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from cosmo_compression.data import data
from cosmo_compression.model import represent
from cosmo_compression.downstream import anomaly_det_model as ad

def parse_args():
    parser = argparse.ArgumentParser(description="Anomaly detection on CAMELS latents or image reconstruction errors")
    parser.add_argument('-c', '--checkpoint', type=str, required=True,
                        help='Path to pretrained representer checkpoint (.ckpt)')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Directory to save outputs (models, metrics, figures)')
    parser.add_argument('--use-latents', action='store_true',
                        help='Use latent vectors as features instead of image reconstruction errors')
    parser.add_argument('--wdm', action='store_true',
                        help='Include WDM data alongside CDM')
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('-n', '--n-steps', type=int, default=30,
                        help='Number of sampling steps for reconstruction')
    parser.add_argument('-f', '--fig-dir', type=str, default=None,
                        help='Directory to save figures (default: <output-dir>/figures)')
    return parser.parse_args()


def compute_latents(fm, dataset, batch_size):
    latents, params = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for imgs, cosmo in tqdm.tqdm(loader, desc="Computing latents"):
        imgs = imgs.cuda()
        with torch.no_grad():
            spatial = fm.encoder(imgs)
            vec = fm.decoder.velocity_model.fc(fm.decoder.velocity_model.pool(spatial).squeeze())
        latents.append(vec.cpu().numpy())
        params.append(cosmo.numpy())
    return np.vstack(latents), np.vstack(params)


def compute_diffs(fm, dataset, batch_size, n_steps):
    diffs, params = [], []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    for imgs, cosmo in tqdm.tqdm(loader, desc="Computing reconstruction diffs"):
        imgs = imgs.cuda()
        with torch.no_grad():
            spatial = fm.encoder(imgs)
            vec = fm.decoder.velocity_model.fc(fm.decoder.velocity_model.pool(spatial).squeeze())
            # generate reconstruction via decoder.predict
            recon = fm.decoder.predict(x0=torch.randn_like(imgs), h=spatial,
                                      n_sampling_steps=n_steps)
        diff = (imgs - recon).cpu().numpy()
        # flatten per-sample
        diffs.append(diff.reshape(diff.shape[0], -1))
        params.append(cosmo.numpy())
    return np.vstack(diffs), np.vstack(params)


def split_data(X, y, ratios=(0.6, 0.2, 0.2)):
    n = len(X)
    i1 = int(ratios[0]*n)
    i2 = i1 + int(ratios[1]*n)
    return (X[:i1], y[:i1]), (X[i1:i2], y[i1:i2]), (X[i2:], y[i2:])


def make_dataloader(X, y, batch_size, shuffle=False):
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    ds = TensorDataset(X_t.unsqueeze(1), y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    fig_dir = args.fig_dir or os.path.join(args.output_dir, 'figures')
    os.makedirs(fig_dir, exist_ok=True)

    # load representer
    fm = represent.Represent.load_from_checkpoint(args.checkpoint)
    fm.encoder = fm.encoder.cuda()
    fm.encoder.eval()

    # prepare datasets
    cdm_params = ['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2']
    wdm_params = ['Omega_m', 'sigma_8', 'A_SN1', 'A_AGN1', 'A_AGN2', 'WDM']
    cdm_ds = data.CAMELS(idx_list=range(15000), parameters=cdm_params,
                         suite="IllustrisTNG", dataset="LH", map_type="Mcdm")
    if args.wdm:
        wdm_ds = data.CAMELS(idx_list=range(15000), parameters=wdm_params,
                             suite="IllustrisTNG", dataset="WDM", map_type="Mcdm")

    # compute or load features
    if args.use_latents:
        X_cdm, y_cdm = compute_latents(fm, cdm_ds, args.batch_size)
    else:
        X_cdm, y_cdm = compute_diffs(fm, cdm_ds, args.batch_size, args.n_steps)
    y_cdm = np.zeros(len(X_cdm))

    if args.wdm:
        if args.use_latents:
            X_wdm, y_wdm = compute_latents(fm, wdm_ds, args.batch_size)
        else:
            X_wdm, y_wdm = compute_diffs(fm, wdm_ds, args.batch_size, args.n_steps)
        y_wdm = np.ones(len(X_wdm))
        X = np.vstack([X_cdm, X_wdm])
        y = np.concatenate([y_cdm, y_wdm])
    else:
        X, y = X_cdm, y_cdm

    # split into train/val/test
    (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = split_data(X, y)

    # create loaders
    tr_loader = make_dataloader(X_tr, y_tr, args.batch_size, shuffle=True)
    val_loader = make_dataloader(X_val, y_val, args.batch_size)
    te_loader = make_dataloader(X_te, y_te, args.batch_size)

    # build model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    in_dim = X.shape[1]
    model = ad.ADVec(hidden_dim=200, num_hiddens=3, in_dim=in_dim, output_size=1).to(device)
    crit = nn.BCEWithLogitsLoss()
    opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-7)

    # training loop
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
            loss.backward(); opt.step()
            train_loss += loss.item()
        # validation
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
        if ep % 10 == 0:
            print(f"Epoch {ep+1}: train={train_loss/len(tr_loader):.4f}, val={val_loss:.4f}")

    # save best model and metrics
    model_path = os.path.join(args.output_dir, 'best_ad_model.pt')
    torch.save(best_state, model_path)

    # test evaluation
    model.load_state_dict(best_state)
    model.eval()
    correct = 0
    total = 0
    for Xb, yb in te_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        preds = (torch.sigmoid(model(Xb).squeeze()) > 0.5).float()
        correct += (preds == yb).sum().item()
        total += yb.size(0)
    acc = correct / total * 100

    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Test Accuracy: {acc:.2f}%\n")

    print(f"Saved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")

if __name__ == '__main__':
    main()