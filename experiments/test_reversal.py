from torch.utils.data import DataLoader

import Pk_library as PKL
import time
import torch
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import shapiro, kstest, normaltest, probplot


os.environ["CUDA_VISIBLE_DEVICES"] = "5" 

from cosmo_compression.data.data import CAMELS
from cosmo_compression.model.represent import Represent 


data = CAMELS(
        # idx_list=range(15000),
        map_type='Mcdm',
        dataset='1P',
        parameters=['Omega_m', 'sigma_8', 'A_SN1', 'A_SN2', 'A_AGN1', 'A_AGN2','Omega_b'],
    )
loader = DataLoader(
    data,
    batch_size=1,
    shuffle=False,
    num_workers=1,
    pin_memory=True,
)

fm = Represent.load_from_checkpoint("soda-comp/step=step=3500-val_loss=0.445.ckpt").cuda()
fm.reverse = True
fm.decoder.reverse = True
fm.eval()

gts = []
for cosmology, y in loader:
    if (cosmology.cpu() == torch.Tensor([[0.1, 0.8, 1.00000, 1.00000, 1.00000, 1.00000, 0.04900]])).all() and len(gts) == 0:
        h = fm.encoder(y.cuda()).cuda().unsqueeze(0)
        gts.append((cosmology, y, h))
        x1 = y.cuda()
        x0 = fm.decoder.predict(
            x1.cuda(),
            h=h,
            n_sampling_steps=10*fm.hparams.n_sampling_steps,
        )
        data = x0.reshape(-1).detach().cpu().numpy()
        shapiro_test = shapiro(data)
        kstest_result = kstest(data, 'norm', args=(np.mean(data), np.std(data)))
        dagostino_test = normaltest(data)

        # Create a single figure
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle("Gaussian Distribution Analysis", fontsize=16)

        # Histogram
        axs[0, 0].hist(data, bins=40, density=True, alpha=0.6, color='g')
        axs[0, 0].set_title("Histogram")
        axs[0, 0].set_xlabel("Data")
        axs[0, 0].set_ylabel("Density")

        # Q-Q Plot
        probplot(data, dist="norm", plot=axs[0, 1])
        axs[0, 1].set_title("Q-Q Plot")

        # Statistical Test Results
        results_text = (
            f"Shapiro-Wilk Test:\n  Statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}\n\n"
            f"Kolmogorov-Smirnov Test:\n  Statistic={kstest_result.statistic:.4f}, p-value={kstest_result.pvalue:.4f}\n\n"
            f"D'Agostino and Pearson Test:\n  Statistic={dagostino_test.statistic:.4f}, p-value={dagostino_test.pvalue:.4f}"
        )
        axs[1, 0].axis('off')  # Hide axis for text box
        axs[1, 0].text(0.1, 0.5, results_text, fontsize=10, va='center', ha='left', wrap=True)

        # Add a placeholder for additional visualizations or notes
        axs[1, 1].axis('off')
        axs[1, 1].text(0.5, 0.5, "Add more visualizations or notes here!", fontsize=10, va='center', ha='center')

        # Adjust layout
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig("cosmo_compression/results/gaussianity.png")
        
        fm.reverse = False
        fm.decoder.reverse = False
        
        pred = fm.decoder.predict(
            x0.cuda(),
            h=h,
            n_sampling_steps=fm.hparams.n_sampling_steps,
        ).cpu().numpy()
        
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(x1.cpu().numpy()[0, 0, :, :])
        ax[0].set_title("Ground Truth")
        
        ax[1].imshow(pred[0, 0, :, :])
        ax[1].set_title("Reversal Generated")
        plt.savefig("cosmo_compression/results/reverse.png")
        break

        
