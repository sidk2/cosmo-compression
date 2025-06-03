import matplotlib.pyplot as plt
import numpy as np
import scienceplots

# Use the science style and set base font sizes
plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({
    "font.size": 16,           # Base font size for labels, ticks, legend
    "axes.titlesize": 20,      # Font size for the title
    "axes.labelsize": 14,      # Font size for x/y labels
    "xtick.labelsize": 12,     # Font size for x‐tick labels
    "ytick.labelsize": 12,     # Font size for y‐tick labels
    "legend.fontsize": 14      # Font size for legend
})

# Data
x = [4, 8, 12, 16, 20, 64]
# omega_m_pct_error = np.array([0.1493, 0.0524, 0.0884, 0.0372, 0.0594, 0.0519]) * 100 # IllustrisTNG
# sigma_8_pct_error = np.array([0.0491, 0.0403, 0.0353, 0.0316, 0.0319, 0.0375]) * 100 # IllustrisTNG
omega_m_pct_error = np.array([0.1580, 0.0471, 0.0869, 0.0372, 0.0591, 0.0437]) * 100 # Astrid
sigma_8_pct_error = np.array([0.0452, 0.0382, 0.0306, 0.0300, 0.0288, 0.0361]) * 100 # Astrid
mse = [0.407358, 0.312346, 0.318532, 0.297846, 0.269314, 0.178118] # Astrid

# Parameter Estimation network achieves 0.0818 for Omega_m, 0.0499 for sigma_8 on WDM without training
# Achieves Omega_m: 0.0651, sigma_8: 0.0491 with training but no fine tuning

fig, ax1 = plt.subplots(figsize=(8, 6))

# Plot percent errors on the left y-axis
lns1 = ax1.plot(
    x,
    omega_m_pct_error,
    label=r"$\Omega_m$",
    marker="o",
    color="C0"
)
lns2 = ax1.plot(
    x,
    sigma_8_pct_error,
    label=r"$\sigma_8$",
    marker="o",
    color="C1"
)
ax1.set_xlabel("Number of Channels")
ax1.set_ylabel("Parameter Inference Percent Error")
ax1.tick_params(axis="y")

# Create a second y-axis for MSE
ax2 = ax1.twinx()
lns3 = ax2.plot(
    x,
    mse,
    label="MSE",
    marker="s",
    linestyle="--",
    color="C2"
)
ax2.set_ylabel("MSE")
ax2.tick_params(axis="y")

# Combine legends from both axes
lns = lns1 + lns2 + lns3
labels = [l.get_label() for l in lns]
ax1.legend(lns, labels, loc="upper right")

plt.title("Percent Error and MSE vs Number of Channels")
plt.tight_layout()
plt.savefig("cosmo_compression/results/workshop_figures/parameter_estimation_with_mse.png")
