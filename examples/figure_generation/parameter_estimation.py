import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import matplotlib.lines as mlines

# Use the science style and set base font sizes
plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")
plt.rcParams.update({
    "font.size": 20,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 20,
    "ytick.labelsize": 20,
    "legend.fontsize": 20
})

# Data
x = [4, 8, 12, 16, 20, 64]

# IllustrisTNG data
# omega_m_pct_error = np.array([0.1398, 0.0519, 0.0858, 0.0405, 0.0577, 0.0427]) * 100
# omega_m_stdev      = np.array([0.0475, 0.0387, 0.0790, 0.0312, 0.0462, 0.0329]) * 100 / np.sqrt(1000)
# sigma_8_pct_error  = np.array([0.1253, 0.0424, 0.0359, 0.0339, 0.0326, 0.0410]) * 100 
# sigma_8_stdev      = np.array([0.0377, 0.0344, 0.0284, 0.0271, 0.0266, 0.0321]) * 100 / np.sqrt(1000)
# mse = []
# mse_stdev = [] / np.sqrt(1000)
# Astrid Data
omega_m_pct_error = np.array([0.1626, 0.0556, 0.0914, 0.0397, 0.0581, 0.0412]) * 100
omega_m_stdev      = np.array([0.1451, 0.0465, 0.0801, 0.0312, 0.0462, 0.0329]) * 100 / np.sqrt(1000)
sigma_8_pct_error  = np.array([0.0453, 0.0377, 0.0325, 0.0300, 0.0291, 0.0372]) * 100 
sigma_8_stdev      = np.array([0.0351, 0.0305, 0.0271, 0.0241, 0.0245, 0.0294]) * 100 / np.sqrt(1000)
mse = [0.407433, 0.312505, 0.319278, 0.297564,  0.269318, 0.178145]
mse_stdev = [0.072062,  0.058970, 0.061146,  0.057043,0.053485, 0.037183] / np.sqrt(1000)

fig, ax1 = plt.subplots(figsize=(7, 5))

# Plot percent errors with horizontal caps
ax1.errorbar(
    x,
    omega_m_pct_error,
    yerr=omega_m_stdev,
    capsize=5,
    marker="o",
    color="C0"
)
ax1.errorbar(
    x,
    sigma_8_pct_error,
    yerr=sigma_8_stdev,
    capsize=5,
    marker="o",
    color="C1"
)
ax1.set_xlabel("Number of Channels")
ax1.set_ylabel("Percent Error")

# Secondary axis for MSE
ax2 = ax1.twinx()
ax2.errorbar(
    x,
    mse,
    yerr=mse_stdev,
    capsize=5,
    marker="s",
    linestyle="--",
    color="C2"
)
ax2.set_ylabel("MSE")

# Create proxy artists for the legend
proxy_om = mlines.Line2D([], [], color='C0', marker='o', linestyle='None', label=r"$\Omega_m$")
proxy_s8 = mlines.Line2D([], [], color='C1', marker='o', linestyle='None', label=r"$\sigma_8$")
proxy_mse = mlines.Line2D([], [], color='C2', marker='s', linestyle='--', label="MSE")

ax1.legend(handles=[proxy_om, proxy_s8, proxy_mse], loc="upper right")

plt.title("Percent Error and MSE vs Number of Channels")
plt.tight_layout()
plt.savefig("cosmo_compression/results/workshop_figures/parameter_estimation_with_mse.pdf")
