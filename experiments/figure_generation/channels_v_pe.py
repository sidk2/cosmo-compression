import matplotlib.pyplot as plt
import numpy as np
import scienceplots

plt.style.use("science")
plt.rc("text", usetex=True)
plt.rc("font", family="serif")

x = [4, 8, 12, 16, 20, 64]
y = np.array([0.1500, 0.1122, 0.1052, 0.0817, 0.0742, 0.0478])
z = np.array([0.0464, 0.0387, 0.0396, 0.0379, 0.0380, 0.0358])

plt.figure(figsize=(8, 5))
plt.plot(x, y*100, label=r"$\Omega_m$")
plt.plot(x, z*100, label=r"$\sigma_8$")
plt.xlabel("Number of Channels")
plt.ylabel("Percent Error")

plt.title("Percent Error vs Number of Channels")
plt.legend()
plt.tight_layout()
plt.savefig("cosmo_compression/results/workshop_figures/channels_v_pe.png")