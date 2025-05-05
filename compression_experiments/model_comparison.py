import pickle
import matplotlib.pyplot as plt

with open("hyperprior_results.pkl", "rb") as f:
    hp_results = pickle.load(f)

with open("fm_results.pkl", "rb") as f:
    fm_results = pickle.load(f)
def reorganize_results(results):
    out = {}
    for key in results[0].keys():
        out[key] = [results[i][key] for i in range(len(results))]
    
    return out

hp_results = reorganize_results(hp_results)
fm_results = reorganize_results(fm_results)

print(hp_results["compress_bpp"])
print(hp_results["log_spectrum_MSE"])
print(hp_results["field_MSE"])

print(fm_results["compress_bpp"])
print(fm_results["log_spectrum_MSE"])
print(fm_results["field_MSE"])

fig, axes = plt.subplots(1, 2, figsize=(8,3))
axes[0].plot(hp_results["compress_bpp"], hp_results["log_spectrum_MSE"], label = "VAE compression")
axes[0].plot(fm_results["compress_bpp"], fm_results["log_spectrum_MSE"], label = "FM compression")
axes[0].legend()

axes[1].plot(hp_results["compress_bpp"], hp_results["field_MSE"], label = "VAE compression")
axes[1].plot(fm_results["compress_bpp"], fm_results["field_MSE"], label = "FM compression")
axes[1].legend()

plt.savefig("cosmo_compression/compression_experiments/model_comparison.png")