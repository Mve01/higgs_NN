import os
import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.distributions import MultivariateNormal

def sample_drellyan_maf(model, n_in):
    model.eval()
    n_samples = 1000
    
    u = torch.zeros(n_samples, n_in).normal_(0, 1)
    mvn = MultivariateNormal(torch.zeros(n_in), torch.eye(n_in))
    log_prob = mvn.log_prob(u)
    samples, log_det = model.backward(u)

    # Remove NaNs/Infs across the whole tensor
    mask_inf = torch.isfinite(samples).all(dim=-1)
    samples = samples[mask_inf]  

    samples = samples.detach().cpu()
    return samples


def plot_hists(samples, validation, epoch, list_data_features):
    val_batches = []
    for batch in validation:
            val_batches.append(batch)  
    validation = torch.cat(val_batches).cpu().numpy()

    _, n_features = samples.shape
    # Create subplots in a grid
    n_cols = 2  
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.ravel()  # flatten to 1D array for easy indexing

    print(f"n_samples: {len(samples)}", flush = True)

    for i in range(n_features):
    # Compute common bins across both distributions
        all_data = np.concatenate([samples[:, i], validation[:, i]])
        bins = np.linspace(all_data.min(), all_data.max(), 50)

        axes[i].hist(samples[:, i], bins=bins, histtype="step", color="steelblue",
             linewidth=2, density=True, label="Samples")
        axes[i].hist(validation[:, i], bins=bins, histtype="step", color="red",
             linewidth=2, density=True, label="Validation")

        axes[i].set_xlabel(list_data_features[i])
        axes[i].set_ylabel("Density")
        axes[i].legend()

    # Add titles to each column
    column_titles = ["Lead muon", "Sub muon"]
    for col in range(n_cols):
        axes[col].set_title(column_titles[col])

    fig.tight_layout()
    safe_folder = "train_hists" 
    if not os.path.exists(safe_folder):
        os.mkdir(safe_folder)
    save_path = f"{safe_folder}/hist_{epoch}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
