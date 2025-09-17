import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.distributions import MultivariateNormal

def sample_drellyan_maf(model):
    model.eval()
    n_samples = 1000
    
    u = torch.zeros(n_samples, 6).normal_(0, 1)
    mvn = MultivariateNormal(torch.zeros(6), torch.eye(6))
    log_prob = mvn.log_prob(u)
    samples, log_det = model.backward(u)

    # Remove NaNs/Infs across the whole tensor
    mask_inf = torch.isfinite(samples).all(dim=-1)
    samples = samples[mask_inf]  


    samples = samples.detach().cpu()
    return samples


def plot_hists(samples, validation, epoch):
    val_batches = []
    for batch in validation:
        if isinstance(batch, (list, tuple)):
            val_batches.append(batch[0])  # assume (data, labels)
        else:
            val_batches.append(batch)     # assume only data
    validation = torch.cat(val_batches).cpu().numpy()

    _, n_features = samples.shape
    # Create subplots in a grid
    n_cols = 3  
    n_rows = int(np.ceil(n_features / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))

    axes = axes.ravel()  # flatten to 1D array for easy indexing

    x_labels = ["Muons_Eta_Lead", "Muons_Eta_Sub", "Muons_PT_Lead", "Muons_PT_Sub", "Muons_Phi_Lead", "Muons_Phi_Sub"]
    print(f"n_samples: {len(samples)}")

    for i in range(n_features):
    # Compute common bins across both distributions
        all_data = np.concatenate([samples[:, i], validation[:, i]])
        bins = np.linspace(all_data.min(), all_data.max(), 50)

        axes[i].hist(samples[:, i], bins=bins, histtype="step", color="steelblue",
             linewidth=2, density=True, label="Samples")
        axes[i].hist(validation[:, i], bins=bins, histtype="step", color="red",
             linewidth=2, density=True, label="Validation")

        axes[i].set_xlabel(x_labels[i])
        axes[i].set_ylabel("Density")
        axes[i].legend()

    # Remove unused axes if n_features doesn’t fill the grid
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    safe_folder = "train_hists" 
    if not os.path.exists(safe_folder):
        os.mkdir(safe_folder)
    save_path = f"{safe_folder}/hist_{epoch}.png"
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_losses(epochs, train_losses, val_losses, title=None):
    sns.set(style="white")
    fig, axes = plt.subplots(
        ncols=1, nrows=1, figsize=[10, 5], sharey=True, sharex=True, dpi=400
    )

    train = pd.Series(train_losses).astype(float)
    val = pd.Series(val_losses).astype(float)
    train.index += 1
    val.index += 1

    axes = sns.lineplot(data=train, color="gray", label="Training loss")
    axes = sns.lineplot(data=val, color="orange", label="Validation loss")

    axes.set_ylabel("Negative log-likelihood")
    axes.legend(
        frameon=False,
        prop={"size": 14},
        fancybox=False,
        handletextpad=0.5,
        handlelength=1,
    )
    axes.set_ylim(1250, 1600)
    axes.set_xlim(0, 50)
    axes.set_title(title) if title is not None else axes.set_title(None)
    if not os.path.exists("plots"):
        os.makedirs("plots")
    save_path = "plots/train_plots" + str(epochs[-1]) + ".pdf"
    plt.savefig(
        save_path, dpi=300, bbox_inches="tight", pad_inches=0,
    )
    plt.close()
