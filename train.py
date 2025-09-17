import torch
import numpy as np
from maf import MAF
from made import MADE
from data_loaders import get_data, get_data_loaders
from utils.train import train_one_epoch_maf, train_one_epoch_made
from utils.validation import val_maf, val_made
from utils.test import test_maf, test_made
from utils.plot_backup_0909copy import sample_drellyan_maf, plot_hists
import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend for remote servers
import matplotlib.pyplot as plt
import os


# --------- SET PARAMETERS ----------
model_name = "maf"  # 'MAF' or 'MADE'
dataset_name = "drellyan"
batch_size = 1024
n_mades = 10
hidden_dims = [128]
lr = 3e-4
random_order = False
patience = 100  # For early stopping
seed = 290713
plot = True
max_epochs = 3
save_dir = "plots"
os.makedirs(save_dir, exist_ok=True)
feature_lows = torch.tensor([0, 0], dtype=torch.float32) #[0, 0, -np.pi, -np.pi, -2.5, -2.5], dtype=torch.float32)    
feature_highs = torch.tensor([700, 700], dtype=torch.float32)  #[700, 700, np.pi, np.pi, 2.5, 2.5], dtype=torch.float32)
# -----------------------------------

# Get dataset.
data = get_data(dataset_name)
train = torch.from_numpy(data.x_train)
# Get data loaders.
train_loader, val_loader, test_loader = get_data_loaders(data, batch_size)
# Get model.
n_in = data.n_dims
if model_name.lower() == "maf":
    model = MAF(n_in, n_mades, hidden_dims, feature_lows, feature_highs)
elif model_name.lower() == "made":
    model = MADE(n_in, hidden_dims, random_order=random_order, seed=seed, gaussian=True)
# Get optimiser.
optimiser = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

# Format name of model save file.
save_name = f"{model_name}_{dataset_name}_{'_'.join(str(d) for d in hidden_dims)}.pt"
# Initialise list for plotting.
epochs_list = []
train_losses = []
val_losses = []
# Initialise early stopping.
i = 0
max_loss = np.inf

# Training loop.
for epoch in range(1, max_epochs):
    # Train and validate
    if model_name == "maf":
        train_loss = train_one_epoch_maf(model, epoch, optimiser, train_loader)
        val_loss = val_maf(model, train, val_loader)
    elif model_name == "made":
        train_loss = train_one_epoch_made(model, epoch, optimiser, train_loader)
        val_loss = val_made(model, val_loader)

    if plot:
        samples = sample_drellyan_maf(model, n_in)
        plot_hists(samples, val_loader, epoch)

    #Make sure too big losses are capped at 150 for the loss plot to make sense
    max_plot_loss = 150
    epochs_list.append(epoch)
    if train_loss <= max_plot_loss:
        train_losses.append(train_loss)
    else:
        train_losses.append(max_plot_loss)
    if val_loss <=max_plot_loss:
        val_losses.append(val_loss)
    else:
        val_losses.append(max_plot_loss)

    # Early stopping. Save model on each epoch with improvement.
    if val_loss < max_loss:
        i = 0
        max_loss = val_loss
        torch.save(model, "model_saves/" + save_name)
    else:
        i += 1

    if i < patience:
        print(f"Patience counter: {i}/{patience}", flush = True)
    else:
        print(f"Patience counter: {i}/{patience}\nTerminate training!", flush = True)
        break

    # Save loss plot every 10 epochs
    if epoch % 10 == 0:
        plt.figure(figsize=(8,5))
        plt.plot(epochs_list, train_losses, label='Train Loss', color='blue')
        plt.plot(epochs_list, val_losses, label='Validation Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'loss_plot_epoch_{epoch}.png'))
        plt.close()  # Close figure to save memory

# Save final loss plot
plt.figure(figsize=(8,5))
plt.plot(epochs_list, train_losses, label='Train Loss', color='blue')
plt.plot(epochs_list, val_losses, label='Validation Loss', color='orange')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(save_dir, f'loss_plot_final.png'))
plt.close()


