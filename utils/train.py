import math
import numpy as np
import torch
import torch.nn as nn


def train_one_epoch_maf(model, epoch, optimizer, train_loader, device):           
    model.train()
    train_loss = 0
    for batch in train_loader:
        batch = batch.to(device).float()  # Move batch to GPU
        u, log_det = model.forward(batch.float())
    
        negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
        negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
        negloglik_loss -= log_det
        negloglik_loss = torch.mean(negloglik_loss)

        negloglik_loss.backward()
        train_loss += negloglik_loss.item()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = train_loss / len(train_loader)
    print("Epoch: {} Average loss: {:.5f}".format(epoch, avg_loss))
    return avg_loss

