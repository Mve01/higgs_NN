import math
import numpy as np
import torch


def val_maf(model, train, val_loader):
    model.eval()
    val_loss = []
    with torch.no_grad():
        for batch in val_loader:
            u, log_det = model.forward(batch.float())
            negloglik_loss = 0.5 * (u ** 2).sum(dim=1)
            negloglik_loss += 0.5 * batch.shape[1] * np.log(2 * math.pi)
            negloglik_loss -= log_det
            val_loss.extend(negloglik_loss.tolist())

    N = len(val_loader.dataset)
    loss = np.sum(val_loss) / N
    print("Validation loss: {:.4f} +/- {:.4f}".format(
            loss, 2 * np.std(val_loss) / np.sqrt(N)), flush = True)
    return loss
