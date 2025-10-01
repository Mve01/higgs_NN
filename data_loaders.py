import torch
from drellyan import DrellYanDataset


def get_data(list_data_features):   #remove dataset !!!!
    return DrellYanDataset(list_data_features)


def get_data_loaders(data, batch_size: int = 128):
    train = torch.from_numpy(data.x_train)
    val = torch.from_numpy(data.x_val)
    test = torch.from_numpy(data.x_test)

    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle = True)
    val_loader = torch.utils.data.DataLoader(val, batch_size=batch_size, shuffle = True)
    test_loader = torch.utils.data.DataLoader(test, batch_size=batch_size, shuffle = True)
    return train_loader, val_loader, test_loader
