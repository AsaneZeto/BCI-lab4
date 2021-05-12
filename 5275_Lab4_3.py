import torch
import torch.utils.data as Data
import numpy as np
import scipy.io as io

def split_train_valid_set(x_train, y_train, **kwargs):

    if 'valid_split_rate' in kwargs.keys():
        valid_split_rate = kwargs['valid_split_rate']
    else:
        valid_split_rate = 0

    n_train = int(x_train.shape[0] * (1 - valid_split_rate))
    idx = torch.randperm(x_train.shape[0])
    x_train = x_train[idx]
    y_train = y_train[idx]

    return x_train[:n_train], y_train[:n_train], x_train[n_train:], y_train[n_train:]


def load_data(data_dir, subject, batch_size=32, split=False, valid_split_rate=0):
    
    train = io.loadmat(data_dir + 'BCIC_' + subject + '_T.mat')
    test = io.loadmat(data_dir + 'BCIC_' + subject + '_E.mat')

    x_train = torch.Tensor(train['x_train']).unsqueeze(1) # Shape: [288, 1, 22, 562]
    y_train = torch.Tensor(train['y_train']).view(-1) # Shape [288]
    x_test = torch.Tensor(test['x_test']).unsqueeze(1)
    y_test = torch.Tensor(test['y_test']).view(-1)

    if split:
        if valid_split_rate < 0 or valid_split_rate >= 1:
            raise ValueError('Unexpected Valid_split_rate.')
        x_train, y_train, x_valid, y_valid = split_train_valid_set(x_train, y_train, valid_split_rate)
        valid_dataset = Data.TensorDataset(x_valid, y_valid)
        validloader = Data.DataLoader(
            dataset = valid_dataset,
            batch_size = 1,
            shuffle = False,
            num_workers = 0,
        )

    train_dataset = Data.TensorDataset(x_train, y_train)
    test_dataset = Data.TensorDataset(x_test, y_test)

    trainloader = Data.DataLoader(
        dataset = train_dataset,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 0,
    )

    testloader =  Data.DataLoader(
        dataset = test_dataset,
        batch_size = 1,
        shuffle = False,
        num_workers = 0,
    )

    return trainloader, validloader, testloader
