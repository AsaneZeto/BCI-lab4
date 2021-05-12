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


def load_data(data_path, subject, dev, batch_size=32):
    
    train = io.loadmat(data_path + 'BCIC_' + subject + '_T.mat')
    test = io.loadmat(data_path + 'BCIC_' + subject + '_E.mat')

    x_train = torch.Tensor(train['x_train']).unsqueeze(1).to(dev) # Shape: [288, 1, 22, 562]
    y_train = torch.Tensor(train['y_train']).view(-1).long().to(dev) # Shape [288]
    x_test = torch.Tensor(test['x_test']).unsqueeze(1).to(dev)
    y_test = torch.Tensor(test['y_test']).view(-1).to(dev)


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

    return trainloader, testloader

def print_EEGNet_metric(acc, clf_acc, conf_m):
    print('Accuracy of EEGNet: {:5f}'.format(acc))
    print('-----------------------------------')
    print('Overall Accuracy:\nLabel\tAccuracy')
    for label, acc in zip(range(len(clf_acc)), clf_acc):
        print('{}\t{:5f}'.format(label, acc))
    print('-----------------------------------')
    print('confusion matrix (prediction by Ground Truth):')
    for i in range(len(conf_m)):
        row_i = conf_m[i]
        print('{} {} {} {}'.format(row_i[0], row_i[1], row_i[2], row_i[3]))