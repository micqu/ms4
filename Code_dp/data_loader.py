import gzip
import pickle
from typing import List
import numpy as np
from sklearn.preprocessing import StandardScaler

import pandas as pd
from sklearn import model_selection
import torch as th
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.dataset import Dataset, Subset


def load_data_boston():
    with open('../data/boston_housing.pickle','rb') as f:
        ((X, y), (X_test, y_test)) = pickle.load(f)

    X = th.from_numpy(X).float()
    y = th.from_numpy(y).float()
    X_test = th.from_numpy(X_test).float()
    y_test = th.from_numpy(y_test).float()

    # preprocessing
    mean = X.mean(0, keepdim=True)
    dev = X.std(0, keepdim=True)
    mean[:, 3] = 0. # the feature at column 3 is binary,
    dev[:, 3] = 1.  # so we don't standardize it
    X = (X - mean) / dev
    X_test = (X_test - mean) / dev

    print(X.shape, y.shape, X_test.shape, y_test.shape)

    train = TensorDataset(X, y)
    test = TensorDataset(X_test, y_test)

    return train, test, X, y, X_test, y_test


def load_data_mnist():
    f = gzip.open('../data/mnist.pkl.gz', 'rb')
    (X, y), (X_test, y_test) = pickle.load(f, encoding="bytes")
    
    # Count labels
    train_idxs = []
    test_idxs = []
    for k in range(10):
        idxs_k = np.where(y == k)[0]
        train_idxs.extend(idxs_k[:160])
        idxs_k = np.where(y_test == k)[0]
        test_idxs.extend(idxs_k[:40])

    np.random.shuffle(train_idxs)
    np.random.shuffle(test_idxs)

    X = X[train_idxs]
    y = y[train_idxs]
    X_test = X_test[test_idxs]
    y_test = y_test[test_idxs]
    
    X_all = np.concatenate((X, X_test))
    y_all = np.concatenate((y, y_test))


    X = th.from_numpy(X).float()
    y = th.from_numpy(y).long()
    X_test = th.from_numpy(X_test).float()
    y_test = th.from_numpy(y_test).long()

    X_all = th.from_numpy(X_all).float()
    y_all = th.from_numpy(y_all).long()

    # preprocessing
    mean = X.mean()
    dev = X.std()
    X = (X - mean) / dev
    X_test = (X_test - mean) / dev

    X = X.view(-1, 784)
    X_test = X_test.view(-1, 784)

    X_all = (X_all - mean) / dev

    print(X.shape, y.shape, X_test.shape, y_test.shape)

    train = TensorDataset(X, y)
    test = TensorDataset(X_test, y_test)
    all_data = TensorDataset(X_all, y_all)

    return (train, test, X, y, X_test, y_test, all_data)


def load_data_completes():
    df = pd.read_csv("../data/data_all_2020_12_10/processed/completes_with_embedding_no_zeros.csv") 

    df = df.loc[:, (df != 0).any(axis=0)]

    X = df.drop("Complete", axis=1)
    y = df["Complete"]

    X, X_test, y, y_test = model_selection.train_test_split(X, y, test_size=0.20, random_state=12346)

    X = np.array(X)
    X_test = np.array(X_test)
    y = np.array(y)
    y_test = np.array(y_test)

    # preprocessing
    # X, X_test, y, y_test, scaler = scale_data_for_embedder(X, X_test, y, y_test, 17)
    X, X_test, y, y_test, scaler = scale_data_for_embedder(X, X_test, y, y_test, 8)

    X = th.from_numpy(X).float()
    y = th.from_numpy(y).long()
    X_test = th.from_numpy(X_test).float()
    y_test = th.from_numpy(y_test).long()

    X_all = np.concatenate((X, X_test))
    y_all = np.concatenate((y, y_test))
    X_all = th.from_numpy(X_all).float()
    y_all = th.from_numpy(y_all).long()

    train = TensorDataset(X, y)
    test = TensorDataset(X_test, y_test)
    all_data = TensorDataset(X_all, y_all)
    
    return (train, test, X, y, X_test, y_test, all_data)

def scale_data_for_embedder(X_train: np.ndarray, X_valid: np.ndarray,
                            y_train: np.ndarray, y_valid: np.ndarray,
                            n_num_cols: int):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train[:,:n_num_cols])
    X_valid_sc = scaler.transform(X_valid[:,:n_num_cols])
    X_train = np.concatenate([X_train_sc, X_train[:,n_num_cols:]], axis=1)
    X_valid = np.concatenate([X_valid_sc, X_valid[:,n_num_cols:]], axis=1)

    return X_train, X_valid, y_train, y_valid, scaler

def get_data_loader(data: Dataset, batch_size: int, kwargs, data_idxs: List[int] = None, shuffle=True) -> DataLoader:
    if data_idxs is None or any(data_idxs) == False:
        loader = DataLoader(data, batch_size=batch_size, shuffle=shuffle, **kwargs)
    else:
        subset = Subset(data, data_idxs)
        loader = DataLoader(subset, batch_size=batch_size, shuffle=shuffle, **kwargs)
        
    return loader

def get_column_names(path: str):
    df = pd.read_csv(path)
    return list(df.columns)