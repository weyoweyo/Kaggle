import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def get_from_files(train_file, test_file, sub_file, path='../../../data/', ext='.csv'):
    train = pd.read_csv(path + train_file + ext)
    test = pd.read_csv(path + test_file + ext)
    submission = pd.read_csv(path + sub_file + ext)
    return train, test, submission


def get_train(train):
    n_features = train.shape[1]
    x_train = train.iloc[:, 1:n_features-1]
    y_train = train.iloc[:, -1]
    return x_train, y_train


def get_test(test):
    return test.iloc[:, 1:]


def standard_scaler(features):
    x = features.copy()
    model = StandardScaler()
    return model.fit_transform(x)


def custom_standard_scaler_train(features):
    x = features.copy()
    x_means = np.mean(x)
    x_std = np.std(x)
    x_scale = (x - x_means) / x_std
    return x_means, x_std, x_scale


def custom_standard_scaler_test(test, train_mean, train_std):
    x = test.copy()
    test_scale = (x - train_mean) / train_std
    return test_scale