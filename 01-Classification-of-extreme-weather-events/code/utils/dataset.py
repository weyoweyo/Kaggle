import pandas as pd
import numpy as np


def get_from_files(train_file, test_file, sub_file, path='../../data/', ext='.csv'):
    train = pd.read_csv(path + train_file + ext)
    test = pd.read_csv(path + test_file + ext)
    submission = pd.read_csv(path + sub_file + ext)

    # Masks
    ## Train data with date greater than year 2000.
    #train = train[train['time'] > 20020000]

    return train, test, submission


def split_train_test_v1(X, y, X_std, training_size):
    m = X.shape[0]
    nb_train = int(m * training_size)
    X_train = X.iloc[0:nb_train, :]
    y_train = y[0:nb_train]
    X_test = X_std.iloc[nb_train:m, :]
    y_test = y[nb_train:m]

    return X_train, y_train, X_test, y_test


def split_train_test(X, y, training_size=0.7, val_size=0.15):
    m = X.shape[0]
    nb_train = int(m * training_size)
    X_train = X.iloc[0:nb_train, :]
    y_train = y[0:nb_train]

    nb_val = int(m * val_size)

    val_index = nb_train + nb_val
    X_val = X.iloc[nb_train: val_index, :]
    y_val = y[nb_train: val_index]

    X_test = X.iloc[val_index: m, :]
    y_test = y[val_index: m]
    return X_train, y_train, X_val, y_val, X_test, y_test


def preprocessing(features, removed_features):
    X = features.copy()
    X = X.iloc[:, 1:19]
    X = X.drop(columns=removed_features)
    #print(X)
    X.insert(0, 'bias', 1)
    X_means = np.mean(X)
    X_std = np.std(X)
    X_scale = (X - X_means) / X_std
    X_scale.iloc[:, 0] = np.ones((X_scale.shape[0], 1))
    return X_scale


def preprocessing_without_scaling(features, removed_features):
    X = features.copy()
    X = X.iloc[:, 1:19]
    X = X.drop(columns=removed_features)
    X.insert(0, 'bias', 1)
    return X


def preprocessing_nn(features, removed_features):
    X = features.copy()
    X = X.drop(columns=removed_features)
    X_means = np.mean(X)
    X_std = np.std(X)
    X_scale = (X - X_means) / X_std
    return X_scale


def feature_scaling(train, all_features, removed_features):
    features_scale = preprocessing(all_features, removed_features)
    train_data = features_scale.iloc[0:train.shape[0], :]
    train_data_y = train.iloc[:, -1]
    test_data = features_scale.iloc[train.shape[0]:features_scale.shape[0], :]

    return train_data, train_data_y, test_data


def month(times):
  month = []
  for each_time in times:
    time_str = str(each_time)
    each_month = time_str[4:6]
    month.append(int(each_month))
  return month


def add_seasons(data, train):
  new_data = data.copy()
  month_list = month(train['time'])
  new_data['winter'] = [1 if (x == 1 or x == 2 or x == 12) else 0 for x in month_list]
  new_data['summer'] = [1 if (x == 6 or x == 7 or x == 8) else 0 for x in month_list]
  new_data['automn'] = [1 if (x == 9 or x == 10 or x == 11) else 0 for x in month_list]
  new_data['spring'] = [1 if (x == 3 or x == 4 or x == 5) else 0 for x in month_list]
  return new_data


def separate_train_test(train, data):
    train_data = data.iloc[0:train.shape[0], :]
    train_data_y = train.iloc[:, -1]
    test_data = data.iloc[train.shape[0]:data.shape[0], :]
    return train_data, train_data_y, test_data
