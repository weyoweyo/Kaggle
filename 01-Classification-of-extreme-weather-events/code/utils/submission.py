import numpy as np
import pandas as pd
import os


def check_labels(Y_pred):
    result = []
    for i in range(0, len(Y_pred)):
        if Y_pred[i] > 0:
            result.append((Y_pred[i], i))

    return result


def save_data(Y_pred, path, file_name, ext=".csv"):
    index = np.zeros(len(Y_pred), dtype=np.int)
    for i in range(0, len(Y_pred)):
        if i == 0:
            index[i] = 0
            continue
        index[i] = index[i - 1] + 1
    result = np.vstack([index, Y_pred])
    pd.DataFrame(result.T).to_csv(path + file_name + ext, header=["S.No", "LABELS"], index=False)


def save_data_with_model(submission_file, Y_pred):
    submission_file.iloc[:, 1] = Y_pred
    submission_file.to_csv(index=False)


def save_logs(info_params, accuracy, pred_test):
    path_logs = "../../data/logs"
    path_pred = "../../data/predictions"
    try:
        os.mkdir(path_logs)
        os.mkdir(path_pred)
    except OSError:
        print("Creation of the directory failed. Probably because it is already existing.")
    else:
        print("Successfully created the directory.")

    path, dirs, files_log = next(os.walk(path_logs))
    path, dirs, files_pred = next(os.walk(path_pred))
    file_count_log = len(files_log)
    file_count_pred = len(files_pred)

    with open('../../data/logs/trace_' + str(file_count_log) + '_' + str(accuracy) + '.txt', 'w') as f:
        for i, data in enumerate(info_params):
            f.write(data + '\n')

    save_data(pred_test, '../../data/predictions/', 'sub_' + str(file_count_pred) + '_' + str(accuracy))