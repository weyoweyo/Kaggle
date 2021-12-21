import numpy as np


def accuracy_percent(X_test, y_test, theta):
    X_test_array = X_test.to_numpy()
    mat = X_test_array.dot(theta.T)
    y_pred = np.argmax(mat, axis=1)
    y_test_array = y_test.to_numpy()
    accuracy_rate = np.sum(y_test_array == y_pred) / y_test_array.shape[0]
    return accuracy_rate


def label_percentages(labels):
    n0 = 0
    n1 = 0
    n2 = 0
    total = labels.shape[0]
    for label in labels:
        if label == 0:
            n0 += 1
        elif label == 1:
            n1 += 1
        elif label == 2:
            n2 += 1

    return (n0, n1, n2), (n0 / total, n1 / total, n2 / total), total


def mat_prob_test(test_data, final_theta):
    res = test_data.dot(np.transpose(final_theta))
    return np.argmax(res.to_numpy(), axis=1)
