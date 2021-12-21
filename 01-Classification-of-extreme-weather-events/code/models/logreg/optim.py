import numpy as np

from models.logreg import train_model
from utils import metric
from models.logreg import activation_func


def hyperparameter_tuning(lambda_list, X_train, y_onehot, X_test, y_test, eps, alpha, max_iter, nb_classes):
    n = X_train.shape[1]
    all_theta = {}
    all_losses = {}
    print("Hyperparameter tuning: Lambda")
    for each_lambda in lambda_list:
        theta0 = np.zeros((nb_classes, n))
        print(each_lambda)
        theta, loss_dict = train_model.iterative(X_train, y_onehot, theta0, each_lambda, eps, alpha, max_iter, nb_classes)
        all_theta[each_lambda] = theta
        all_losses[each_lambda] = loss_dict
        accuracy = metric.accuracy_percent(X_test, y_test, theta)
        print("accuracy for lambda = {}: {:.8f}".format(each_lambda, accuracy))
        print("-------------------------------------------------")

    return all_theta, all_losses



def cost(X, y, theta, lambda_):
    # m: number of examples
    # n: number of features with bias term

    # X: (m, n), y (one column of y_onehot): (m, 1),
    # theta: (n, 1)
    m = X.shape[0]
    n = X.shape[1]

    h = activation_func.sigmoid(np.dot(X, theta))
    # h: (m, nb_classes)

    log_h = np.log(h)  # (m, 1)
    log_one_minusH = np.log(1 - h)  # (m, 1)

    yT = np.transpose(y)  # (1, m)
    one_minus_yT = np.transpose(1 - y)  # (1, m)

    sum_ = np.dot(-yT, log_h) - np.dot(one_minus_yT, log_one_minusH)
    # (1, m) * (m, 1) - (1, m) * (m, 1)

    theta_without_bias = theta[1:n]
    reg = (lambda_ / (2 * m)) * np.sum(theta_without_bias ** 2)

    return sum_ / m + reg


def reg_cost_softmax(X, y_onehot, theta, lambda_):
  n_samples = X.shape[0]
  softmax_res = activation_func.softmax(np.dot(X, theta.T))  # (n_samples, n_classes)
  cost = - (1.0 / n_samples) * np.sum(y_onehot * np.log(softmax_res))

  theta_without_bias = theta[:, 1:theta.shape[1]]
  reg = lambda_ / n_samples  * np.sum(theta_without_bias ** 2)
  return cost + reg