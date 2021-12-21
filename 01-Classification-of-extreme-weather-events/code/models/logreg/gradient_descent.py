import numpy as np

from models.logreg import activation_func, optim


def gradient(X, y, theta, lambda_):
    m = X.shape[0]
    n = X.shape[1]
    #print(np.dot(X, theta).shape)
    h = activation_func.sigmoid(np.dot(X, theta))
    #print(h.shape)

    sum_ = np.transpose(X).dot(h - y)  # (n, m) * (m, 1)
    gradient = sum_ / m

    theta_without_bias = theta[1:n]
    reg = lambda_ / m * theta_without_bias

    gradient[1:n] = gradient[1:n] + reg
    return gradient


def gradient_descent_old(X, y, theta, lambda_, eps, alpha, max_iter):  # alpha is learning rate
    losses = []
    i = 0
    print("Iteration: Cost")

    while (i < max_iter):
        i += 1
        grad = gradient(X, y, theta, lambda_)
        theta -= alpha * grad
        loss = optim.cost(X, y, theta, lambda_)
        if (i % 1000 == 0):
            print("{}: {:.8f}".format(i, loss))

        len_losses = len(losses)
        if (len_losses == 0):
            diff = np.abs(loss)
        else:
            diff = np.abs(losses[len_losses - 1] - loss)

        losses.append(loss)
        if (diff < eps):
            return theta, losses

    return theta, losses


def gradient_descent(X, y_onehot, theta, lambda_, eps, alpha, max_iter):
    losses = []
    i = 0
    print("Iteration: Cost")

    while (i < max_iter):
        i += 1
        grad = reg_gradient_softmax(X, y_onehot, theta, lambda_)
        theta -= alpha * grad

        loss = optim.reg_cost_softmax(X, y_onehot, theta, lambda_)
        if (i % 500 == 0):
            print("{}: {:.8f}".format(i, loss))

        len_losses = len(losses)
        if (len_losses == 0):
            print("{}: {:.8f}".format(i, loss))
            diff = np.abs(loss)
        else:
            diff = np.abs(losses[len_losses - 1] - loss)

        losses.append(loss)
        if (diff < eps):
            return theta, losses

    return theta, losses


def reg_gradient_softmax(X, y_onehot, theta, lambda_):
  n_samples = X.shape[0]
  softmax_res = activation_func.softmax(np.dot(X, theta.T))

  gradient = (-1.0 / n_samples) * np.dot((y_onehot - softmax_res).T, X)
  # (n_classes, n_features)

  theta_without_bias = theta[:, 1:theta.shape[1]]
  # theta: (n_classes, n_features)
  # n_feautres = X features + 1(bias term)
  # theta_without_bias: (n_classes, n_features - 1)
  reg = -lambda_ / n_samples * theta_without_bias

  gradient[:, 1:gradient.shape[1]] = gradient[:, 1:gradient.shape[1]] + reg

  return gradient