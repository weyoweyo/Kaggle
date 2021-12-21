import numpy as np
from models.logreg import gradient_descent


def iterative_old(X_train, y_train, theta0, lambda_, eps, alpha, max_iter, nb_classes):
    n = X_train.shape[1]  # number of features including bias term
    theta = np.zeros((n, 3))
    loss_dict = {}
    for i in range(nb_classes):
        print("Cost for {}th column of theta".format(i))
        losses = []
        theta[:, i], losses = gradient_descent.gradient_descent(X_train,
                                               y_train[:, i],
                                               theta0[:, i],
                                               lambda_,
                                               eps,
                                               alpha,
                                               max_iter)
        loss_dict[i] = losses
    return theta, loss_dict


def iterative(X_train, y_train, theta0, lambda_, eps, alpha, max_iter, nb_classes):
    n_features = X_train.shape[1]  # number of features including bias term
    theta, losses = gradient_descent.gradient_descent(X_train, y_train, theta0, lambda_, eps, alpha, max_iter)
    return theta, losses