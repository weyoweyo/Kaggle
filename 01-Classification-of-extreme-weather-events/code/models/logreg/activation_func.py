import numpy as np


def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))


# input product = X * theta
def softmax(product):
    if len(product.shape) > 1:
      max_each_row = np.max(product, axis=1, keepdims=True)
      exps = np.exp(product - max_each_row)
      sum_exps = np.sum(exps, axis=1, keepdims=True)
      res = exps / sum_exps

    else:
        product_max = np.max(product)
        product = product - product_max
        numerator = np.exp(product)
        denominator = 1.0 / np.sum(numerator)
        res = numerator.dot(denominator)
    return res
