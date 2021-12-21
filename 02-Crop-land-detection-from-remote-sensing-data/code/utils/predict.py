import numpy as np
from utils import metrics


def get_prediction(x_train, y_train, x_test, model, model_name):
    model.fit(x_train, y_train)
    y_pred_train = model.predict(x_train)
    print(model_name + '  f1_score on training dataset: ')
    print(metrics.get_f1(y_train, y_pred_train))
    metrics.get_classification_report(y_train, y_pred_train)
    y_pred = model.predict(x_test)
    y_pred = y_pred.astype(np.uint8)
    print(y_pred)
    return y_pred