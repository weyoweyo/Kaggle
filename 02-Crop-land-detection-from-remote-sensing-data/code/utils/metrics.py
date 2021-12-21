from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score


def get_f1(labels, y_pred):
    result = f1_score(labels, y_pred)
    return result


def get_classification_report(labels, y_pred):
    print(classification_report(labels, y_pred))


def f1_cv(model, features, labels, kf):
    f1 = cross_val_score(model, features, labels, scoring='f1', cv=kf)
    return f1


def label_percentages(labels):
    n0 = 0
    n1 = 0
    total = labels.shape[0]
    for label in labels:
        if label == 0:
            n0 += 1
        elif label == 1:
            n1 += 1
    return n0 / total, n1 / total