# basic
import warnings
import matplotlib.pyplot as plt
import numpy as np

# models
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# metrics
from utils.metrics import get_f1
warnings.filterwarnings('ignore')


def knn(X_train, y_train, X_test, y_test, title=None):
    np.random.seed(42)
    kVals = np.arange(1, 20, 2)
    metrics = ['euclidean', 'cosine', 'canberra', 'manhattan', 'correlation']
    plt.figure(figsize=(8, 8))

    for each_metric in metrics:
        print(each_metric)
        f1_scores = []
        for k in kVals:
            model = KNeighborsClassifier(n_neighbors=k, p=2, metric=each_metric)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            f1 = get_f1(y_test, pred)
            f1_scores.append(f1)
            print("K = " + str(k) + "; f1 score: " + str(f1))
        plt.plot(kVals, f1_scores, label=each_metric)
        plt.xlabel("K Value")
        plt.ylabel("f1")
        plt.legend(loc=3)
        plt.savefig('plots/' + title + '.png')


def run(features, labels, title):
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.2,
                                                        random_state=42)
    knn(X_train, y_train, X_test, y_test, title)

