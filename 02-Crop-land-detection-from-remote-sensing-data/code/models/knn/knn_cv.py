from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from utils import metrics, dataset
from datetime import datetime


print("starting knn...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
x_train, y_train = dataset.get_train(train)

kfold = KFold(n_splits=50, random_state=42, shuffle=True)

models = [KNeighborsClassifier(n_neighbors=9, metric='canberra')]

scores = {}

names = ['knn 9']

for name, model in zip(names, models):
    score = metrics.f1_cv(model, x_train, y_train, kfold)
    print("{}: {:.6f}, {:.4f}".format(name, score.mean(), score.std()))

print("end")
end_time = datetime.now()
print(end_time)