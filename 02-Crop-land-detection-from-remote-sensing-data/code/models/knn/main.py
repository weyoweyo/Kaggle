from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier
from utils import dataset, metrics, predict, submission
import warnings
warnings.filterwarnings('ignore')

print("starting knn...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
x_train, y_train = dataset.get_train(train)
test = dataset.get_test(raw_test)

print('label percentage for training dataset: ')
print(metrics.label_percentages(y_train))

knn = KNeighborsClassifier(n_neighbors=9, metric='canberra')
y_pred = predict.get_prediction(x_train,
                                y_train,
                                test,
                                knn,
                                "knn1 original data")
print('label percentage for prediction: ')
n0_per, n1_per = metrics.label_percentages(y_pred)
print(n0_per)
print(n1_per)
filename = 'n0_' + str(format(n0_per, '.4f')) + '_n1_' + str(format(n1_per, '.4f'))
submission.save_data(sub, y_pred, 'predictions/', filename)

print("end")
end_time = datetime.now()
print(end_time)


