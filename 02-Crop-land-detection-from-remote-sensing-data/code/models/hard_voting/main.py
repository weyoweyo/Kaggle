from utils import predict, metrics, dataset, submission
from datetime import datetime
import warnings
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier, VotingClassifier
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
warnings.filterwarnings('ignore')


print("starting voting classifier...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
x_train, y_train = dataset.get_train(train)
test = dataset.get_test(raw_test)

print('label percentage for training dataset: ')
print(metrics.label_percentages(y_train))


adaboost = AdaBoostClassifier(n_estimators=150)
knn = KNeighborsClassifier(n_neighbors=1)
lgbm = LGBMClassifier()
rf = RandomForestClassifier(n_estimators=350,
                             max_depth=45,
                             max_features='auto',
                             bootstrap=True,
                             min_samples_leaf=1,
                             min_samples_split=2)
xgboost = XGBClassifier(max_depth= 6,
                        n_estimators=157,
                        learning_rate= 0.23989473738149508,
                        gamma= 0.7442032316202452,
                        random_state=666)

voting = VotingClassifier(estimators=[('adaboost', adaboost),
                                      ('knn', knn),
                                      ('lgbm', lgbm),
                                      ('rf', rf),
                                      ('xgboost', xgboost)
                                      ],
                          voting='hard')


y_pred = predict.get_prediction(x_train,
                                y_train,
                                test,
                                voting,
                                "hard voting")
print('label percentage for prediction: ')
n0_per, n1_per = metrics.label_percentages(y_pred)
print(n0_per)
print(n1_per)
filename = 'n0_' + str(format(n0_per, '.4f')) + '_n1_' + str(format(n1_per, '.4f'))
submission.save_data(sub, y_pred, 'predictions/', filename)

print("end")
end_time = datetime.now()
print(end_time)