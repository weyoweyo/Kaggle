from xgboost import XGBClassifier
from utils import predict, metrics
from utils import dataset
from sklearn.model_selection import train_test_split
from datetime import datetime


print("starting lgbm training...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
all_x, all_y = dataset.get_train(train)

x_train, x_test, y_train, y_test = train_test_split(all_x,
                                                    all_y,
                                                    test_size=0.1,
                                                    random_state=42)

'''
'''


xgboost_best = XGBClassifier()

y_test_pred = predict.get_prediction(x_train, y_train, x_test, xgboost_best, 'xgboost')
metrics.get_f1(y_test, y_test_pred)
metrics.get_classification_report(y_test, y_test_pred)