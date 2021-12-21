from utils import dataset
from sklearn.model_selection import train_test_split
from datetime import datetime
from models.adaboost import param_tuning


print("starting ada boost hyperparameter tuning...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
all_x, all_y = dataset.get_train(train)

x_train, x_test, y_train, y_test = train_test_split(all_x,
                                                    all_y,
                                                    test_size=0.1,
                                                    random_state=42)

n_estimators = [30, 50, 70, 100, 150]
random_grid = {'n_estimators': n_estimators}

random_search_best_params, random_search_best_score = \
    param_tuning.random_search(random_grid, x_train, y_train)
print("random_search_best_params")
print(random_search_best_params)
print("random_search_best_score")
print(random_search_best_score)

'''
random_search_best_params
{'n_estimators': 150}
random_search_best_score
0.784068100358423
'''

