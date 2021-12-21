import numpy as np
import pandas as pd
import warnings

from utils import dataset, submission, metric
from models.logreg import onehot, train_model

warnings.filterwarnings('ignore')

print("Legistic Regression")

# Declaration of files, hyperparameters and others.
train, test, sub = dataset.get_from_files('train', 'test', 'sample_submission')

nb_classes = 3
split_perc_train = 0.7
split_perc_test = 0.15

lambda_ = 3
eps = 10 ** -12
alpha = 0.85
max_iter = 4000

all_features = pd.concat([train.iloc[:, :-1], test]).reset_index(drop=True)
removed_features = ["PS", "PRECT"]


# Initialize/compute needed data.
train_data, train_data_y, test_data \
    = dataset.feature_scaling(train, all_features, removed_features)

X_train, y_train, X_val, y_val, X_test, y_test \
    = dataset.split_train_test(train_data, train_data_y, split_perc_train, split_perc_test)

y_onehot \
    = onehot.y(pd.Series.to_numpy(y_train.copy()), nb_classes)

theta \
    = np.zeros((nb_classes, train_data.shape[1]))


# Hyperparameter tuning.
# max_iter_param_tuning = 100
# lambda_list = [0.1, 1, 3, 5]
# all_theta, all_losses = optim.hyperparameter_tuning(lambda_list, X_train, y_onehot, X_test, y_test, eps, alpha, max_iter_param_tuning, nb_classes)


# Train model.
final_theta, loss_dict_final = train_model.iterative(X_train, y_onehot, theta, lambda_, eps, alpha, max_iter, nb_classes)


# Validation on test and metrics.
pred_test = metric.mat_prob_test(test_data, final_theta)

accuracy = metric.accuracy_percent(X_train, y_train, final_theta)

labels_disparity = metric.label_percentages(pred_test)


# Print and save results
print("====================================================")
accuracy_show = "accuracy: " + str(accuracy)
labels_disparity_show = "labels_disparity: " + str(labels_disparity)
lambda_show = "lambda: " + str(lambda_)
eps_show = "eps: " + str(eps)
alpha_show = "alpha: " + str(alpha)
max_iter_show = "max_iter: " + str(max_iter)
removed_features_show = "removed_features: " + ' '.join(removed_features)

print(accuracy_show)
print(labels_disparity_show)
print(lambda_show)
print(eps_show)
print(alpha_show)
print(max_iter_show)
print(removed_features_show)

params = [accuracy_show, labels_disparity_show, lambda_show, eps_show, alpha_show, max_iter_show,removed_features_show]

submission.save_logs(params, accuracy, pred_test)
