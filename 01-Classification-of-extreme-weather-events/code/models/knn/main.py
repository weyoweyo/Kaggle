import pandas as pd
import warnings
from utils import dataset, metric, submission
from sklearn.neighbors import KNeighborsClassifier
warnings.filterwarnings('ignore')

print("knn...")

train, test, sub = dataset.get_from_files('train', 'test', 'sample_submission')
removed_features = ['PS', 'PRECT', 'T200']
all_features = pd.concat([train.iloc[:, :-1], test]).reset_index(drop=True)
clean_data = dataset.preprocessing_without_scaling(all_features, removed_features)
clean_data_add_season = dataset.add_seasons(clean_data, all_features)
fea_eng_train, train_y, fea_eng_test = dataset.separate_train_test(train, clean_data_add_season)

clr = KNeighborsClassifier(n_neighbors=1)
clr.fit(fea_eng_train, train_y)
print("Training done...")

labels_knn = clr.predict(fea_eng_test)
print("Labels...")

labels_perc = metric.label_percentages(labels_knn)
print(labels_perc)

submission.save_data(labels_knn, '../../data/predictions/', str(labels_perc))


