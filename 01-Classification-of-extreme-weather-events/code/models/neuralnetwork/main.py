import numpy as np
import pandas as pd
import warnings
from utils import dataset, submission, metric

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

print("Neural Network")

train, test, sub = dataset.get_from_files('train', 'test', 'sample_submission')


def onehot_labels(labels):
    # encode class values as integers
    encoder = LabelEncoder()
    encoder.fit(labels)
    encoded_Y = encoder.transform(labels)
    # convert integers to dummy variables (i.e. one hot encoded)
    dummy_y = np_utils.to_categorical(encoded_Y)
    return dummy_y


removed_features = ['S.No', 'time', 'PS', 'PRECT', 'T200']
all_features = pd.concat([train.iloc[:, :-1], test]).reset_index(drop=True)
X_train = train.iloc[:, 0:train.shape[1] - 1]
y_train = train.iloc[:, -1:]
clean_data = dataset.preprocessing_nn(all_features, removed_features)
clean_data_add_season = dataset.add_seasons(clean_data, all_features)
fea_eng_train, train_y, fea_eng_test = dataset.separate_train_test(train, clean_data_add_season)

y_onehot = onehot_labels(train_y)


X_train_nn, X_val_nn, y_train_nn, y_val_nn = train_test_split(fea_eng_train,
                                                              y_onehot,
                                                              test_size=0.3,
                                                              random_state=42)

# create model
model = Sequential()
model.add(Dense(14, input_dim=19, activation='relu'))
model.add(Dense(3, activation='softmax', kernel_regularizer='l2'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

training_result = model.fit(X_train_nn,
                            y_train_nn,
                            batch_size=10,
                            epochs=50,
                            validation_data=(X_val_nn, y_val_nn))

model.evaluate(X_train_nn, y_train_nn)
y_mat_prob = model.predict(fea_eng_test)
y_pred_nn = np.argmax(y_mat_prob, axis=1)

labels_perc = metric.label_percentages(y_pred_nn)
print(labels_perc)

submission.save_data(y_pred_nn, '../../data/predictions/', str(labels_perc))
