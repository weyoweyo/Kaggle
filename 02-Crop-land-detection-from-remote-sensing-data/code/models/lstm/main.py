import torch
from utils import predict, metrics, dataset, submission
from models.lstm.lstm import *
warnings.filterwarnings('ignore')
# https://www.kaggle.com/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99
from datetime import datetime

print("starting lstm...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')

# params for one direct lstm 
batch_size1 = 128
input_size1 = 18       # number of features
hidden_size1 = 200     # number of hidden neurons
layer_size1 = 3        # number of layers
output_size1 = 2      # possible choices

# one direction lstm
lstm_model = LSTM_CROP(input_size1, hidden_size1, layer_size1, output_size1, bidirectional=False)

y_pred = predict_with_all_train(lstm_model)
print('label percentage for prediction: ')
n0_per, n1_per = metrics.label_percentages(y_pred)
print(n0_per)
print(n1_per)
filename = 'lstm_' + 'n0_' + str(format(n0_per, '.4f')) + '_n1_' + str(format(n1_per, '.4f'))
submission.save_data(sub, y_pred, 'predictions/', filename)
print("ending lstm...")
end_time = datetime.now()
print(end_time)


'''
print("starting bilstm...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')

# params for bi-lstm
batch_size = 128
input_size = 18       # number of features
hidden_size = 100     # number of hidden neurons
layer_size = 2        # number of layers
output_size = 2      # possible choices

# Creating the Model
# bi-lstm
lstm_model = LSTM_CROP(input_size, hidden_size, layer_size, output_size, bidirectional=True)

y_pred = predict_with_all_train(lstm_model)
print('label percentage for prediction: ')
n0_per, n1_per = metrics.label_percentages(y_pred)
print(n0_per)
print(n1_per)
filename = 'bilstm_' + 'n0_' + str(format(n0_per, '.4f')) + '_n1_' + str(format(n1_per, '.4f'))
submission.save_data(sub, y_pred, 'predictions/', filename)

print("ending bilstm...")
end_time = datetime.now()
print(end_time)
'''