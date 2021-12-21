import torch
from utils import predict, metrics, dataset, submission
from models.cnn.cnn import *
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt


print("starting cnn...")
start_time = datetime.now()
print(start_time)

train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
#features, labels = dataset.get_train(train)
#print(reshape_data(features.values)[0][0])


trainer = Trainer()
logs = trainer.train_loop(100)
val_accu = logs['validation_accuracy']
train_accu = logs['train_accuracy']
val_loss = logs['validation_loss']
train_loss = logs['train_loss']

len1 = len(val_accu)
len2 = len(train_accu)
len3 = len(val_loss)
len4 = len(train_loss)
plt.figure(figsize=(6, 4))
plt.plot([i for i in range(len1)], [val_accu[i] for i in range(len1)], label='val accuracy')
plt.plot([i for i in range(len2)], [train_accu[i] for i in range(len2)], label='train accuracy')
plt.legend(loc=3)
plt.savefig('plots/accuracy.png')

plt.figure(figsize=(6, 4))
plt.plot([i for i in range(len3)], [val_loss[i] for i in range(len3)], label='val loss')
plt.plot([i for i in range(len4)], [train_loss[i] for i in range(len4)], label='train loss')
plt.legend(loc=1)
plt.savefig('plots/loss.png')

y_pred = trainer.predict()
print('label percentage for prediction: ')
n0_per, n1_per = metrics.label_percentages(y_pred)
print(n0_per)
print(n1_per)
filename = 'n0_' + str(format(n0_per, '.4f')) + '_n1_' + str(format(n1_per, '.4f'))
submission.save_data(sub, y_pred, 'predictions/', filename)

print("end")
end_time = datetime.now()
print(end_time)


