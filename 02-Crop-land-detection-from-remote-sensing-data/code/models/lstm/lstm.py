import torch
from utils import predict, metrics, dataset, submission
from models.cnn.cnn import *
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import torch.optim as optim
from sklearn.metrics import f1_score


# https://www.kaggle.com/andradaolteanu/pytorch-rnns-and-lstms-explained-acc-0-99

from sklearn.model_selection import train_test_split
from torch import nn
import numpy as np
import torch


def reshape_data(x):
    return x.reshape(x.shape[0], 12, 18)


def get_tensor_data(normalization=True):
    train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
    features, labels = dataset.get_train(train)
    testnolabels = dataset.get_test(raw_test)
    X_train, X_test, y_train, y_test = train_test_split(features,
                                                        labels,
                                                        test_size=0.1,
                                                        random_state=42)

    trainX = torch.tensor(reshape_data(X_train.values)).float()
    trainY = torch.tensor(y_train.values.astype(np.int64))
    valX = torch.tensor(reshape_data(X_test.values)).float()
    valY = torch.tensor(y_test.values.astype(np.int64))
    testdata = torch.tensor(reshape_data(testnolabels.values)).float()
    if normalization:
        mean_train = torch.mean(trainX, dim=0)
        std_train = torch.std(trainX, dim=0)
        trainX = (trainX - mean_train) / std_train
        valX = (valX - mean_train) / std_train
        testdata = (testdata - mean_train) / std_train
    return trainX, trainY, valX, valY, testdata


def get_accuracy(out, actual_labels, batch_size):
    softmax = nn.Softmax(dim=1)
    out = softmax(out)
    predictions = out.max(dim=1)[1]
    correct = (predictions == actual_labels).sum().item()
    accuracy = correct / batch_size
    return accuracy


def train_network_predict(model, batchSize=128, num_epochs=50, learning_rate=0.001):
    print('Get data ready...')
    # Create dataloader for training dataset - so we can train on multiple batches
    # Shuffle after every epoch
    trainX, trainY, valX, valY, testdata = get_tensor_data(normalization=True)
    n_batches = int(np.ceil(trainX.shape[0] / batchSize))

    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Training started...')
    # Train the data multiple times
    for epoch in range(num_epochs):
        # Save Train and Test Loss
        train_loss = 0
        train_acc = 0
        # Set model in training mode:
        model.train()
        for batch in range(n_batches):
            minibatchX = trainX[batchSize * batch: batchSize * (batch + 1), :]
            minibatchY = trainY[batchSize * batch: batchSize * (batch + 1)]
            # Create log probabilities
            out = model(minibatchX)
            # Clears the gradients from previous iteration
            optimizer.zero_grad()
            # Computes loss: how far is the prediction from the actual?
            loss = criterion(out, minibatchY)
            # Computes gradients for neurons
            loss.backward()
            # Updates the weights
            optimizer.step()

            # Save Loss & Accuracy after each iteration
            train_loss += loss.item()
            train_acc += get_accuracy(out, minibatchY, batchSize)

            # Print Average Train Loss & Accuracy after each epoch
        print('TRAIN | Epoch: {}/{} | Loss: {:.2f} | Accuracy: {:.2f}'.format(epoch + 1, num_epochs, train_loss / n_batches,
                                                                              train_acc / n_batches))
        print('Testing Started...')
        # Save Test Accuracy
        # Evaluation mode
        model.eval()
        # Create logit predictions
        out = model(valX)
        # Add Accuracy of this batch
        y_pred_labels_val = torch.argmax(out, dim=1)
        f1_test = metrics.get_f1(valY.detach().numpy(), y_pred_labels_val.detach().numpy())
        # Print Final Test Accuracy
        print('TEST | f1 for test : {:.5f}'.format(f1_test))
        # metrics.get_classification_report(valY.detach().numpy(), y_pred_labels.detach().numpy())
    print('Prediction Started...')
    # prediction
    with torch.no_grad():
        output_prob = model(testdata)
    y_pred_labels = torch.argmax(output_prob, dim=1)
    print(y_pred_labels)
    return y_pred_labels.detach().numpy()


def predict_with_all_train(model, batchSize=128, num_epochs=50, learning_rate=0.001):
    print('Get data ready...')
    train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
    features, labels = dataset.get_train(train)
    testnolabels = dataset.get_test(raw_test)

    trainX = torch.tensor(reshape_data(features.values)).float()
    trainY = torch.tensor(labels.values.astype(np.int64))
    testdata = torch.tensor(reshape_data(testnolabels.values)).float()
    mean_train = torch.mean(trainX, dim=0)
    std_train = torch.std(trainX, dim=0)
    trainX = (trainX - mean_train) / std_train
    testdata = (testdata - mean_train) / std_train
    n_batches = int(np.ceil(trainX.shape[0] / batchSize))

    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print('Training started...')
    # Train the data multiple times
    for epoch in range(num_epochs):
        # Save Train and Test Loss
        train_loss = 0
        train_acc = 0
        # Set model in training mode:
        model.train()
        for batch in range(n_batches):
            minibatchX = trainX[batchSize * batch: batchSize * (batch + 1), :]
            minibatchY = trainY[batchSize * batch: batchSize * (batch + 1)]
            # Create log probabilities
            out = model(minibatchX)
            # Clears the gradients from previous iteration
            optimizer.zero_grad()
            # Computes loss: how far is the prediction from the actual?
            loss = criterion(out, minibatchY)
            # Computes gradients for neurons
            loss.backward()
            # Updates the weights
            optimizer.step()

            # Save Loss & Accuracy after each iteration
            train_loss += loss.item()
            train_acc += get_accuracy(out, minibatchY, batchSize)

            # Print Average Train Loss & Accuracy after each epoch
        print('TRAIN | Epoch: {}/{} | Loss: {:.2f} | Accuracy: {:.2f}'.format(epoch + 1, num_epochs,
                                                                              train_loss / n_batches,
                                                                              train_acc / n_batches))



    # prediction
    with torch.no_grad():
        output_prob = model(testdata)
    y_pred_labels = torch.argmax(output_prob, dim=1)
    print(y_pred_labels)
    return y_pred_labels.detach().numpy()


class LSTM_CROP(nn.Module):
    def __init__(self, input_size, hidden_size, layer_size, output_size, bidirectional=True):
        super(LSTM_CROP, self).__init__()

        self.input_size, self.hidden_size, self.layer_size, self.output_size = input_size, hidden_size, layer_size, output_size
        self.bidirectional = bidirectional

        # the LSTM model
        # batch_first:  tensor of shape (L, N, H_in) when batch_first=False
        # or (N, L, H_in) when batch_first=True containing the features of the input sequence.
        # N: batch size
        # L: sequence length, 12 in our case
        # H_in: input size, 18 in our case
        self.lstm = nn.LSTM(input_size, hidden_size, layer_size, batch_first=True, bidirectional=bidirectional)

        if bidirectional:  # we'll have 2 more layers
            self.layer = nn.Linear(hidden_size * 2, output_size)
        else:
            self.layer = nn.Linear(hidden_size, output_size)

    def forward(self, data, prints=False):
        if prints: print('data shape:', data.shape)

        # Set initial states
        if self.bidirectional:
            # Hidden state:
            hidden_state = torch.zeros(self.layer_size * 2, data.size(0), self.hidden_size)
            # Cell state:
            cell_state = torch.zeros(self.layer_size * 2, data.size(0), self.hidden_size)
        else:
            # Hidden state:
            hidden_state = torch.zeros(self.layer_size, data.size(0), self.hidden_size)
            # Cell state:
            cell_state = torch.zeros(self.layer_size, data.size(0), self.hidden_size)
        if prints: print('hidden_state t0 shape:', hidden_state.shape, '\n' +
                         'cell_state t0 shape:', cell_state.shape)

        # LSTM:
        output, (last_hidden_state, last_cell_state) = self.lstm(data, (hidden_state, cell_state))
        if prints: print('LSTM: output shape:', output.shape, '\n' +
                         'LSTM: last_hidden_state shape:', last_hidden_state.shape, '\n' +
                         'LSTM: last_cell_state shape:', last_cell_state.shape)
        # Reshape
        # ht(batc，num_layers * num_directions, h, hidden_size)
        output = output[:, -1, :]
        if prints: print('output reshape:', output.shape)

        # fully connected layer
        output = self.layer(output)
        if prints: print('fully connected layer: Final output shape:', output.shape)
        return output


