from sklearn.model_selection import train_test_split
from torch import nn
from utils import dataset
import numpy as np
import torch
from typing import Tuple
import tqdm
# reference: devoir 3


def reshape_data(x):
    # Conv1d expects inputs of shape [batch, channels, features]
    return x.reshape(x.shape[0], 12, 18)


class Trainer:
    def __init__(self,
                 in_channel: int = 12,
                 out_channel: int = 24,
                 kernel_size: int = 1,
                 stride: int = 1,
                 padding: int = 0,
                 dense_layer: int = 256,
                 n_classes: int = 2,
                 lr: float = 0.0005,
                 batch_size: int = 128,
                 normalization: bool = True
                 ):

        train, raw_test, sub = dataset.get_from_files('train', 'test_nolabels', 'sample_submission')
        features, labels = dataset.get_train(train)
        testnolabels = dataset.get_test(raw_test)

        X_train, X_test, y_train, y_test = train_test_split(features,
                                                            labels,
                                                            test_size=0.1,
                                                            random_state=42)

        self.trainX = torch.tensor(reshape_data(X_train.values)).float()
        self.trainY = torch.tensor(y_train.values.astype(np.int64))
        self.valX = torch.tensor(reshape_data(X_test.values)).float()
        self.valY = torch.tensor(y_test.values.astype(np.int64))
        self.testdata = torch.tensor(reshape_data(testnolabels.values)).float()
        if normalization:
            self.trainX, self.valX, self.testdata = self.normalize()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dense_layer = dense_layer
        self.n_classes = n_classes

        self.network = self.create_cnn(in_channel,
                                       out_channel,
                                       kernel_size,
                                       stride,
                                       padding,
                                       dense_layer,
                                       n_classes)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)
        self.lr = lr
        self.batch_size = batch_size
        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': [],
                           'train_gradient_norm': []}

    def normalize(self):
        mean_train = torch.mean(self.trainX, dim=0)
        std_train = torch.std(self.trainX, dim=0)
        normalized_train = (self.trainX - mean_train) / std_train
        normalized_valid = (self.valX - mean_train) / std_train
        normalized_test = (self.testdata - mean_train) / std_train
        return normalized_train, normalized_valid, normalized_test

    @staticmethod
    def create_cnn(in_channel, out_channel, kernel_size, stride, padding, dense_layer, n_classes):
        layers = []
        conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channel,
                      out_channels=out_channel,
                      kernel_size=kernel_size,
                      stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Flatten()
        )
        layers.append(conv)

        in_fea1 = (18 - kernel_size + 2 * padding) / stride + 1
        in_fea = int(in_fea1 * out_channel)
        dense1 = nn.Linear(in_features=in_fea, out_features=dense_layer)
        layers.append(dense1)
        dense2 = nn.Linear(in_features=dense_layer, out_features=dense_layer)
        layers.append(dense2)
        dense3 = nn.Linear(in_features=dense_layer, out_features=n_classes)
        layers.append(dense3)
        return nn.Sequential(*layers)

    def one_hot(self, y: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(y, self.n_classes)

    def compute_loss_and_accuracy(self, X: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, float]:
        loss_criterion = nn.CrossEntropyLoss()
        output_prob = self.network(X).clone()
        y_labels = torch.argmax(y, dim=1)
        loss = loss_criterion(output_prob, y_labels)
        # precision
        y_pred_labels = torch.argmax(output_prob, dim=1)
        accuracy = torch.sum(y_pred_labels == y_labels) / y_labels.size()[0]
        return (loss, accuracy)

    @staticmethod
    def compute_gradient_norm(network: torch.nn.Module) -> float:
        parameters = [p for p in network.parameters()]
        total_norm = 0
        for p in parameters:
            p_norm = p.grad.detach().data.norm(2)
            total_norm = total_norm + p_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return total_norm

    def training_step(self, X_batch: torch.Tensor, y_batch: torch.Tensor) -> float:
        self.optimizer.zero_grad()
        loss, accu = self.compute_loss_and_accuracy(X_batch, y_batch)
        # update params
        loss.backward()
        self.optimizer.step()
        grad_norm = self.compute_gradient_norm(self.network)
        return grad_norm

    def log_metrics(self, X_train: torch.Tensor, y_train_oh: torch.Tensor,
                    X_valid: torch.Tensor, y_valid_oh: torch.Tensor) -> None:
        self.network.eval()
        with torch.no_grad():
            train_loss, train_accuracy = self.compute_loss_and_accuracy(X_train, y_train_oh)
            valid_loss, valid_accuracy = self.compute_loss_and_accuracy(X_valid, y_valid_oh)
        self.train_logs['train_accuracy'].append(train_accuracy)
        self.train_logs['validation_accuracy'].append(valid_accuracy)
        self.train_logs['train_loss'].append(float(train_loss))
        self.train_logs['validation_loss'].append(float(valid_loss))

    def train_loop(self, n_epochs: int):
        # Prepare train and validation data
        X_train = self.trainX
        y_train = self.trainY
        y_train_oh = self.one_hot(y_train)
        X_valid = self.valX
        y_valid_oh = self.one_hot(self.valY)
        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))
        # initial loss and accuracy
        self.log_metrics(X_train, y_train_oh, X_valid, y_valid_oh)

        # training loop
        for epoch in tqdm.tqdm(range(n_epochs)):
            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch: self.batch_size * (batch + 1), :]
                minibatchY = y_train_oh[self.batch_size * batch: self.batch_size * (batch + 1), :]
                gradient_norm = self.training_step(minibatchX, minibatchY)

            # Just log the last gradient norm
            self.train_logs['train_gradient_norm'].append(gradient_norm)
            # loss and accuracy for each epoch
            self.log_metrics(X_train, y_train_oh, X_valid, y_valid_oh)
        return self.train_logs

    def predict(self):
        self.network.eval()
        with torch.no_grad():
            output_prob = self.network(self.testdata).clone()
        y_pred_labels = torch.argmax(output_prob, dim=1)
        return y_pred_labels.detach().numpy()
