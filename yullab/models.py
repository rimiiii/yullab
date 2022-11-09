import torch.nn as nn
import torch
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = nn.MaxPool2d(x)
        x = F.relu(self.conv2(x))
        x = x.reshape()
        x = F.relu(self.fc1(x))
        return x


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=28, hidden_size=128)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=64)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        outputs, hidden = self.rnn1(x)
        outputs, hidden = self.rnn2(hidden)
        x = self.fc1(outputs[-1])
        x = F.log_softmax(x, dim=1)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        outputs, hidden = self.lstm(x)
        print(outputs.shape)
        x = self.fc1(outputs[-1])
        #x = F.log_softmax(x, dim=1)
        return x