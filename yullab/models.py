import torch.nn as nn
import torch.nn.functional as F
import torch


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.reshape(-1, 784)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool1 = nn.MaxPool2d(3)
        self.conv2 = nn.Conv2d(32, 128, 3)
        self.pool2 = nn.MaxPool2d(3)
        self.fc1 = nn.Linear(512, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return x


class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1 = nn.RNN(input_size=28, hidden_size=128, batch_first=True)
        self.rnn2 = nn.RNN(input_size=128, hidden_size=64, batch_first=True)
        self.fc1 = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.squeeze(x)
        outputs, hidden = self.rnn1(x)
        outputs, hidden = self.rnn2(hidden)
        x = self.fc1(outputs[-1])
        x = F.relu(x)
        return x


class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.squeeze(x)
        outputs, hidden = self.lstm(x)
        x = self.fc1(outputs[:, -1, :])
        x = F.relu(x)
        return x


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # encode
        self.fc1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.latent_dim)

        # decode
        self.fc3 = nn.Linear(self.latent_dim, self.hidden_dim)
        self.fc4 = nn.Linear(self.hidden_dim, self.input_dim)

    def encode(self, x):
        hidden = F.relu(self.fc1(x))
        mu = F.relu(self.fc2(hidden))
        sigma = F.relu(self.fc2(hidden))
        return mu, sigma

    def decode(self, z):
        hidden = F.relu(self.fc3(z))
        output = self.fc4(hidden)
        return output

    def forward(self, x):
        mu, sigma = self.encode(x)
        z = mu + sigma * torch.randn(self.latent_dim)
        reconstructed_z = self.decode(z)
        return reconstructed_z, mu, sigma
