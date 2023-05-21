import torch
import torch.nn.functional as F
import torch.nn as nn

k = 256
n_hidden = 300
n_instruments = 1
dropout_p = 0.1

class Discriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net_learner_supervised = torch.nn.Sequential(
            torch.nn.Linear(784, k),
            torch.nn.BatchNorm1d(k),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Dropout(p=dropout_p),
            torch.nn.Linear(k, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(256, 10),
            torch.nn.Softmax(dim=1),
            )# input shape (batch, 784), output shape (batch, 1)
    def predict(self,x):# 输出为真的概率
        # print("x:",x.shape)
        ret = 1.0 - 1.0/(1.0 + torch.sum(torch.exp(x), dim=-1))
        # print("ret:",ret)
        return ret
    def forward(self, x):
        x = x.reshape(x.shape[0], 784)
        temp = self.net_learner_supervised(x) #shape is (batch, 10)
        confident = self.predict(temp)
        temp = torch.cat((temp, confident.unsqueeze(1)), dim=1)# shape is (batch, 11)(100, 11)
        # print("temp:",temp.shape)
        # print("temp:",temp)
        return temp

net_adversary = torch.nn.Sequential(
            torch.nn.Linear(4, k),
            torch.nn.BatchNorm1d(k),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(k, 256), #200
            torch.nn.BatchNorm1d(256),
            torch.nn.LeakyReLU(),
            nn.Dropout(p=0.1),                        
            torch.nn.Linear(256, 784),
            torch.nn.Tanh()
            )# input shape (batch, 1, 28, 28), output shape (batch, 1, 28, 28)


class Net_X(nn.Module):
    def __init__(self):
        super(Net_X, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return x.squeeze()

import torchvision.models as models
class ResNet50(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet50, self).__init__()
        self.resnet = models.resnet50(pretrained=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 10)

    def predict(self, x):# 输出为真的概率
        # print("x:",x.shape)
        ret = 1.0 - 1.0/(1.0 + torch.sum(torch.exp(x), dim=-1))
        # print("ret:",ret)
        return ret
    def forward(self, x):
        x = x.reshape(x.shape[0], 3, 32, 32)
        # print(x.shape)
        temp = self.resnet(x)
        # print(temp.shape)
        confident = self.predict(temp)
        # print(confident.shape)
        temp = torch.cat((temp, confident.unsqueeze(1)), dim=1)
        # print(temp.shape)
        return temp

 