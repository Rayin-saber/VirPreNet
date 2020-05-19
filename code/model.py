# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 14:33:15 2019

@author: YINR0002
"""

import os
import numpy as np
import random
import math
import torch
import torch.nn as nn
from sklearn import neighbors
from sklearn import ensemble
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from validation import evaluate

os.chdir('/content/drive/Colab Notebooks/NTU/code5')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def reshape_to_linear(x):
    output = np.reshape(x, (x.shape[0], -1))

    return output


# split data into training and testing
def train_test_split_data(feature, label, split_ratio, shuffled_index):
    setup_seed(20)
    train_x, test_x, train_y, test_y = [], [], [], []
    feature_new, label_new = [], []
    num_of_training = int(math.floor(len(feature) * (1 - split_ratio)))

    #shuffled_index = np.arange(len(feature))
    #random.shuffle(shuffled_index)
    for i in range(0, len(feature)):
        feature_new.append(feature[shuffled_index[i]])
        label_new.append(label[shuffled_index[i]])

    train_x = feature_new[:num_of_training]
    train_y = label_new[:num_of_training]
    test_x = feature_new[num_of_training:]
    test_y = label_new[num_of_training:]

    train_x = np.array(train_x)
    train_y = np.array(train_y)
    test_x = np.array(test_x)
    test_y = np.array(test_y)

    return train_x, test_x, train_y, test_y


def lr_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = linear_model.LogisticRegression().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    outcome = [train_acc, train_pre, train_rec, train_fscore, val_acc, precision, recall, fscore]
    return outcome
    
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))


def knn_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = neighbors.KNeighborsClassifier().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    outcome = [train_acc, train_pre, train_rec, train_fscore, val_acc, precision, recall, fscore]
    return outcome
    
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))


def svm_baseline(X, Y, X_test, Y_test, method=None):
    setup_seed(20)
    clf = SVC(gamma='auto', class_weight='balanced').fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    outcome = [train_acc, train_pre, train_rec, train_fscore, val_acc, precision, recall, fscore]
    return outcome
    
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))


def rf_baseline(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = ensemble.RandomForestClassifier().fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    outcome = [train_acc, train_pre, train_rec, train_fscore, val_acc, precision, recall, fscore]
    return outcome
    
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))


def nn_baseline(X, Y, X_test, Y_test):
    setup_seed(20)
    clf = MLPClassifier(random_state=100).fit(X, Y)
    train_acc = accuracy_score(Y, clf.predict(X))
    train_pre = precision_score(Y, clf.predict(X))
    train_rec = recall_score(Y, clf.predict(X))
    train_fscore = f1_score(Y, clf.predict(X))
    train_mcc = matthews_corrcoef(Y, clf.predict(X))

    Y_pred = clf.predict(X_test)
    precision, recall, fscore, mcc, val_acc = evaluate(Y_test, Y_pred)
    outcome = [train_acc, train_pre, train_rec, train_fscore, val_acc, precision, recall, fscore]
    return outcome
    
    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f\tT_mcc %.3f'
          % (train_acc, train_pre, train_rec, train_fscore, train_mcc))
    print('V_acc  %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f'
          % (val_acc, precision, recall, fscore, mcc))


class CNN_HA(nn.Module):
    def __init__(self):
        super(CNN_HA, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=2)
        self.mp = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(32 * 147 * 26, 120)  # overlapped feature dimension
        # self.fc1 = nn.Linear(32 * 50 * 26,120)  #non-overlapped feature dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 2/3 classes
        # self.dropout = nn.Dropout(p=0.3)
        self.logsoftmax = nn.LogSoftmax()

    def forward(self, x):
        in_size = x.size(0)
        out = self.relu(self.mp(self.conv1(x)))
        out = self.relu(self.mp(self.conv2(out)))
        out = out.view(in_size, -1)
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # out = self.dropout(out)
        return self.logsoftmax(out)

class VGG(nn.Module):
    def __init__(self, model):
        super(VGG, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        #self.Linear_layer1 = nn.Linear(377600, 256)  # need edition during different segments' training !!!
        self.Linear_layer2 = nn.Linear(3136, 2)
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out

class ResNet(nn.Module):
    def __init__(self, model):
        super(ResNet, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(512, 256)
        self.Linear_layer2 = nn.Linear(256, 2)
        # self.Linear_layer3 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.6)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnet_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out


class ResNext(nn.Module):
    def __init__(self, model):
        super(ResNext, self).__init__()
        self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
        self.Linear_layer1 = nn.Linear(2048, 256)
        self.Linear_layer2 = nn.Linear(256, 2)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.conv_layer(x)
        x = self.resnext_layer(x)
        x = x.view(x.size(0), -1)
        x = self.Linear_layer1(x)
        out = self.Linear_layer2(x)
        out = self.dropout(out)
        return out