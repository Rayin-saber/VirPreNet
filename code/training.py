# -*- coding: utf-8 -*-
"""
Created on Thu May 30 19:50:10 2019

@author: yinr0002
"""
import os, sys
import numpy as np
import torch
import warnings
import xlrd
import random
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn as nn
from sklearn.metrics import roc_curve, auc
from torchvision.models.resnet import resnet34
from feature_engineering import cnn_training_data
from feature_engineering import data_transform
from model import train_test_split_data
from model import ResNet
from model import ResNext
from model import VGG
from model import svm_baseline
from model import rf_baseline
from model import lr_baseline
from model import knn_baseline
from model import nn_baseline
from model import reshape_to_linear
from model import setup_seed
from train_cnn import train_cnn
from validation import evaluate
from train_cnn import calculate_prob
from train_cnn import predictions_from_output
from sklearn.model_selection import StratifiedKFold

sys.path.append(os.path.abspath('/content/drive/Colab Notebooks/NTU/code5'))
warnings.filterwarnings('ignore')

os.chdir('/content/drive/Colab Notebooks/NTU/code5')

index = None


def get_index():
    '''
    Ensure all segments' samples indexs  are the same
    '''
    HA_data = xlrd.open_workbook("HA_data_with_site.xlsx")  # file path
    HA_data = HA_data.sheet_by_index(0)
    raw_feature, raw_label = data_transform(HA_data, 2)  # transform raw data
    feature, label = cnn_training_data(raw_feature, raw_label, 2)

    global index
    setup_seed(14)  # change seed

    shuffled_index = np.arange(len(feature))
    random.shuffle(shuffled_index)

    index = shuffled_index

    return index


def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))


def gradAscent(dataMatIn, classLabels):
    '''
    get weights of different segments
    '''

    dataMatrix = np.mat(dataMatIn)
    labelMat = np.mat(classLabels).transpose()
    # labelMat = mat(classLabels).T
    m, n = np.shape(dataMatrix)
    alpha = 0.001  # learning rate
    maxCycles = 2500  # num of iterations
    weights = np.ones((n, 1))
    for cycle in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose() * error
        if cycle % 50 == 1:
            print("Regression epoches: ", cycle)

    return weights


def main():
    parameters = {

        # select influenza subtype
        'segment': segment,

        ## select the way for feature generation
        # 'feature_type': feature_type,

        # 'rf', lr', 'knn', 'svm', 'cnn'
        # 'model': model,

        # Number of predictive virulent class
        'vir_class': 2,

        # Number of hidden units in the encoder
        'hidden_size': 128,

        # Droprate (applied at input)
        'dropout_p': 0.5,

        # Note, no learning rate decay implemented
        'learning_rate': 0.0005,# 0.0005 resnet resnext

        # Size of mini batch
        'batch_size': 4,

        # Number of training iterations
        'num_of_epochs': 100,
        
        # Number of cross validation
        'split_num': 4
    }

    shuffled_index = get_index()

    if parameters['segment'] == 'PB2':
        subtype = 'PB2'
        setup_seed(14)
        # read raw sequence data
        HA_data = xlrd.open_workbook("PB2_data_with_site.xlsx")  # file path
        HA_data = HA_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(HA_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 4:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + PB2:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + PB2:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + PB2:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + PB2:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + PB2:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + PB2:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + PB2:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")          
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + PB2:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'PB1':
        subtype = 'PB1'
        setup_seed(14)
        # read raw sequence data
        PB1_data = xlrd.open_workbook("PB1_data_with_site.xlsx")  # file path
        PB1_data = PB1_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(PB1_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 4:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + PB1:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + PB1:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + PB1:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + PB1:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + PB1:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + PB1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + PB1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + PB1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'PA':
        subtype = 'PA'
        setup_seed(14)
        # read raw sequence data
        PA_data = xlrd.open_workbook("PA_data_with_site.xlsx")  # file path
        PA_data = PA_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(PA_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 4:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + PA:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + PA:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + PA:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + PA:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + PA:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + PA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + PA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + PA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'PAX':
        subtype = 'PAX'
        setup_seed(14)
        # read raw sequence data
        PA_data = xlrd.open_workbook("PAX_data_with_site.xlsx")  # file path
        PA_data = PA_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(PA_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold")
            train_x = np.array(feature)[train_index].tolist()
            test_x = np.array(feature)[test_index].tolist()
            train_y = np.array(label)[train_index].tolist()
            test_y = np.array(label)[test_index].tolist()
            #print(test_y[0:15])
    
            if baseline == 'rf_baseline':
                print('rf_baseline + PAX:')
                outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'lr_baseline':
                print('lr_baseline + PAX:')
                outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'svm_baseline':
                print('svm_baseline + PAX:')
                outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'knn_baseline':
                print('knn_baseline + PAX:')
                outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'nn_baseline':
                print('nn_baseline + PAX:')
                outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'vgg':
                print('VGG + PAX:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = VGG(models.vgg16_bn(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'res':
                print('ResNet + PAX:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = ResNet(resnet34(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'resx':
                print('ResNext + PAX:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = ResNext(models.resnext50_32x4d(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'HA':
        subtype = 'HA'
        setup_seed(14)
        # read raw sequence data
        HA_data = xlrd.open_workbook("HA_data_with_site.xlsx")  # file path
        HA_data = HA_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(HA_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 3:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + HA:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + HA:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + HA:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + HA:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + HA:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + HA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + HA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + HA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'NP':
        subtype = 'NP'
        setup_seed(14)
        # read raw sequence data
        NP_data = xlrd.open_workbook("NP_data_with_site.xlsx")  # file path
        NP_data = NP_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(NP_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 1:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + NP:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + NP:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + NP:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + NP:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + NP:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + NP:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + NP:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + NP:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'NA':
        subtype = 'NA'
        setup_seed(14)
        # read raw sequence data
        NA_data = xlrd.open_workbook("NA_data_with_site.xlsx")  # file path
        NA_data = NA_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(NA_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 3:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + NA:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + NA:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + NA:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + NA:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + NA:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + NA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + NA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + NA:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'M1':
        subtype = 'M1'
        setup_seed(14)
        # read raw sequence data
        M1_data = xlrd.open_workbook("M1_data_with_site.xlsx")  # file path
        M1_data = M1_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(M1_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 4:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + M1:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + M1:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + M1:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + M1:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + M1:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + M1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + M1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + M1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'M2':
        subtype = 'M2'
        setup_seed(14)
        # read raw sequence data
        M1_data = xlrd.open_workbook("M2_data_with_site.xlsx")  # file path
        M1_data = M1_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(M1_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold")
            train_x = np.array(feature)[train_index].tolist()
            test_x = np.array(feature)[test_index].tolist()
            train_y = np.array(label)[train_index].tolist()
            test_y = np.array(label)[test_index].tolist()
            #print(test_y[0:15])
    
            if baseline == 'rf_baseline':
                print('rf_baseline + M2:')
                outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'lr_baseline':
                print('lr_baseline + M2:')
                outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'svm_baseline':
                print('svm_baseline + M2:')
                outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'knn_baseline':
                print('knn_baseline + M2:')
                outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'nn_baseline':
                print('nn_baseline + M2:')
                outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'vgg':
                print('VGG + M2:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = VGG(models.vgg16_bn(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'res':
                print('ResNet + M2:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = ResNet(resnet34(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'resx':
                print('ResNext + M2:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = ResNext(models.resnext50_32x4d(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'NS1':
        subtype = 'NS1'
        setup_seed(14)
        # read raw sequence data
        NS1_data = xlrd.open_workbook("NS1_data_with_site.xlsx")  # file path
        NS1_data = NS1_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(NS1_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        flag = 1
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold", flag)
            if flag != 1:
                flag = flag + 1
                pass
            else:
                flag = flag + 1
                train_x = np.array(feature)[train_index].tolist()
                test_x = np.array(feature)[test_index].tolist()
                train_y = np.array(label)[train_index].tolist()
                test_y = np.array(label)[test_index].tolist()
                #print(test_y[0:15])
        
                if baseline == 'rf_baseline':
                    print('rf_baseline + NS1:')
                    outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'lr_baseline':
                    print('lr_baseline + NS1:')
                    outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'svm_baseline':
                    print('svm_baseline + NS1:')
                    outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'knn_baseline':
                    print('knn_baseline + NS1:')
                    outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'nn_baseline':
                    print('nn_baseline + NS1:')
                    outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'vgg':
                    print('VGG + NS1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = VGG(models.vgg16_bn(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'res':
                    print('ResNet + NS1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNet(resnet34(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
                elif baseline == 'resx':
                    print('ResNext + NS1:')
                    train_x = np.reshape(train_x, (
                        np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                    test_x = np.reshape(test_x,
                                        (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                    #print(np.array(train_x).shape)
                    #print(np.array(test_x).shape)
        
                    train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                    train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                    test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                    test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                    net = ResNext(models.resnext50_32x4d(pretrained=True))
                    if torch.cuda.is_available():
                        print('running with GPU')
                        net.cuda()
        
                    outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                              train_y, test_x, test_y, subtype)
                    print("############################ Best_outcome... ##################################")
                    print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                        outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                        outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                    outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))

    if parameters['segment'] == 'NS2':
        subtype = 'NS2'
        setup_seed(14)
        # read raw sequence data
        NS1_data = xlrd.open_workbook("NS2_data_with_site.xlsx")  # file path
        NS1_data = NS1_data.sheet_by_index(0)
        raw_feature, raw_label = data_transform(NS1_data, parameters['vir_class'])  # transform raw data
        feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
        stratified_folder = StratifiedKFold(n_splits=parameters['split_num'], random_state=2, shuffle=False)
        outcome = []
        for train_index, test_index in stratified_folder.split(feature, label):
            print("New Fold")
            train_x = np.array(feature)[train_index].tolist()
            test_x = np.array(feature)[test_index].tolist()
            train_y = np.array(label)[train_index].tolist()
            test_y = np.array(label)[test_index].tolist()
            #print(test_y[0:15])
    
            if baseline == 'rf_baseline':
                print('rf_baseline + NS2:')
                outcome_tp = rf_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'lr_baseline':
                print('lr_baseline + NS2:')
                outcome_tp = lr_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'svm_baseline':
                print('svm_baseline + NS2:')
                outcome_tp = svm_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'knn_baseline':
                print('knn_baseline + NS2:')
                outcome_tp = knn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'nn_baseline':
                print('nn_baseline + NS2:')
                outcome_tp = nn_baseline(reshape_to_linear(np.array(train_x)), np.array(train_y), reshape_to_linear(np.array(test_x)), np.array(test_y))
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'vgg':
                print('VGG + NS2:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = VGG(models.vgg16_bn(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'res':
                print('ResNet + NS2:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = ResNet(resnet34(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
            elif baseline == 'resx':
                print('ResNext + NS2:')
                train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
                test_x = np.reshape(test_x,
                                    (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
                #print(np.array(train_x).shape)
                #print(np.array(test_x).shape)
    
                train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
                train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
                test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
                test_y = torch.tensor(test_y, dtype=torch.int64).cuda()
                net = ResNext(models.resnext50_32x4d(pretrained=True))
                if torch.cuda.is_available():
                    print('running with GPU')
                    net.cuda()
    
                outcome_tp = train_cnn(net, parameters['num_of_epochs'], parameters['learning_rate'], parameters['batch_size'], train_x,
                          train_y, test_x, test_y, subtype)
                print("############################ Best_outcome... ##################################")
                print('T_acc %.3f\tT_pre %.3f\tT_rec %.3f\tT_fscore %.3f' % (
                    outcome_tp[0], outcome_tp[1], outcome_tp[2], outcome_tp[3]))
                print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f' % (
                    outcome_tp[4], outcome_tp[5], outcome_tp[6], outcome_tp[7]))
                outcome.append(outcome_tp)
        outcome_final = np.mean(np.array(outcome), axis=0)
        outcome_std = np.std(np.array(outcome), axis=0)
        print("########################### Average_outcome... ################################")
        print('T_acc %.3f(%.3f)\tT_pre %.3f(%.3f)\tT_rec %.3f(%.3f)\tT_fscore %.3f(%.3f)' % (
                    outcome_final[0], outcome_std[0], outcome_final[1], outcome_std[1], outcome_final[2], outcome_std[2], outcome_final[3], outcome_std[3]))
        print('V_acc %.3f(%.3f)\tV_pre %.3f(%.3f)\tV_rec %.3f(%.3f)\tV_fscore %.3f(%.3f)' % (
                    outcome_final[4], outcome_std[4], outcome_final[5], outcome_std[5], outcome_final[6], outcome_std[6], outcome_final[7], outcome_std[7]))


if __name__ == '__main__':

    print("plot...")

    segment_type = ['PB2', 'PB1', 'PA', 'PAX', 'HA', 'NP', 'NA', 'M1', 'M2', 'NS1', 'NS2']
    baselines = ['resx']
    # baselines = ['rf_baseline', 'lr_baseline', 'knn_baseline', 'nn_baseline']
    for baseline in baselines:
        for i in [7, 9]:# 0, 1, 2, 4, 5, 6, 7, 9
            segment = segment_type[i]
            main()


    ############### Ensemble model based on ResNeXt-50 ######################
    class ResNext(nn.Module):
        def __init__(self, model):
            super(ResNext, self).__init__()
            self.conv_layer = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.resnext_layer = nn.Sequential(*list(model.children())[1:-1])
            self.Linear_layer1 = nn.Linear(2048, 256)
            self.Linear_layer2 = nn.Linear(256, 2)
            self.dropout = nn.Dropout(p=0.3)

        def forward(self, x):
            x = self.conv_layer(x)
            x = self.resnext_layer(x)
            x = x.view(x.size(0), -1)
            x = self.Linear_layer1(x)
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
            self.dropout = nn.Dropout(p=0.4)
    
        def forward(self, x):
            x = self.conv_layer(x)
            x = self.resnet_layer(x)
            x = x.view(x.size(0), -1)
            x = self.Linear_layer1(x)
            out = self.Linear_layer2(x)
            out = self.dropout(out)
            return out

    model1 = ResNext(models.resnext50_32x4d())
    #model1 = ResNet(resnet34())

    shuffled_index = index

    for segment in segment_type:
        if segment == 'PB2':
            subtype = 'PB2'
            # setup_seed(7)
            # read raw sequence data
            PB2_data = xlrd.open_workbook("PB2_data_with_site.xlsx")  # file path
            PB2_data = PB2_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(PB2_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('PB2_params.pkl'))
            print(" Successfully load PB2 model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores1 = model1(test_x)
                prediction1 = predictions_from_output(test_scores1)
                prediction1 = prediction1.view_as(test_y)
                prediction1_t = torch.from_numpy(prediction1.cpu().numpy().reshape(-1, 1))
        if segment == 'PB1':
            subtype = 'PB1'
            # setup_seed(7)
            # read raw sequence data
            PB1_data = xlrd.open_workbook("PB1_data_with_site.xlsx")  # file path
            PB1_data = PB1_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(PB1_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('PB1_params.pkl'))
            print(" Successfully load PB1 model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()

            with torch.no_grad():
                model1.eval()
                test_scores2 = model1(test_x)
                prediction2 = predictions_from_output(test_scores2)
                prediction2 = prediction2.view_as(test_y)
                prediction2_t = torch.from_numpy(prediction2.cpu().numpy().reshape(-1, 1))
        if segment == 'PA':
            subtype = 'PA'
            # setup_seed(7)
            # read raw sequence data
            PA_data = xlrd.open_workbook("PA_data_with_site.xlsx")  # file path
            PA_data = PA_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(PA_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('PA_params.pkl'))
            print(" Successfully load PA model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores3 = model1(test_x)
                prediction3 = predictions_from_output(test_scores3)
                prediction3 = prediction3.view_as(test_y)
                prediction3_t = torch.from_numpy(prediction3.cpu().numpy().reshape(-1, 1))

        '''
        if segment == 'PAX':
            subtype = 'PAX'
            #setup_seed(7)
            # read raw sequence data
            PAX_data = xlrd.open_workbook("SEG3p2.xlsx")  # file path
            PAX_data = PAX_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(PAX_data, parameters['vir_class'])  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, parameters['vir_class'])  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                    np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            #print(np.array(train_x).shape)
            #print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('PAX_params.pkl'))
            print(" Successfully load PAX model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores4 = model1(test_x)
                prediction4 = predictions_from_output(test_scores4)
                prediction4 = prediction4.view_as(test_y)
                prediction4_t = torch.from_numpy(prediction4.cpu().numpy().reshape(-1,1))
        '''

        if segment == 'HA':
            # subtype = 'HA'
            setup_seed(7)
            # read raw sequence data
            HA_data = xlrd.open_workbook("HA_data_with_site.xlsx")  # file path
            HA_data = HA_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(HA_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('HA_params.pkl'))
            print(" Successfully load HA model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores5 = model1(test_x)
                prediction5 = predictions_from_output(test_scores5)
                prediction5 = prediction5.view_as(test_y)
                prediction5_t = torch.from_numpy(prediction5.cpu().numpy().reshape(-1, 1))
        if segment == 'NP':
            subtype = 'NP'
            # setup_seed(7)
            # read raw sequence data
            NP_data = xlrd.open_workbook("NP_data_with_site.xlsx")  # file path
            NP_data = NP_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(NP_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('NP_params.pkl'))
            print(" Successfully load NP model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores6 = model1(test_x)
                prediction6 = predictions_from_output(test_scores6)
                prediction6 = prediction6.view_as(test_y)
                prediction6_t = torch.from_numpy(prediction6.cpu().numpy().reshape(-1, 1))
        if segment == 'NA':
            subtype = 'NA'
            # setup_seed(7)
            # read raw sequence data
            NA_data = xlrd.open_workbook("NA_data_with_site.xlsx")  # file path
            NA_data = NA_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(NA_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('NA_params.pkl'))
            print(" Successfully load NA model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores7 = model1(test_x)
                prediction7 = predictions_from_output(test_scores7)
                prediction7 = prediction7.view_as(test_y)
                prediction7_t = torch.from_numpy(prediction7.cpu().numpy().reshape(-1, 1))
        if segment == 'M1':
            subtype = 'M1'
            # setup_seed(7)
            # read raw sequence data
            M1_data = xlrd.open_workbook("M1_data_with_site.xlsx")  # file path
            M1_data = M1_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(M1_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('M1_params.pkl'))
            print(" Successfully load M1 model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores8 = model1(test_x)
                prediction8 = predictions_from_output(test_scores8)
                prediction8 = prediction8.view_as(test_y)
                prediction8_t = torch.from_numpy(prediction8.cpu().numpy().reshape(-1, 1))
        if segment == 'NS1':
            subtype = 'NS1'
            # setup_seed(7)
            # read raw sequence data
            NS1_data = xlrd.open_workbook("NS1_data_with_site.xlsx")  # file path
            NS1_data = NS1_data.sheet_by_index(0)
            raw_feature, raw_label = data_transform(NS1_data, 2)  # transform raw data
            feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
            train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)

            train_x = np.reshape(train_x, (
                np.array(train_x).shape[0], 1, np.array(train_x).shape[1], np.array(train_x).shape[2]))
            test_x = np.reshape(test_x,
                                (np.array(test_x).shape[0], 1, np.array(test_x).shape[1], np.array(test_x).shape[2]))
            # print(np.array(train_x).shape)
            # print(np.array(test_x).shape)

            train_x = torch.tensor(train_x, dtype=torch.float32).cuda()
            train_y = torch.tensor(train_y, dtype=torch.int64).cuda()
            test_x = torch.tensor(test_x, dtype=torch.float32).cuda()
            test_y = torch.tensor(test_y, dtype=torch.int64).cuda()

            model1.load_state_dict(torch.load('NS1_params.pkl'))
            print(" Successfully load NS1 model.")

            if torch.cuda.is_available():
                print(' running with GPU')
                model1.cuda()
            with torch.no_grad():
                model1.eval()
                test_scores9 = model1(test_x)
                prediction9 = predictions_from_output(test_scores9)
                prediction9 = prediction9.view_as(test_y)
                prediction9_t = torch.from_numpy(prediction9.cpu().numpy().reshape(-1, 1))

    ###### loading dataset ######
    NS1_data = xlrd.open_workbook("NS1_data_with_site.xlsx")  # file path
    NS1_data = NS1_data.sheet_by_index(0)
    raw_feature, raw_label = data_transform(NS1_data, 2)  # transform raw data
    feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
    train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)
    # print(test_y[:15])

    input_prediction = torch.cat((prediction1_t, prediction2_t, prediction3_t, prediction5_t, prediction6_t,
                                  prediction7_t, prediction8_t, prediction9_t), 1).numpy()
    weights_res = gradAscent(input_prediction, test_y)
    print(weights_res)

    weight_prediction = []
    test_scores = []
    weight_right = 0

    for i in range(len(input_prediction)):
        prob = sigmoid(sum(input_prediction[i] * weights_res))
        test_scores.append([np.array(prob)[0][0], 1 - np.array(prob)[0][0]])
        if prob > 0.5:
            weight_prediction.append(1)
        else:
            weight_prediction.append(0)

    for i in range(len(weight_prediction)):
        if weight_prediction[i] == test_y[i]:
            weight_right += 1
        else:
            pass

    precision, recall, fscore, mcc, val_acc = evaluate(test_y, weight_prediction)

    print('After weight regression, the accuracy is:')
    print('V_acc %.3f\tV_pre %.3f\tV_rec %.3f\tV_fscore %.3f\tV_mcc %.3f' % (
        val_acc, precision, recall, fscore, mcc))
    # plot VirPreNet's ROC Curve
    test_scores = torch.from_numpy(np.array(test_scores))
    pred_prob = calculate_prob(test_scores)
    y_test = torch.tensor(test_y, dtype=torch.int64).cuda()
    fpr_cnn, tpr_cnn, _ = roc_curve(y_test.cpu(), pred_prob.cpu())
    print("Ensemble's auc:", auc(fpr_cnn, tpr_cnn))
    plt.figure(1)
    # plt.xlim(0, 0.8)
    plt.ylim(0, 1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr_cnn, tpr_cnn, label = "VirPreNet(" + str(auc(fpr_cnn, tpr_cnn)).split('.')[0] + '.' + str(auc(fpr_cnn, tpr_cnn)).split('.')[1][:3] + ")")
    plt.legend(loc='best')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig("ROC.eps", format="eps", dpi=300)

'''    
    #########################################################################hard vote#########################################################################

    # hard voting 8
    h8_prediction = prediction1 + prediction2 + prediction3 + prediction5 + prediction6  + prediction7 + prediction8 + prediction9
    for i in range(len(h8_prediction)):
        if h8_prediction[i] < 4:
            h8_prediction[i] = 0
        elif h8_prediction[i] == 4:
            h8_prediction[i] == random.randint(0,1)
        else:
            h8_prediction[i] = 1


    #hard voting 5
    h5_prediction = prediction1 + prediction2 + prediction5 + prediction6 + prediction9 # top 5
    for i in range(len(h5_prediction)):
        if h5_prediction[i] < 3:
            h5_prediction[i] = 0
        else:
            h5_prediction[i] = 1


    #hard voting 3 
    h3_prediction = prediction1 + prediction5 + prediction6 # top 3
    for i in range(len(h3_prediction)):
        if h3_prediction[i] < 2:
            h3_prediction[i] = 0
        else:
            h3_prediction[i] = 1

    # hard voting conform
    h_prediction = h8_prediction + h5_prediction + h3_prediction
    for i in range(len(h_prediction)):
        if h_prediction[i] < 2:
            h_prediction[i] = 0
        else:
            h_prediction[i] = 1

    ###########################################################################soft vote#######################################################################

    # soft voting 8
    s8_prediction = []
    test_score8 = (test_scores1 + test_scores2 + test_scores3 +test_scores5 + test_scores6 + test_scores7 + test_scores8 +test_scores9)/8
    for i in range(len(test_score8)):
        if test_score8[i][0] > test_score8[i][1]:
            s8_prediction.append(0)
        elif test_score8[i][0] == test_score8[i][1]:
            s8_prediction.append(random.randint(0,1))
        else:
            s8_prediction.append(1)


    # soft voting 5
    s5_prediction = []
    test_score5 = (test_scores1 + test_scores2 + test_scores5 + test_scores6 + test_scores9)/5 # top 5
    for i in range(len(test_score5)):
        if test_score5[i][0] > test_score5[i][1]:
            s5_prediction.append(0)
        elif test_score5[i][0] == test_score5[i][1]:
            s5_prediction.append(random.randint(0,1))
        else:
            s5_prediction.append(1)


    # soft voting 3
    s3_prediction = []
    test_score3 = (test_scores1 + test_scores5 + test_scores6)/3 # top 3
    for i in range(len(test_score3)):
        if test_score3[i][0] > test_score3[i][1]:
            s3_prediction.append(0)
        elif test_score3[i][0] == test_score3[i][1]:
            s3_prediction.append(random.randint(0,1))
        else:
            s3_prediction.append(1)


    # soft voting conform
    s_prediction = np.array(s8_prediction) + np.array(s5_prediction) + h3_prediction.cpu().numpy()
    for i in range(len(s_prediction)):
        if s_prediction[i] < 2:
            s_prediction[i] = 0
        else:
            s_prediction[i] = 1
    print(len(s_prediction))


    ##########################################################################accuracy#######################################################################        


    NS1_data = xlrd.open_workbook("SEG8.xlsx")  # file path
    NS1_data = NS1_data.sheet_by_index(0)
    raw_feature, raw_label = data_transform(NS1_data, 2)  # transform raw data
    feature, label = cnn_training_data(raw_feature, raw_label, 2)  # feature engineering
    train_x, test_x, train_y, test_y = train_test_split_data(feature, label, 0.2, shuffled_index)
    print(test_y[:15])
    print(len(test_y))

    h_right = 0
    s_right = 0

    for i in range(len(h_prediction)):
        if h_prediction[i] == test_y[i]:
            h_right += 1
        else:
            pass

    for i in range(len(s_prediction)):
        if s_prediction[i] == test_y[i]:
            s_right += 1
        else:
            pass


    print("After hard voting, the accuracy is :", float(h_right/len(h_prediction)))
    print("After soft voting, the accuracy is :", float(s_right/len(s_prediction)))
'''