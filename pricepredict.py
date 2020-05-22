# -*- coding: utf-8 -*-
# @auther zhouhq
# create time 2020-05-15 17:47:38
import sys
import os
from core import dbconnect
from flask import Blueprint
from flask import Flask, request, jsonify, send_file, make_response, Response,redirect,session,g
from functools import wraps
import uuid

import json
from math import sqrt
from numpy import concatenate
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
from pandas import read_csv
from pandas import DataFrame
import numpy as np
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import load_model
from tensorflow import keras
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
from torch.autograd import Variable
import logging
import pickle
from sklearn.preprocessing import MinMaxScaler


#create route view
pricepredict_route = Blueprint('pricepredict', __name__) 


@pricepredict_route.route('/')
def home():
    return 'fields home'

@pricepredict_route.route('/pricepre',methods = ['GET','POST'])
def pricepre():
    result = {"success": 'False'}
    class LSTM(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=6, output_size=1):
            super().__init__()
            self.hidden_layer_size = hidden_layer_size

            self.lstm = nn.LSTM(input_size, hidden_layer_size)

            self.linear = nn.Linear(hidden_layer_size, output_size)

            self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size),
                                torch.zeros(1,1,self.hidden_layer_size)) # (num_layers * num_directions, batch_size, hidden_size)

        def forward(self, input_seq):
            lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
            predictions = self.linear(lstm_out.view(len(input_seq), -1))
            return predictions[-1]

    #model = LSTM()
    model = LSTM(1,6)
    model.load_state_dict(torch.load('model_parameters.pkl'))  # 提取net1的状态参数，将状态参数给net3

    #单维最大最小归一化和反归一化函数
    data_csv = pd.read_csv('price.csv', usecols=[1])  #导入第一列，即价格那一列的数据
    data_csv = data_csv.dropna() #去掉na数据
    dataset = data_csv.values #dataframe 转为 ndarray
    dataset = dataset.astype('float32')

    # 数据集分为训练集和测试集
    test_data_size = 8
    train_data = dataset[:-test_data_size]
    test_data = dataset[-test_data_size:]
    def Normalize(list):
        list = np.array(list)
        low, high = np.percentile(list, [0, 100])
        delta = high - low
        if delta != 0:
            for i in range(0, len(list)):
                list[i] = (list[i]-low)/delta
        return  list,low,high

    def FNoramlize(list,low,high):
        delta = high - low
        if delta != 0:
            for i in range(0, len(list)):
                list[i] = list[i]*delta + low
        return list
    list,low,high = Normalize(train_data)
    ##3.模型预测
    fut_pred = 8
    train_window = 5    

    data = request.get_data()
    print('data=%s'%data)
    json_data = json.loads(data.decode("utf-8"))
    print(json_data)
    test_list = json_data.get('test_list')
    print(test_list)    
    def Normalize2(list,low,high):
        list = np.array(test_list)
        delta = high - low
        if delta != 0:
            for i in range(0, len(list)):
                list[i] = (list[i]-low)/delta
        return  list.tolist()
    test_inputs = Normalize2(test_list,low,high)
    print(test_inputs)  
    ###预测
    model.eval()    
    for i in range(fut_pred):
        seq = torch.FloatTensor(test_inputs[-train_window:])  ##是个张量    
        print(seq)
        with torch.no_grad():
            model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                            torch.zeros(1, 1, model.hidden_layer_size))
            pre = model(seq)########################## 预测的结果
            print(pre)  
            test_inputs.append(model(seq).item())########################## 模型预测时模型的输入    
    print(test_inputs) #它包含13个元素。5+8
    print(len(test_inputs))#13
    test_pre = test_inputs[-fut_pred:] #最后8个预测结果 
    def FNoramlize(list,low,high):
        delta = high - low
        if delta != 0:
            for i in range(0, len(list)):
                list[i] = list[i]*delta + low
        return list 
    test_real = FNoramlize(test_pre,low,high)
    print(test_real)

    result["prediction"] = test_real
    result["success"] = True    
    return jsonify(result)  

