#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 07:01:21 2021

@author: user
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import scaler as sc
from environment_model import EnvironmentModel


'''
data_dict['AHUTSP'] = cleaned_data[...,0:1]
data_dict['VAVRAT'] = cleaned_data[...,1:6]
data_dict['VAVZI'] = cleaned_data[...,6:11]
data_dict['FCURAT'] = cleaned_data[...,11:16]
data_dict['FCUCV'] = cleaned_data[...,16:21]
data_dict['FCUMOD'] = cleaned_data[...,21:26]
data_dict['VAVMOD'] = cleaned_data[...,26:31]
data_dict['AHUPS'] = cleaned_data[...,31:32]
data_dict['MAUTS'] = cleaned_data[...,32:33]
data_dict['OAT'] = cleaned_data[...,33:34]
data_dict['RS'] = cleaned_data[...,34:35]
data_dict['LOAD'] = cleaned_data[...,35:37]
'''
torch.multiprocessing.freeze_support()
cleaned_data = pd.read_csv('./data_logs/clean_data_15A_3F_AHU301_2018.csv',header=None).values[...,5:]
training_data = cleaned_data[:15240]
testing_data = cleaned_data[15240:]

class Data(torch.utils.data.Dataset):
    def __init__(self, x, step=1):
        self.x = x[:-step]
        self.y = x[step:][...,1:6]
        
    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)
        
#test model
ENV = EnvironmentModel().cuda()

train_set = Data(training_data)
train_loader = torch.utils.data.DataLoader(dataset=train_set, 
                                           batch_size=1000, 
                                           shuffle=True, 
                                           num_workers=4, 
                                           pin_memory=True)

test_set = Data(testing_data)
test_loader = torch.utils.data.DataLoader(dataset=test_set, 
                                           batch_size=1000, 
                                           shuffle=True, 
                                           num_workers=4, 
                                           pin_memory=True)


optimizer = torch.optim.Adam(ENV.parameters(),lr=1e-3)

relation_label = torch.from_numpy(np.array([ 1, #AHU TSP
                                             1, 1, 1, 1, 1, #VAV RAT
                                            -1,-1,-1,-1,-1, #VAV ZI
                                             1, 1, 1, 1, 1, #FCU RAT
                                            -1,-1,-1,-1,-1, #FCU CV
                                            -1,-1,-1,-1,-1, #FCU CMD
                                             1, 1, 1, 1, 1, #VAV CMD
                                            -1, #AHU PS
                                             1, #MAU TS
                                             1, #OAT
                                            -1, #RS
                                             1,1  #load
                                            ])).float().cuda()

historical_low = 1000
MAXEPOCH = 3000
if __name__=='__main__':
    for epoch in range(MAXEPOCH):
        train_err = []
        train_relation_loss = []
        for count, item in enumerate(train_loader):
            x_, y_ = item[0].cuda(), item[1].cuda()
            pred, _ = ENV.train_forward(x_.float())
            
            Uniform_permutation = torch.rand_like(x_)
            
            pred, norm_x = ENV.train_forward(x_.float())
            
            optimizer.zero_grad()
            grads_val = torch.autograd.grad(outputs=pred, inputs=norm_x,
                                  grad_outputs=torch.ones_like(pred),
                                  create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]

            constraint_loss = 10*torch.mean(torch.sum(torch.clamp(-relation_label[None]*grads_val, 0),-1))
        
            loss = torch.mean((pred - y_)**2) + constraint_loss
            loss.backward()
            optimizer.step()
        
            train_err += [torch.mean(torch.abs(pred - y_)).item()]
            train_relation_loss += [constraint_loss.item()]
    
        test_err = []
        test_relation_loss = []
    
        for count, item in enumerate(test_loader):
            x_, y_ = item[0].cuda(), item[1].cuda()
            pred, norm_x = ENV.train_forward(x_.float())
        
            grads_val = torch.autograd.grad(outputs=pred, inputs=norm_x,
                                        grad_outputs=torch.ones_like(pred),
                                        create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]

            constraint_loss = 10*torch.mean(torch.sum(torch.clamp(-relation_label[None]*grads_val, 0),-1))
        
        
            test_err += [torch.mean(torch.abs(pred - y_)).item()]
            test_relation_loss += [constraint_loss.item()]
    
        if sum(test_err)/len(test_err) < historical_low:
            historical_low = sum(test_err)/len(test_err)
            torch.save(ENV,'environment_model_weight/best_model_15A301.pth')
    
        print('[Epoch:',epoch,']')
        print('train_err=',sum(train_err)/len(train_err),
          ' / grad_err=',sum(train_relation_loss)/len(train_relation_loss))
        print('test_err=',sum(test_err)/len(test_err),
          ' / grad_err=',sum(test_relation_loss)/len(test_relation_loss))









