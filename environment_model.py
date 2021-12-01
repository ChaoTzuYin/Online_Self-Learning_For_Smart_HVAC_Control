#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 21 10:07:11 2021

@author: user
"""

import torch
import torch.nn as nn
import pandas as pd
import scaler as sc

class EnvironmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        cleaned_data = pd.read_csv('./data_logs/clean_data_15A_3F_AHU301_2018.csv').values[...,5:]
        training_data = cleaned_data[:15000]
        
        self.global_feature_extractor = nn.Sequential(nn.Linear(37, 128),
                                                 nn.LeakyReLU(0.2),
                                                 nn.Linear(128, 64),
                                                 nn.LeakyReLU(0.2))
        
        self.predictor = nn.Sequential(nn.Conv1d(64+5+1, 32, 1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv1d(32, 1, 1))
        
        self.state_normalizer = sc.scaler(group_index=[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 31, 32],
                                           [ 5,  6,  7,  8,  9, 
                                            15, 16, 17, 18, 19],
                                           [30],
                                           [34, 35]],
                                  method=[sc.stander_normalize(),
                                          sc.maximum_normalize(),
                                          sc.stander_normalize(),
                                          sc.stander_normalize()],
                                  name=['t', 'level', 'p', 'kw'])
        self.state_normalizer.fit(training_data[..., 1:])
        self.action_normalizer = self.state_normalizer.scaler_group['t']
        
    def train_forward(self, X):
        X = torch.cat([self.action_normalizer.normalize(X[:,:1]), 
                       self.state_normalizer.normalize(X[:,1:])],-1)
        X = torch.autograd.Variable(X, requires_grad=True)
        A = X[:,:1]
        S = X[:,1:6]
        E = X[:,6:]
        A = torch.where(A < torch.mean(X[:,1:6],-1)[...,None], 
                        A, 
                        torch.mean(X[:,1:6],-1)[...,None] + A - A.detach())
        
        place_label = torch.eye(S.shape[-1]).to(S.device)
        place_label_expand = place_label[None].expand(S.shape[0],-1,-1) #B, 5, 5
        
        global_inp = torch.cat([A,S,E],-1)
        global_feature = self.global_feature_extractor(global_inp) #B, 64
        global_feature_expand = global_feature[...,None].repeat(1,1,S.shape[-1]) #B, 64, 5
        
        local_inp = torch.cat([S[:,None], place_label_expand, global_feature_expand], 1)
        St_plus_1 = self.predictor(local_inp)[:,0]
        return St_plus_1, X
    
    def forward(self, X):
        X = torch.cat([self.action_normalizer.normalize(X[:,:1]), 
                       self.state_normalizer.normalize(X[:,1:])],-1)
        A = X[:,:1]
        S = X[:,1:6]
        E = X[:,6:]
        A = torch.where(A < torch.mean(X[:,1:6],-1)[...,None], A, torch.mean(X[:,1:6],-1)[...,None])
        
        place_label = torch.eye(S.shape[-1]).to(S.device)
        place_label_expand = place_label[None].expand(S.shape[0],-1,-1) #B, 5, 5
        
        global_inp = torch.cat([A,S,E],-1)
        global_feature = self.global_feature_extractor(global_inp) #B, 64
        global_feature_expand = global_feature[...,None].repeat(1,1,S.shape[-1]) #B, 64, 5
        
        local_inp = torch.cat([S[:,None], place_label_expand, global_feature_expand], 1)
        St_plus_1 = self.predictor(local_inp)[:,0]
        return St_plus_1


class EnvironmentModelReduceOnOff(nn.Module):
    def __init__(self):
        super().__init__()
        
        cleaned_data = pd.read_csv('./data_logs/clean_data_15A_3F_AHU301_2018.csv').values[...,5:]
        training_data = cleaned_data[:15000]
        
        self.global_feature_extractor = nn.Sequential(nn.Linear(27, 128),
                                                 nn.LeakyReLU(0.2),
                                                 nn.Linear(128, 64),
                                                 nn.LeakyReLU(0.2))
        
        self.predictor = nn.Sequential(nn.Conv1d(64+5+1, 32, 1),
                                  nn.LeakyReLU(0.2),
                                  nn.Conv1d(32, 1, 1))
        
        self.state_normalizer = sc.scaler(group_index=[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 31, 32],
                                           [ 5,  6,  7,  8,  9, 
                                            15, 16, 17, 18, 19],
                                           [30],
                                           [34, 35]],
                                  method=[sc.stander_normalize(),
                                          sc.maximum_normalize(),
                                          sc.stander_normalize(),
                                          sc.stander_normalize()],
                                  name=['t', 'level', 'p', 'kw'])
        self.state_normalizer.fit(training_data[..., 1:])
        self.action_normalizer = self.state_normalizer.scaler_group['t']
        
    def train_forward(self, X):
        X = torch.cat([self.action_normalizer.normalize(X[:,:1]), 
                       self.state_normalizer.normalize(X[:,1:])],-1)
        X = torch.autograd.Variable(X, requires_grad=True)
        A = X[:,:1]
        S = X[:,1:6]
        E = torch.cat([X[:,6:21], X[:,31:]],-1)
        A = torch.where(A < torch.mean(X[:,1:6],-1)[...,None], 
                        A, 
                        torch.mean(X[:,1:6],-1)[...,None] + A - A.detach())
        
        place_label = torch.eye(S.shape[-1]).to(S.device)
        place_label_expand = place_label[None].expand(S.shape[0],-1,-1) #B, 5, 5
        
        global_inp = torch.cat([A,S,E],-1)
        global_feature = self.global_feature_extractor(global_inp) #B, 64
        global_feature_expand = global_feature[...,None].repeat(1,1,S.shape[-1]) #B, 64, 5
        
        local_inp = torch.cat([S[:,None], place_label_expand, global_feature_expand], 1)
        St_plus_1 = self.predictor(local_inp)[:,0]
        return St_plus_1, X
    
    def forward(self, X):
        X = torch.cat([self.action_normalizer.normalize(X[:,:1]), 
                       self.state_normalizer.normalize(X[:,1:])],-1)
        A = X[:,:1]
        S = X[:,1:6]
        E = torch.cat([X[:,6:21], X[:,31:]],-1)
        A = torch.where(A < torch.mean(X[:,1:6],-1)[...,None], A, torch.mean(X[:,1:6],-1)[...,None])
        
        place_label = torch.eye(S.shape[-1]).to(S.device)
        place_label_expand = place_label[None].expand(S.shape[0],-1,-1) #B, 5, 5
        
        global_inp = torch.cat([A,S,E],-1)
        global_feature = self.global_feature_extractor(global_inp) #B, 64
        global_feature_expand = global_feature[...,None].repeat(1,1,S.shape[-1]) #B, 64, 5
        
        local_inp = torch.cat([S[:,None], place_label_expand, global_feature_expand], 1)
        St_plus_1 = self.predictor(local_inp)[:,0]
        return St_plus_1



