# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 19:02:11 2022
@author: Chao, Tzu-Yin
"""
import torch
from torch import nn
import pandas as pd
import Trainer as tr
import Network as nk
import Normalizer as sc
import Actor as ac
import Memory as mom
import numpy as np
        
class OSLN_Prototype(nn.Module):
    #Abstract class
    def __init__(self):
        super().__init__()
        self.Trainer = None
        self.Normalizer = None #Done
        self.Memory = None #Done
        self.Model = None 
        self.Actor = None #Done
    
    def Record(self, data_in):
        #data_in: List of Numpy array with shape [N,] or [1,N]
        if(len(data_in[0].shape)<2):
            data_in = [d[None] for d in data_in]
        self.Memory.record(data_in)
        self.Model.eval()
        self.Actor.update_confidence(self.Trainer, self.Memory)
        print('Data recorded.')
    
    def Choose_Action(self, Current_State, Target, TSPexpert):
        #Current_State: Numpy array with shape [N,]
        #Target: Float value representing the target temperature wanted by the users.
        self.Model.eval()
        Action = self.Actor(self.Model, Current_State, Target, TSPexpert)
        #Action: Numpy array with shape [B,1]
        return Action
    
    def Learn(self, epoch=10, show_info=False):
        self.Model.train()
        for e in range(epoch):
            train_info = self.Trainer(self.Memory)
            if(show_info):
                print(train_info)
        print('Finish learning.')

    def Info(self):
        print('########Trainer##########')
        try:
            self.Trainer.Info()
        except:
            print('WARNING: Cannot get info from Trainer. Please remember to add the function "info" for a better function management.')
        
        print('########Normalizer##########')
        try:
            self.Normalizer.Info()
        except:
            print('WARNING: Cannot get info from Normalizer. Please remember to add the function "info" for a better function management.')
                
        print('########Memory##########')
        try:
            self.Memory.Info()
        except:
            print('WARNING: Cannot get info from Memory. Please remember to add the function "info" for a better function management.')
               
        print('########Model##########')
        try:
            self.Model.Info()
        except:
            print('WARNING: Cannot get info from Model. Please remember to add the function "info" for a better function management.')
                
        print('########Actor##########')
        try:
            self.Actor.Info()
        except:
            print('WARNING: Cannot get info from Actor. Please remember to add the function "info" for a better function management.')


class TD_Simulation_Agent(OSLN_Prototype):
    def __init__(self,
                 nstep=1,
                 GARMMA=0.05,
                 BETA=0.5,
                 SIGMA=0.9,
                 update_target_per=100,
                 batch_size=1000,
                 initial_datapath='./data_logs/15A_dataset_from_real_env/clean_dataset_15A_AHU301_2018_whole.csv',
                 solution='modify_coefficients',
                 lr=1e-3,
                 a_dim=np.array([0]),
                 s_dim=np.array([i for i in range(1, 6)]),
                 e_dim=np.array([i for i in range(6, 37)])):
        super().__init__()
        self.Normalizer = sc.Normalizer(group_index=[[0, 1, 2, 3, 4, 5,
                                                        11, 12, 13, 14, 15,
                                                        32, 33],
                                                       [ 6,  7,  8,  9, 10,
                                                        16, 17, 18, 19, 20],
                                                       [31],
                                                       [35, 36]],
                                          method=[sc.standard_normalizer(),
                                                  sc.minmax_normalizer(),
                                                  sc.standard_normalizer(),
                                                  sc.standard_normalizer()],
                                          name=['t', 'level', 'p', 'kw'],
                                          hold_out_set=[0])
        
        self.Memory = mom.MEMORY()
        
        self.Model = nk.MainModel(a_dim=a_dim, 
                                  s_dim=s_dim, 
                                  e_dim=e_dim)
        
        self.Actor = ac.Actor(BETA=0.5, 
                              SIGMA=0.9, 
                              ALPHA = 0.5,
                              solution = solution)
        self.Trainer = tr.Trainer(self.Model,GARMMA,lr,batch_size,update_target_per)

        # initialize the normalizer and the memory        
        data_T = pd.read_csv(initial_datapath).values[:240*3,5:]
        self.Normalizer.fit(data_T)
        self.Model.Set_Normalizer(self.Normalizer)
        
        if(type(a_dim) is list):
            self.a_dim = np.array(a_dim)
            self.s_dim = np.array(s_dim)
            self.e_dim = np.array(e_dim)
        
        else:
            self.a_dim = a_dim
            self.s_dim = s_dim
            self.e_dim = e_dim      
        
        self.se_dim = np.concatenate([s_dim, e_dim],-1)
        
        initial_st = data_T[:-nstep, self.se_dim]
        initial_at = data_T[:-nstep, self.a_dim]
        initial_st_plus_1 = data_T[nstep:, self.se_dim]
        initial_r = data_T[nstep:, self.s_dim]
        
        self.Memory.initialize([initial_st, initial_at, initial_st_plus_1, initial_r])

if __name__ == '__main__':
    #for testing
    import numpy as np
    a = TD_Simulation_Agent().cuda()
    a.Info()
    a.Choose_Action(np.arange(37), 25.25, 22.7)
    a.Learn(20000,True)
    a.Record([np.arange(36),
              np.arange(1),
              np.arange(36),
              np.arange(5),])
