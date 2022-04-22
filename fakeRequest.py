import numpy as np
import torch
import pandas as pd
from environment_model import EnvironmentModel
import os

class fakeRequest():
    def __init__(self, File_path, Environment_path, device, 
                 s_dim=[1,2,3,4,5], 
                 a_dim=[0],
                 start_point=0,
                 end_point = None,
                 split_date=False, 
                 load_from_dict=False, 
                 noisy_ratio=None):  
        self.device=device
        self.environment = torch.load(Environment_path, map_location=device).to(device)
        df = pd.read_csv(File_path,header=None).values
        if(end_point==None):
            end_point = len(df)
        df = df[start_point:end_point]
        
        #set up simulation situation
        if(split_date):
            
            self.base_data = df[...,1:].astype(np.float32)
            base_date = df[...,0]
            Y, M, D, Hr, Min = [], [], [], [], []
            for date in base_date:
                DATE, TIME = date.split(' ')
                y, m, d = DATE.split('-')
                hr, min_ = TIME.split(':')[:-1]
                Y.append(int(y))
                M.append(int(m)) 
                D.append(int(d)) 
                Hr.append(int(hr))
                Min.append(int(min_))
            self.base_Y = np.array(Y)
            self.base_M = np.array(M)
            self.base_D = np.array(D)
            self.base_Hr = np.array(Hr)
            self.base_Min = np.array(Min)            
            
        else:
            df = df.astype(np.float32)
            self.base_data = df[...,5:]
            self.base_Y = df[...,0]
            self.base_M = df[...,1]
            self.base_D = df[...,2]
            self.base_Hr = df[...,3]
            self.base_Min = df[...,4]
            
        self.s_dim = s_dim
        self.a_dim = a_dim
  
        self.max_steps = len(self.base_data)
        self.pointer = 0
        
        self.current_A = np.array([21]) #[1,]
        self.current_S = self.base_data[0,self.s_dim] #[s_dim,]
        self.current_SAE = self.base_data[0].copy() #[all_dim,]
            
        self.noisy_ratio = noisy_ratio
        
    def get_time(self):
        Y, M, D = self.base_Y[self.pointer], self.base_M[self.pointer], self.base_D[self.pointer]
        Hr, Min = self.base_Hr[self.pointer], self.base_Min[self.pointer]
        return Y, M, D, Hr, Min
    
    def is_saving_mode(self):
        _, _, _, Hr, _ = self.get_time()
        if(Hr>=19 or Hr<7):
            return True
        else:
            return False
    
    def next_simulation_step(self):
        self.pointer += 1  
        
        # night
        if(self.is_saving_mode()):
            self.current_SAE[self.a_dim] = self.current_A+4
            self.current_SAE[self.s_dim] = self.current_S
            prediction_input = torch.from_numpy(self.current_SAE[None]).float().to(self.device) #[1,37]
        
        else:
            self.current_SAE[self.a_dim] = self.current_A
            self.current_SAE[self.s_dim] = self.current_S
            prediction_input = torch.from_numpy(self.current_SAE[None]).float().to(self.device) #[1,37]
        
        out_tenser = self.environment(prediction_input) #[1,5]
        self.current_S = out_tenser[0].cpu().detach().numpy()
        
        self.current_SAE = np.copy(self.base_data[self.pointer])
        self.current_SAE[self.s_dim] = self.current_S
        self.current_SAE[self.a_dim] = self.current_A
         
        if(self.noisy_ratio!=None and self.noisy_ratio!=0):
            self.current_S = self.current_S + np.random.uniform(-self.noisy_ratio/2,
                                                                self.noisy_ratio/2,
                                                                size=self.current_S.shape)
    
    def get_current_state(self):
        self.current_SAE[self.s_dim] = self.current_S
        self.current_SAE[self.a_dim] = self.current_A
        return self.current_SAE[None]
    
    def set_action(self, setpoint):
        #setpoint: a float value
        self.current_A = np.array([setpoint])
        print('AHU TSP has been changed to ',self.current_A, 'degree.')
