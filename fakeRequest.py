"""
@ 2021/04/27 by Chao Tzu-Yin, Huang Sin-Han
"""
import numpy as np
import torch
import pandas as pd
from environment_model import EnvironmentModel

class fakeRequest():
    def __init__(self, load_from_dict=False):     
        self.environment = torch.load('environment_model_weight/best_model.pth', map_location='cpu').cuda()
        #set up simulation situation
        df = pd.read_csv('./data_logs/clean_data_15A_3F_AHU301_2018.csv',header=None)
        self.base_data = df.values[...,5:]
        self.base_Y = df.values[...,0]
        self.base_M = df.values[...,1]
        self.base_D = df.values[...,2]
        self.base_Hr = df.values[...,3]
        self.base_Min = df.values[...,4]
        
        self.max_steps = len(self.base_data)
        self.pointer = 0
        
        self.current_A = np.array([21]) #[1,]
        self.current_S = self.base_data[0,1:6] #[5,]
        
        self.uncontrolable_factors = self.base_data[:,6:]
        
        self.current_E = np.copy(self.uncontrolable_factors[0]) #[31,]       

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
        
        if(self.is_saving_mode()):
            prediction_input = torch.from_numpy(np.concatenate([self.current_A + 4,
                                                                 self.current_S,
                                                                 self.current_E],0)[None]).float().cuda() #[1,37]
        
        else:
            prediction_input = torch.from_numpy(np.concatenate([self.current_A,
                                                                 self.current_S,
                                                                 self.current_E],0)[None]).float().cuda() #[1,37]

        assert list(prediction_input.shape) == [1,37], 'shape incorrect.'
        out_tenser = self.environment(prediction_input) #[1,5]
        self.current_S = out_tenser[0].cpu().detach().numpy()
        self.current_E = np.copy(self.uncontrolable_factors[self.pointer])
          
    
    def get_current_state(self):
        return np.concatenate([self.current_A,self.current_S,self.current_E],0)[None]
    
    def set_action(self, setpoint):
        #setpoint: a float value
        self.current_A = np.array([setpoint])
        print('AHU TSP has been changed to ',self.current_A, 'degree.')
