import numpy as np
import torch
from torch import nn

class minmax_normalizer(nn.Module):
    def __init__(self, maximum=None, minimum=None, fitting_lock=False):
        super().__init__()
        self.maximum = maximum
        self.minimum = minimum
        self.fitting_lock = fitting_lock
        
    def fit(self, data):
        #for adapting the old version
        if(not hasattr(self,'minimum')):
            self.minimum = 0
            self.fitting_lock = False
            
        if(self.fitting_lock==False):
            self.maximum = (data).max()
            self.minimum = (data).min()
        else:
            pass
    
    def normalize(self, data):
        #for adapting the old version
        if(not hasattr(self,'minimum')):
            self.minimum = 0
            self.fitting_lock = False
        
        if isinstance(data, torch.Tensor):
            data[data>self.maximum] = torch.Tensor([self.maximum]).to(data.device)
            data[data<self.minimum] = torch.Tensor([self.minimum]).to(data.device)
        else:
            data[data>self.maximum] = self.maximum
            data[data<self.minimum] = self.minimum
        

        
        return (data - self.minimum) / (self.maximum - self.minimum + 1e-16)

    def denormalize(self, data):
        #for adapting the old version
        if(not hasattr(self,'minimum')):
            self.minimum = 0
            self.fitting_lock = False
        return data*(self.maximum - self.minimum) + self.minimum


class standard_normalizer(nn.Module):
    def __init__(self, mean=None, std=None, fitting_lock=False):
        super().__init__()
        self.mean = None
        self.std = None
        self.fitting_lock = fitting_lock

    def fit(self, data):
        if(self.fitting_lock==False):
            self.mean = (data).mean()
            self.std = (data).std()
        else:
            pass
        
    def normalize(self, data):
        return (data - self.mean) / (self.std + 1e-16)

    def denormalize(self, data):
        return data*self.std + self.mean


class Normalizer(nn.Module):
    def __init__(self, group_index, method, name, hold_out_set=[]):
        super().__init__()
        '''
        group_index : A list contain index of array you want to group to normalize together.
        method : A list of class, ether 'scaler.maximum_normalize' or 'scaler.stander_normalize'.
        name : A list of string containing the name of each group.
        '''
        self.scaler_group = {n: m for n, m in zip(name, method)}
        self.group_index = {n: i for n, i in zip(name, group_index)}
            
        self.name = name
        self.hold_out_set = hold_out_set
        self.total_dim = None

    def fit(self, data):
        self.total_dim = data.shape[-1]
        for N in self.name:
            idx = [item for item in self.group_index[N] if item not in self.hold_out_set]
            self.scaler_group[N].fit(data[..., idx])

    def normalize(self, data, related_idx=None):
        if(related_idx is None):
            try:
                result = data.clone()
            except:
                result = np.copy(data)
            
            for N in self.name:
                result[..., self.group_index[N]] = self.scaler_group[N].normalize(
                    data[..., self.group_index[N]])
            
            return result
        
        else:
            datatype = str(type(data))
            if('torch' in datatype):
                expand_data = torch.zeros([data.shape[0],self.total_dim]).to(data.device)
                expand_data[...,related_idx] = expand_data[...,related_idx] + data
            else:
                expand_data = np.zeros([data.shape[0],self.total_dim])
                expand_data[...,related_idx] = expand_data[...,related_idx] + data

            try:
                result = expand_data.clone()
            except:
                result = np.copy(expand_data)

            for N in self.name:
                result[..., self.group_index[N]] = self.scaler_group[N].normalize(
                    expand_data[..., self.group_index[N]])
            
            return result[...,related_idx]

    def denormalize(self, data, related_idx=None):
        if(related_idx is None):
            try:
                result = data.clone()
            except:
                result = np.copy(data)
            
            for N in self.name:
                result[..., self.group_index[N]] = self.scaler_group[N].denormalize(
                    data[..., self.group_index[N]])
            
            return result
        
        else:
            datatype = str(type(data))
            if('torch' in datatype):
                expand_data = torch.zeros([data.shape[0],self.total_dim]).to(data.device)
                expand_data[...,related_idx] = expand_data[...,related_idx] + data
            else:
                expand_data = np.zeros([data.shape[0],self.total_dim])
                expand_data[...,related_idx] = expand_data[...,related_idx] + data
            
            try:
                result = expand_data.clone()
            except:
                result = np.copy(expand_data)
            
            for N in self.name:
                result[..., self.group_index[N]] = self.scaler_group[N].denormalize(
                    expand_data[..., self.group_index[N]])
            
            return result[...,related_idx]
    
    def Info(self):
        for N in self.name:
            print('The idx of '+N+' is '+str(self.group_index[N]))
            try:
                print('Mean: '+str(self.scaler_group[N].mean))
                print('Std: '+str(self.scaler_group[N].std))
            except:
                print('Min: '+str(self.scaler_group[N].minimum))
                print('Max: '+str(self.scaler_group[N].maximum))

class Null_Normalizer(nn.Module):
    def __init__(self):
        super().__init__()
        self.normalize_ = nn.Identity()
        self.denormalize_ = nn.Identity()
        
    def normalize(self, x, other=None):
        return self.normalize_(x)
    
    def denormalize(self, x, other=None):
        return self.denormalize_(x)

if __name__ == '__main__':
    #for testing
    
    a = np.arange(100).reshape([4,25]).T.astype(float)
    
    sc = Normalizer(group_index=[[0,1],
                             [2,3]], 
                method=[standard_normalizer(),
                        standard_normalizer()], 
                name=['mean25',
                      'mean50'], 
                hold_out_set=[])
    
    sc.fit(data=a)
    name = ['mean25',
            'mean50']
    for N in name:
        print(N+':',sc.scaler_group[N].mean,'/',sc.scaler_group[N].std)


    na = sc.normalize(a[...,[0,3]],[0,3])
    na_gt = sc.normalize(a)[...,[0,3]]
    
    rec_gt = sc.denormalize(na_gt,[0,3])

