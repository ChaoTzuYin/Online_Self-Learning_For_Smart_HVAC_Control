import torch
import numpy as np

class MEMORY():
    def __init__(self, capacity=240*42, data_key = ['st','at','st_plus_1','r']):
        self.data_key = data_key
        self.dataset = {N:[] for N in self.data_key}
        self.capacity = capacity
        
    def initialize(self, args):
        self.record(args)
        print('Initialization was done. \nCurrent data samples: ',len(self))

    def __getitem__(self,idx):
        return [self.dataset[key][idx][0] for key in self.data_key]

    def __call__(self, batch_size, shuffle=True):
        return torch.utils.data.DataLoader(self, batch_size=batch_size, shuffle=shuffle)
    
    def __len__(self):
        assert len(self.dataset['st'])==len(self.dataset['at'])==len(self.dataset['st_plus_1'])==len(self.dataset['r'])
        return len(self.dataset['st'])

    def record(self, data):
        for key, item in zip(self.data_key, data):
            self.dataset[key]+=np.split(item,item.shape[0],0)
        
        current_len = self.__len__()
        if(current_len>self.capacity):
            overflow = current_len - self.capacity
            for key, item in zip(self.data_key, data):
                del(self.dataset[key][:overflow])
            
    def get_last_n(self, n=5):
        return [torch.from_numpy(np.concatenate(self.dataset[key][-n:],0).astype(np.float32)) for key in self.data_key]

    def delete_first(self, n=1):
        for key in self.data_key:
            del(self.dataset[key][0])

    def Info(self):
        print('Current data samples: ',self.__len__())
        for item in self.data_key:
            print('Data['+item+'] with shape '+str(np.concatenate(self.dataset[item],0).shape))

