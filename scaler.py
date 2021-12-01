import numpy as np


class maximum_normalize():
    def __init__(self, maximum=None):
        self.maximum = maximum

    def fit(self, data):
        self.maximum = (data).max()

    def normalize(self, data):
        return data/self.maximum

    def denormalize(self, data):
        return data*self.maximum


class stander_normalize():
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = (data).mean()
        self.std = (data).std()

    def normalize(self, data):
        return (data-self.mean)/self.std

    def denormalize(self, data):
        return data*self.std + self.mean


class scaler():
    def __init__(self, group_index, method, name):
        '''
        group_index : A list contain index of array you want to group to normalize together.
        method : A list of class, ether 'scaler.maximum_normalize' or 'scaler.stander_normalize'.
        name : A list of string containing the name of each group.
        '''
        self.scaler_group = {n: m for n, m in zip(name, method)}
        self.group_index = {n: i for n, i in zip(name, group_index)}
        self.name = name

    def fit(self, data):
        for N in self.name:
            self.scaler_group[N].fit(data[..., self.group_index[N]])

    def normalize(self, data, name=None):
        if(name == None):
            try:
                result = data.clone()
            except:
                result = np.copy(data)
            for N in self.name:
                result[..., self.group_index[N]] = self.scaler_group[N].normalize(
                    data[..., self.group_index[N]])
            return result
        else:
            return self.scaler_group[name].normalize(data)

    def denormalize(self, data, name=None):
        if(name == None):
            try:
                result = data.clone()
            except:
                result = np.copy(data)
            for N in self.name:
                result[..., self.group_index[N]] = self.scaler_group[N].denormalize(
                    data[..., self.group_index[N]])
            return result
        else:
            return self.scaler_group[name].denormalize(data)
