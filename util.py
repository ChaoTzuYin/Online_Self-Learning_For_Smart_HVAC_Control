import csv   
import numpy as np
import datetime
import os
import torch
    
def get_current_time():
    now = datetime.datetime.now()
    Date, Time = str(now).split(' ')
    Y,M,D = Date.split('-')
    Hr,Min = Time.split('.')[0].split(':')[:-1]
    return [float(item) for item in [Y,M,D,Hr,Min]]

def record_data(np_array, filename='dataset'):
    fields=np_array[0].tolist()
    with open(r'./data_logs/'+filename+'.csv', 'a') as f:
        writer = csv.writer(f)
        writer.writerow(fields)
        
def save_model(model, checkpoint_dir):
    current_records = os.listdir(checkpoint_dir)
    idx = [int(item.split('.')[1]) for item in current_records]
    latest_idx = max(idx)
    torch.save(model, checkpoint_dir+'/OSLN_plus.'+str(latest_idx+1)+'.pth') 