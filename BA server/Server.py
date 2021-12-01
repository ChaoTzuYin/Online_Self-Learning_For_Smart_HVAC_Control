"""
@ Script for exchanging data with TSMC's BA system. (F15A)
@ 2018/6/21 by Chao Tzu-Yin

need
    pop_alarm.py at least 1.0

@ 2021/04/26 by Chao Tzu-Yin
@ 2021/04/29 by Huang Sin-Han

code path:\\192.168.1.3\d\jason wang\AI\0621\
code path:'D:\jason wang\AI\0621\'
"""


import traceback
import subprocess

import win32com.client as win32

import datetime
import time
import os

import socket
import pickle


code_path = 'D:/\"jason wang\"/AI/0621/'

request_path = '//192.168.1.3/d/jason wang/AI/0621/request/'
answer_path = '//192.168.1.3/d/jason wang/AI/0621/answer/'

if not os.path.isdir(request_path):
    print("Can not find dir: ",request_path)
    print("Create dirs")
    os.makedirs(request_path)
if not os.path.isdir(answer_path):
    print("Can not find dir: ",answer_path)
    print("Create dirs")
    os.makedirs(answer_path)



#excel = win32.DispatchEx('Excel.Application') # danger open
excel = win32.gencache.EnsureDispatch('Excel.Application')


addin_path = 'C:/Program Files/Honeywell/Client/Xldataex/mede.xla'
excel.Workbooks.Open(addin_path)

wb = excel.Workbooks.Open('D:/jason wang/AI/0621/read.xlsx')

ws = wb.Worksheets('工作表1')


def get_value(name):   
    ret = []
    for item in name:
        #print(item)
        point_, type_ = item.split(".")
        ws.Cells(1,1).Formula = '=GetPointVal("f15p1ba01a","'+str(point_)+'","' + str(type_) + '")'
        ret.append(ws.Cells(1,1).Value)
    return ret


def set_value(package):
    ret = []
    for item, number in zip(package[0], package[1]):
        point_, type_ = item.split(".")
        ws.Cells(2,1).Formula = '=PutPointVal_Number("f15p1ba01a","'+str(point_)+'","sp",'+str(number)+')'
        ws.Cells(1,1).Formula = '=GetPointVal("f15p1ba01a","'+str(point_)+'","' + str(type_) + '")'
        ret.append(ws.Cells(1,1).Value)
    return ret


'''
def answer():
    global pop_alarm_counter
    try:
        with open(request_path+'request','rb') as f:
            data_arr = pickle.load(f)
        #print('Get messege:',data_arr)
        print('Connected!')
        ####excel data exchange####
        if(data_arr[0] == 'get'):
            ret = get_value(data_arr[1])
            print('Get sensor data: ', datetime.datetime.now())
        elif(data_arr[0] == 'set'):
            ret = set_value(data_arr[1])
            print('Set AHU setpoint: ', datetime.datetime.now())
        ###########################
        
        os.remove(request_path+'request')
    
        with open(answer_path+'answer','wb') as f:
            pickle.dump(ret,f)
        pop_alarm_counter = 0
        print('\n\nWait for client')
    except:
        print("\n\nError!!!\n")
        print(datetime.datetime.now())
        traceback.print_exc()
        print("\nError!!!\n\n")
                
        if (pop_alarm_counter <= 5):
            pop_alarm_counter += 1
            Execute = "conda activate python35 && python "+code_path+"pop_alarm.py 出現異常請檢視log"
            subprocess.Popen(Execute, shell=True, stdout=subprocess.PIPE)
            time.sleep(0.01)
        
        try:
            os.remove(request_path+'request')
        except:
            pass
        
        print('\n\nWait for client')
'''

def answer():
    success = False
    fall_counter = 0
    try:
        while(not success):
            try:
                with open(request_path+'request','rb') as f:
                    data_arr = pickle.load(f)
                success = True
            except:
                fall_counter += 1
                
                print("\n\nError!!!\n")
                print(datetime.datetime.now())
                traceback.print_exc()
                print("\nError!!!\n\n")
                
                if (fall_counter >= 3):
                    try:
                        os.remove(request_path+'request')
                    except:
                        pass
                    return
                
                time.sleep(2)
            
        #print('Get messege:',data_arr)
        print('Connected!')
        ####excel data exchange####
        if(data_arr[0] == 'get'):
            ret = get_value(data_arr[1])
            print('Get sensor data: ', datetime.datetime.now())
        elif(data_arr[0] == 'set'):
            ret = set_value(data_arr[1])
            print('Set AHU setpoint: ', datetime.datetime.now())
        ###########################
        
        os.remove(request_path+'request')
    
        with open(answer_path+'answer','wb') as f:
            pickle.dump(ret,f)
        print('\n\nWait for client')
    except:
        print("\n\nError!!!\n")
        print(datetime.datetime.now())
        traceback.print_exc()
        print("\nError!!!\n\n")
        
        try:
            os.remove(request_path+'request')
        except:
            pass
        
        print('\n\nWait for client')
        


pop_alarm_counter = 0
print('\n\nWait for client')
while(True):
    if len(os.listdir(request_path))!=0:
        answer()
    time.sleep(0.5)
