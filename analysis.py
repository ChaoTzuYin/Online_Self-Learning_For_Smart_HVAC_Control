import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os

def get_week_day(Y,M,D):
    week_day_dict = {
    0 : 'Sunday',
    1 : 'Monday',
    2 : 'Tuesday',
    3 : 'Wednesday',
    4 : 'Thursday',
    5 : 'Friday',
    6 : 'Saturday',
    }
    day=int(datetime.datetime(Y,M,D).strftime("%w"))
    return week_day_dict[day]

path_doc = ['./data_logs/allVAV_18min.csv',
            './data_logs/allVAV_30min.csv']

df = [pd.read_csv(path,header=None) for path in path_doc]

record = [d.values[:,5:] for d in df] #Start from the begining from a day
date = [d.values[:,:3] for d in df]
name = ['18 min',
        '30 min']


if(not os.path.isdir('./day_result')):
    os.mkdir('./day_result')

day = 0
total_days = int(record[0].shape[0]/240)
working_day_idx = []
for day in range(0, total_days):
    x=[0,20,40,60,80,100,120]
    y=['7:00','9:00','11:00','13:00','15:00','17:00','19:00']
    
    Y = int(date[0][240*day+70,0])
    M = int(date[0][240*day+70,1])
    D = int(date[0][240*day+70,2])
    weekday = get_week_day(Y,M,D)
    
    if(weekday != 'Saturday' and weekday != 'Sunday'):
        working_day_idx += [day]

    plt.figure()
    plt.ion()
    
    ctr_error = []
    
    for l in range(len(record)):
        plt.plot(record[l][240*day+70:240*day+190,0], label='Tsp '+name[l])
        temperature = np.mean(record[l][240*day+70:240*day+190,1:6],-1)
        plt.plot(temperature, label='RAT '+name[l])
        
        ctr_error += [np.mean(np.abs(temperature - 25.25))]
        
    dynamic_title_string = ''
    plt.plot(np.ones([120])*25.25, label='Texp')
    for l in range(len(record)):
        dynamic_title_string = dynamic_title_string + '\nError of ' + name[l] + ':' + str(ctr_error[l])
    plt.title('Day'+str(day)+':'+str(Y)+'/'+str(M)+'/'+str(D)+'('+weekday+')' + dynamic_title_string)
    plt.xticks(x,y)
    plt.ylim(19.,27.)
    plt.legend()
    plt.ioff()
    plt.savefig("./day_result/day" + str(day) + ".png")
 
daily_control_record = [record[l][:total_days*240].reshape([total_days, 240, 37]) for l in range(len(record))]
daily_control_record = [daily_control_record[l][:,70:190] for l in range(len(daily_control_record))]
daily_control_record = [daily_control_record[l][working_day_idx] for l in range(len(daily_control_record))]

err = [np.mean(np.abs(np.mean(daily_control_record[l][...,1:6],-1) - 25.25),-1) for l in range(len(daily_control_record))]

plt.figure()
plt.title('Average of daily control error during the control period\nTarget:25.25')
for l in range(len(err)):
    plt.plot(err[l], label='Control Error '+name[l]+'\nAverage:'+str(err[l].mean(0))+"\nMax:"+str(err[l].max(0)))
plt.xlabel('# Day')
plt.ylabel('Error (°C)')
plt.legend()
plt.savefig("./day_result/compare_diff.png")


daily_control_record = [record[l][:total_days*240].reshape([total_days, 240, 37]) for l in range(len(record))]
daily_control_record = [daily_control_record[l][:,70:190] for l in range(len(daily_control_record))]


err = [np.mean(np.abs(np.mean(daily_control_record[l][...,1:6],-1) - 25.25),-1) for l in range(len(daily_control_record))]
np_err = np.stack(err,0) # n, 127
var_err = np.var(np_err,0) # 127
top_5 = np.argsort(-var_err,-1)[:5]

for rank, day in enumerate(top_5):
    x = [0, 20, 40, 60, 80, 100, 120]
    y = ['7:00', '9:00', '11:00', '13:00', '15:00', '17:00', '19:00']

    Y = int(date[240*day+70, 0])
    M = int(date[240*day+70, 1])
    D = int(date[240*day+70, 2])
    weekday = get_week_day(Y, M, D)
    
    if(weekday != 'Saturday' and weekday != 'Sunday'):
        working_day_idx += [day]
        
    f = plt.figure()
    plt.ion()

    colors=['darkolivegreen', 'green', 'lime', 'teal', 'cyan', 'deepskyblue', 'blue', 'navy', 'mediumpurple', 'purple', 'red', 'black']
    plt.gca().set_prop_cycle(color=colors)
    # plt.plot(record[240*day+70:240*day+190, 0], label='Tsp')
    for l in range(len(record)):
        plt.plot(record[l][240*day+70:240*day+190,0], label='Tsp '+name[l])
        temperature = np.mean(record[l][240*day+70:240*day+190,1:6],-1)
        plt.plot(temperature, label='RAT '+name[l])
    plt.title('Day'+str(day)+'\n'+str(Y)+'/'+str(M)+'/'+str(D)+'('+weekday+')'+'\nVar:'+str(var_err[day])+'\nRain:'+str(np.mean(record[240*day+70:240*day+190, -3:-2])))     
        
    plt.plot(np.ones([120])*25.25, label='Texp')
    plt.plot(np.mean(record[240*day+70:240*day+190][..., -4:-3], -1), linestyle="--", label='Out door temperature')
    plt.plot(np.mean(record[240*day+70:240*day+190][..., 11:16], -1), linestyle="--", label='FCU RAT')
        

    plt.xticks(x, y)
    plt.ylim(19., 35.)
    plt.legend()
    plt.ioff()
    # plt.show()
    plt.savefig("./day_result/top5_diff/top"+str(rank)+".png")
    plt.close(f)

top_5_err = np.argsort(-np.mean(np_err,0))[:5]

for rank, day in enumerate(top_5_err):
    x = [0, 20, 40, 60, 80, 100, 120]
    y = ['7:00', '9:00', '11:00', '13:00', '15:00', '17:00', '19:00']

    Y = int(date[240*day+70, 0])
    M = int(date[240*day+70, 1])
    D = int(date[240*day+70, 2])
    weekday = get_week_day(Y, M, D)
    
    if(weekday != 'Saturday' and weekday != 'Sunday'):
        working_day_idx += [day]
        
    f = plt.figure()
    plt.ion()

    colors=['darkolivegreen', 'green', 'lime', 'teal', 'cyan', 'deepskyblue', 'blue', 'navy', 'mediumpurple', 'purple', 'red', 'black']
    plt.gca().set_prop_cycle(color=colors)
    # plt.plot(record[240*day+70:240*day+190, 0], label='Tsp')
    for l in range(len(record)):
        plt.plot(record[l][240*day+70:240*day+190,0], label='Tsp '+name[l])
        temperature = np.mean(record[l][240*day+70:240*day+190,1:6],-1)
        plt.plot(temperature, label='RAT '+name[l])

    plt.title('Day'+str(day)+'\n'+str(Y)+'/'+str(M)+'/'+str(D)+'('+weekday+')'+'\nMean ERR:'+str(np.mean(np_err,0)[day])+'\nRain:'+str(np.mean(record[240*day+70:240*day+190, -3:-2])))     
    plt.plot(np.ones([120])*25.25, label='Texp')
    plt.plot(np.mean(record[240*day+70:240*day+190, -4:-3], -1), linestyle="--", label='Out door temperature')
    plt.plot(np.mean(record[240*day+70:240*day+190, 11:16], -1), linestyle="--", label='FCU RAT')


    plt.xticks(x, y)
    plt.ylim(19., 35.)
    #plt.legend()
    plt.ioff()
    # plt.show()
    plt.savefig("./day_result/top5_err/top"+str(rank)+".png")
    plt.close(f)



