"""
@ 2021/04/27 by Chao Tzu-Yin
@ 2021/05/03 by Huang Sin-Han

code path:D:\AI\PythonEnvironment\official_code\
"""
import logging
import OSLN_plus
import numpy as np
import os
import torch
from fakeRequest import fakeRequest
import util
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # -1:CPU, >=0:GPU
###load model###
checkpoint_dir = './model_weights_two_stages'
if(not os.path.isdir(checkpoint_dir)):
    os.mkdir(checkpoint_dir)
    
current_records = os.listdir(checkpoint_dir)
if len(current_records) == 0:
    AGENT = OSLN_plus.TD_Simulation_Agent(nstep=3,  # 18 minutes
                                         EPSILON=0.8,
                                         GARMMA=0.25,
                                         BETA=0.9,
                                         SIGMA=0.5,
                                         update_target_per=100,
                                         batch_size=1000,
                                         global_step=0,
                                         load_weight=False,
                                         initial_datapath='./data_logs/bias_data_15A_3F_AHU301_2018.csv',
                                         solution='two_stages')
    
    for i in range(10):
        print('confident:', AGENT.learn(iteration=100))
    torch.save(AGENT, checkpoint_dir+'/OSLN_plus.0.pth')
else:
    idx = [int(item.split('.')[1]) for item in current_records]
    latest_idx = max(idx)
    AGENT = torch.load(checkpoint_dir+'/OSLN_plus.'+str(latest_idx)+'.pth')
    print('Restore from "'+checkpoint_dir + '/OSLN_plus.'+str(latest_idx)+'.pth" ')


# logger


logging.basicConfig(
    filename='./program.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


###############################SETTINGS#########################

Save_checkpoint_per_N_steps = 120

Timesetp_perminute = 6  # Recording sample rate: 6 minute
Change_setting_per_N_Timestep = 3  # this means 6*3=18 minute

Agent_start_to_work_at = 7  # when will the agent start to work
Agent_stop_to_work_at = 19  # when will the agent stop to work
After_work_please_set_the_setpoint_back_to = 26  # what's the temperature should be after the agent stop
offset_in_BA = 4  # what's the offset in BA (This will affect the learning result if it's not correct)

Texp = 25.25      # The expected temperature we hope to have in the office
TSPexpert = 22.7  # The reference temperature setpoint (which usually set before the AI is on)

################################################################
data_t_1 = None
Goin_time = None
################################################################

def get_flags(Hr, Min):
    
    current_time_step = (Hr * 60 + Min - Agent_start_to_work_at * 60)
    
    ##per time step##
    time_to_record = (((Hr * 60 + Min - Agent_start_to_work_at * 60) % Timesetp_perminute) == 0)
    
    ##saving time##
    time_to_save_model = (((Hr * 60 + Min - Agent_start_to_work_at * 60) % (Timesetp_perminute*Save_checkpoint_per_N_steps)) == 0)
 
    ##it should be in the range of control time period##
    Over_working_time = ((Hr * 60 + Min - Agent_stop_to_work_at * 60) >= 0)
    during_the_control_period = ((current_time_step >= 0) and not Over_working_time)
    
    ##it should be the right time step##
    in_the_right_time_step = ((current_time_step % (Timesetp_perminute * Change_setting_per_N_Timestep)) == 0)

    return time_to_record, time_to_save_model, during_the_control_period, in_the_right_time_step

if __name__ == "__main__":
    Request = fakeRequest()
    at = np.array([[TSPexpert]])
    logger = logging.getLogger(__name__)
    
    AGENT.alpha = 0.5
    
    for simulation_steps in range(Request.max_steps):
        
        if(simulation_steps%240==0):
            print('Day ',int(simulation_steps/240),' / ',Request.max_steps,' starts.')
        
        now = Request.get_time()
        Hr = now[-2]
        Min = now[-1]

        time_to_record, time_to_save_model, during_the_control_period, in_the_right_time_step = get_flags(Hr, Min)

        if(Min != Goin_time):
            Goin_time = None

        if(time_to_record and Min != Goin_time):
            Goin_time = Min

            if(time_to_save_model):
                #util.save_model(AGENT, checkpoint_dir)
                print('[', int(Hr), ':', int(Min), ']: A new model checkpoint has been created.')
                logger.info('A new model checkpoint has been created.')
            
            if(in_the_right_time_step):
                
                data_t = Request.get_current_state()
                
                if(data_t_1 is not None):
                    AGENT.record(data_t_1[..., 1:],  # St_1
                                 data_t_1[..., :1],  # At_1
                                 data_t[..., 1:],  # St
                                 data_t[..., 1:6])  # Ground Truth (Indoor temperature)

                if(during_the_control_period):

                    # Decide the action
                    at = AGENT.choose_action(St=data_t[..., 1:],
                                             Texp=Texp,
                                             TSPexpert=TSPexpert)

                    # Set the action into environment
                    Request.set_action(at[0, 0])

                    # record the future "St_1"
                    data_t_1 = Request.get_current_state()

                    print('[', int(Hr), ':', int(Min), ']:')
                    print('Current average temperature (VAV RAT):', np.mean(data_t_1[0, 1:6], -1))
                    print('Control Error:', np.mean(data_t_1[0, 1:6], -1) - Texp)
                    print('Set TSP to', at[0, 0])
                    print('Agent.ALPHA:',AGENT.alpha)

                    logger.info('Current average temperature (VAV RAT):' + str(np.mean(data_t_1[0, 1:6], -1)))
                    logger.info('Control Error:' + str(np.mean(data_t_1[0, 1:6], -1) - Texp))
                    logger.info('Set TSP to'+str(at[0, 0]))
                    logger.info('Agent.ALPHA:'+str(AGENT.alpha))
                    
                    # for recording the dataset
                    data_t = Request.get_current_state()
                    util.record_data(np.concatenate([np.array(now)[None], data_t], axis=-1), 'two_stages')
                    print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                    logger.info('Data recorded.')

                else:
                    Request.set_action(After_work_please_set_the_setpoint_back_to)
                    print('[', int(Hr), ':', int(Min), ']:Setpoint has back to ' + str(After_work_please_set_the_setpoint_back_to) + '.')
                    logger.info('Setpoint has back to ' + str(After_work_please_set_the_setpoint_back_to) + '.')
                    data_t_1 = Request.get_current_state()
                    data_t_1[...,0] = data_t_1[...,0] + offset_in_BA
                    
                    # for recording the dataset
                    data_t = Request.get_current_state()
                    data_t[...,0] = data_t[...,0] + offset_in_BA
                    util.record_data(np.concatenate([np.array(now)[None], data_t], axis=-1), 'two_stages')
                    print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                    logger.info('Data recorded.')
            
            elif(Hr >= Agent_start_to_work_at and Hr < Agent_stop_to_work_at):
                # for recording the dataset
                data_t = Request.get_current_state()
                util.record_data(np.concatenate([np.array(now)[None], data_t], axis=-1), 'two_stages')
                print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                logger.info('Data recorded.')
            
            else:
                # for recording the dataset
                data_t = Request.get_current_state()
                data_t[...,0] = data_t[...,0] + offset_in_BA
                util.record_data(np.concatenate([np.array(now)[None], data_t], axis=-1), 'two_stages')
                print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                logger.info('Data recorded.')

        AGENT.learn(iteration=10, show_message=False)
        Request.next_simulation_step()
