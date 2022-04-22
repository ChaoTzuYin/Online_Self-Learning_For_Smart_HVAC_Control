import logging
import OSLN_framework
import numpy as np
from fakeRequest import fakeRequest
import util
import torch
import os

############ Config ############

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Agent_initial_data = './data_logs/15A_dataset_from_real_env/bias_dataset_15A_AHU301.csv'
Agent_save_path = './agent_weight/experiment_AHU301.pth'
Env_model_for_fakeRequest = "./environment_model_weight/2021_AHU301_envmodel.pth"
Simulation_data_for_fakeRequest = './data_logs/15A_dataset_from_real_env/clean_dataset_15A_AHU301_whole.csv'
Record_path = './experimental_record/2021_301_no_reduce_use_fake_dataset.csv'

if(not os.path.isdir('./experimental_record')):
    os.mkdir('./experimental_record')
    
if(not os.path.isdir('./agent_weight')):
    os.mkdir('./agent_weight')


Save_checkpoint_per_N_steps = 120 # Save the model for every N time steps.

Timesetp_perminute = 6  # Recording sample rate: 6 minute
Change_setting_per_N_Timestep = 3  # this means 6*3=18 minute

Agent_start_to_work_at = 7  # when will the agent start to work
Agent_stop_to_work_at = 19  # when will the agent stop to work
# what's the temperature should be after the agent stop
After_work_please_set_the_setpoint_back_to = 21
# what's the offset in BA (This will affect the learning result if it's not correct)
offset_in_BA = 4

Texp = 25.25      # The expected temperature we hope to have in the office
# The reference temperature setpoint (which usually set before the AI is on)
TSPexpert = 22.7

############ Initialize and pretrain the model ############

AGENT = OSLN_framework.TD_Simulation_Agent(nstep=3,  # 18 minutes
                                          GARMMA=0.25, # Hyperparameter for adjusting the sensitivity to loss during updating confident value in Actor
                                          BETA=0.9, # Hyperparameter for controling the moving average during updating confident value in Actor
                                          SIGMA=0.5, # Hyperparameter for the longterm tendency loss.
                                          update_target_per=100, # Hyperparameter for the updating target network to evaluate network.
                                          batch_size=1000, 
                                          initial_datapath=Agent_initial_data, # File of dataset .
                                          solution='modify_coefficients', # Score function selection for choosing the action.
                                          lr=1e-3).to(device)

for i in range(200):
    print('confident:', AGENT.Learn(1, True))
torch.save(AGENT,Agent_save_path)

############################logger#############################

logging.basicConfig(
    filename='./program.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

################################################################
data_t_1 = None
Goin_time = None
################################################################


def get_flags(Hr, Min):

    current_time_step = (Hr * 60 + Min - Agent_start_to_work_at * 60)

    ##per time step##
    time_to_record = (
        ((Hr * 60 + Min - Agent_start_to_work_at * 60) % Timesetp_perminute) == 0)

    ##saving time##
    time_to_save_model = (((Hr * 60 + Min - Agent_start_to_work_at * 60) %
                          (Timesetp_perminute*Save_checkpoint_per_N_steps)) == 0)

    ##it should be in the range of control time period##
    Over_working_time = ((Hr * 60 + Min - Agent_stop_to_work_at * 60) >= 0)
    during_the_control_period = (
        (current_time_step >= 0) and not Over_working_time)

    ##it should be the right time step##
    in_the_right_time_step = ((current_time_step % (
        Timesetp_perminute * Change_setting_per_N_Timestep)) == 0)

    return time_to_record, time_to_save_model, during_the_control_period, in_the_right_time_step


if __name__ == "__main__":
    Request = fakeRequest(File_path = Simulation_data_for_fakeRequest,
                          Environment_path = Env_model_for_fakeRequest,
                          device = device)
    at = np.array([[TSPexpert]])
    logger = logging.getLogger(__name__)

    AGENT.Actor.ALPHA = 0.5

    for simulation_steps in range(Request.max_steps):

        if(simulation_steps % 240 == 0):
            print('Day ', int(simulation_steps/240),
                  ' / ', Request.max_steps, ' starts.')

        now = Request.get_time()
        Hr = now[-2]
        Min = now[-1]

        time_to_record, time_to_save_model, during_the_control_period, in_the_right_time_step = get_flags(
            Hr, Min)

        if(Min != Goin_time):
            Goin_time = None

        if(time_to_record and Min != Goin_time):
            Goin_time = Min

            if(time_to_save_model):
                # util.save_latest_model(AGENT, './model_weights_simulation/OSLN_plus_2018_old_bias_full_simulation_2.pth')
                print('[', int(Hr), ':', int(Min),
                      ']: A new model checkpoint has been created.')
                logger.info('A new model checkpoint has been created.')

            if(in_the_right_time_step):

                data_t = Request.get_current_state()

                if(data_t_1 is not None):
                    data_package = [data_t_1[..., 1:],  # St_1
                                     data_t_1[..., :1],  # At_1
                                     data_t[..., 1:],  # St
                                     data_t[..., 1:6]]
                    AGENT.Record(data_package)  # Ground Truth (Indoor temperature)

                if(during_the_control_period):

                    # Decide the action
                    at = AGENT.Choose_Action(Current_State=data_t[..., 1:],
                                             Target=Texp,
                                             TSPexpert=TSPexpert)

                    # Set the action into environment
                    Request.set_action(at[0, 0])

                    # record the future "St_1"
                    data_t_1 = Request.get_current_state()

                    print('[', int(Hr), ':', int(Min), ']:')
                    print('Current average temperature (VAV RAT):',
                          np.mean(data_t_1[0, 1:6], -1))
                    print('Control Error:', np.mean(
                        data_t_1[0, 1:6], -1) - Texp)
                    print('Set TSP to', at[0, 0])
                    print('Confident:', AGENT.Actor.ALPHA)

                    logger.info('Current average temperature (VAV RAT):' +
                                str(np.mean(data_t_1[0, 1:6], -1)))
                    logger.info('Control Error:' +
                                str(np.mean(data_t_1[0, 1:6], -1) - Texp))
                    logger.info('Set TSP to'+str(at[0, 0]))
                    logger.info('Confident:'+str(AGENT.Actor.ALPHA))

                    # for recording the dataset
                    data_t = Request.get_current_state()
                    util.record_data(np.concatenate(
                        [np.array(now)[None], data_t, np.array(np.mean(data_t_1[0, 1:6], -1) - Texp)[None][None], np.array(AGENT.Actor.ALPHA)[None][None]], axis=-1), filepath=Record_path)
                    print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                    logger.info('Data recorded.')

                else:
                    Request.set_action(
                        After_work_please_set_the_setpoint_back_to)
                    print('[', int(Hr), ':', int(Min), ']:Setpoint has back to ' +
                          str(After_work_please_set_the_setpoint_back_to) + '.')
                    logger.info(
                        'Setpoint has back to ' + str(After_work_please_set_the_setpoint_back_to) + '.')
                    data_t_1 = Request.get_current_state()
                    data_t_1[..., 0] = data_t_1[..., 0] + offset_in_BA

                    # for recording the dataset
                    data_t = Request.get_current_state()
                    data_t[..., 0] = data_t[..., 0] + offset_in_BA
                    util.record_data(np.concatenate(
                        [np.array(now)[None], data_t, np.array(0)[None][None], np.array(0)[None][None]], axis=-1), filepath=Record_path)
                    print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                    logger.info('Data recorded.')

            elif(Hr >= Agent_start_to_work_at and Hr < Agent_stop_to_work_at):
                # for recording the dataset
                data_t = Request.get_current_state()
                util.record_data(np.concatenate(
                    [np.array(now)[None], data_t, np.array(0)[None][None], np.array(0)[None][None]], axis=-1), filepath=Record_path)
                print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                logger.info('Data recorded.')

            else:
                # for recording the dataset
                data_t = Request.get_current_state()
                data_t[..., 0] = data_t[..., 0] + offset_in_BA
                util.record_data(np.concatenate(
                    [np.array(now)[None], data_t, np.array(0)[None][None], np.array(0)[None][None]], axis=-1), filepath=Record_path)
                print('[', int(Hr), ':', int(Min), ']: Data recorded.')
                logger.info('Data recorded.')

        AGENT.Learn(10, False)
        Request.next_simulation_step()
