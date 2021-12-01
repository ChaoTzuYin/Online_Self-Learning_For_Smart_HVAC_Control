import torch
from torch import nn
import numpy as np
import random
import pandas as pd
import scaler as sc
import math
from itertools import chain
from torch.utils.tensorboard import SummaryWriter

np.random.seed(1)
torch.manual_seed(1)  # cpu
torch.cuda.manual_seed_all(1)  # gpu
GPU_USAGE = torch.cuda.is_available()  # set to False to disable GPU

if not GPU_USAGE:
    print("warning: cuda not available")


class MEMORY(nn.Module):
    def __init__(self, length_list):
        super().__init__()
        self.ls = np.concatenate([np.array([0.]), np.cumsum(
            length_list, axis=-1)], axis=-1).astype(int)
        self.dict = np.zeros([0, self.ls[-1]])

    def record(self, args):
        if len(args) != len(self.ls)-1:
            raise ValueError(
                'The input data number is not equal to group number.')

        self.dict = np.concatenate(
            [self.dict, np.concatenate(args, axis=-1)], axis=0)
    
    def get_last_n(self, n=5):
        return self.dict[-5:]


    def get_batch(self, idx):
        get = self.dict[idx]
        ret = [get[..., self.ls[count]:self.ls[count+1]]
               for count in range(self.ls.shape[0]-1)]
        return ret

    def delete_first(self):
        self.dict = np.delete(self.dict, 0, axis=0)

    def length(self):
        return self.dict.shape[0]


class TD_Simulation_Agent(nn.Module):
    def __init__(self,
                 nstep=1,
                 EPSILON=0.8,
                 GARMMA=0.05,
                 BETA=0.5,
                 SIGMA=0.9,
                 update_target_per=100,
                 batch_size=1000,
                 global_step=0,
                 load_weight=False,
                 initial_datapath='./data_logs/bias_data.csv',
                 solution='modify_coefficients'):
        super().__init__()
        data_T = pd.read_csv(initial_datapath, header=None).values[:240*3]

        self.solution=solution 
        #Either 'modify_coefficients' or 'two_stages'. 
        #Both of the two solutions can solve the peak problem.
        #See function Actor for more detial.
        
        initial_st = np.concatenate([data_T[:-nstep, 1:]
                                     for _ in range(14)], axis=0)
        initial_at = np.concatenate([data_T[:-nstep, :1]
                                     for _ in range(14)], axis=0)
        initial_st_plus_1 = np.concatenate(
            [data_T[nstep:, 1:] for _ in range(14)], axis=0)
        initial_r = np.concatenate([data_T[nstep:, 1:6]
                                    for _ in range(14)], axis=0)

        self.relation_label = torch.from_numpy(np.array([ 1, #AHU TSP
                                             1, 1, 1, 1, 1, #VAV RAT
                                            -1,-1,-1,-1,-1, #VAV ZI
                                             1, 1, 1, 1, 1, #FCU RAT
                                            -1,-1,-1,-1,-1, #FCU CV
                                            -1,-1,-1,-1,-1, #FCU CMD
                                             1, 1, 1, 1, 1, #VAV CMD
                                            -1, #AHU PS
                                             1, #MAU TS
                                             1, #OAT
                                            -1, #RS
                                             1,1  #load
                                            ])).float()

        self.state_normalizer = sc.scaler(group_index=[[0, 1, 2, 3, 4, 10, 11, 12, 13, 14, 31, 32],
                                                       [5, 6, 7, 8, 9, 
                                                        15, 16, 17, 18, 19],
                                                       [30],
                                                       [34, 35]],
                                          method=[sc.stander_normalize(),
                                                  sc.maximum_normalize(),
                                                  sc.stander_normalize(),
                                                  sc.stander_normalize()],
                                          name=['t', 'level', 'p', 'kw'])
        
        self.state_normalizer.fit(data_T[..., 1:])
        self.action_normalizer = self.state_normalizer.scaler_group['t']

        # Initial the settings
        self.BETA = BETA
        self.SIGMA = SIGMA
        self.GARMMA = GARMMA
        self.alpha = 0.5
        self.EPSILON = EPSILON
        self.actions = np.array([20.5, 21, 21.5, 22, 22.5, 23, 23.5, 24, 24.5, 25, 25.5, 26, 26.5, 27])
        self.global_step = global_step
        self.update_target_per = update_target_per
        self.batch_size = batch_size
        
        # Define the simulation network
        self.Eval_share = nn.Sequential(
            nn.Linear(37, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 48),
            nn.LeakyReLU(0.2),
            nn.Linear(48, 32),
            nn.LeakyReLU(0.2)
        )
        self.Eval_r = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 5),
        )
        self.Eval_R = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 5),
        )

        self.Target_share = nn.Sequential(
            nn.Linear(37, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 48),
            nn.LeakyReLU(0.2),
            nn.Linear(48, 32),
            nn.LeakyReLU(0.2)
        )
        self.Target_r = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 5),
        )
        self.Target_R = nn.Sequential(
            nn.Linear(32, 16),
            nn.LeakyReLU(0.2),
            nn.Linear(16, 5),
        )

        if GPU_USAGE:
            self.Eval_share = self.Eval_share.cuda()
            self.Eval_r = self.Eval_r.cuda()
            self.Eval_R = self.Eval_R.cuda()

            self.Target_share = self.Target_share.cuda()
            self.Target_r = self.Target_r.cuda()
            self.Target_R = self.Target_R.cuda()

        for param in chain(self.Target_share.parameters(), self.Target_r.parameters(), self.Target_R.parameters()):
            param.requires_grad = False

        # Define the optimizer
        self.optimizer = torch.optim.Adam(
            chain(self.Eval_share.parameters(),
                  self.Eval_r.parameters(),
                  self.Eval_R.parameters()), lr=1e-3)

        # Initial the memory
        self.CAPACITY = initial_st.shape[0]
        self.MEMORY = MEMORY([36, 1, 36, 5])
        self.MEMORY.dict = np.concatenate(
            [initial_st, initial_at, initial_st_plus_1, initial_r], axis=-1)

        

    def TD_Simulator(self, in_, name, return_Q=True):

        A = in_[:,:1]
        S = in_[:,1:6]
        E = in_[:,6:]
        A = torch.where(A < torch.mean(in_[:,1:6],-1)[...,None], 
                        A, 
                        torch.mean(in_[:,1:6],-1)[...,None] + A - A.detach())

        if name == "Eval":
            net_share = self.Eval_share
            net_r = self.Eval_r
            net_R = self.Eval_R
        elif name == "Target":
            net_share = self.Target_share
            net_r = self.Target_r
            net_R = self.Target_R
        else:
            raise RuntimeError("Name %s not seen before" % (name))

        L2 = net_share(torch.cat([A,S,E],-1).float())
        R = net_R(L2)
        r = net_r(L2)

        if return_Q == True:
            return R + r
        else:
            return R, r

    def Actor(self, in_, action_set, Texp, TSPexpert, alpha, debug=False):
        expanded_data = torch.repeat_interleave(
            in_, action_set.shape[0], dim=0)
        expanded_action_set = action_set.repeat(in_.shape[0], 1)
        Ast = torch.cat([expanded_action_set, expanded_data], dim=-1)
        LS_R, LS_r = self.TD_Simulator(in_=Ast, name="Eval", return_Q=False)
        Tt = in_[..., :5]

        current_Err = torch.abs(torch.mean(Tt, dim=-1) - 
                                torch.mean(Texp, dim=-1))[...,None]

        Tt_plus1_given_a = torch.reshape(LS_r + LS_R, [in_.shape[0], action_set.shape[0], 5]) #long term tempterature

        Err = torch.abs(torch.mean(Tt_plus1_given_a, dim=-1)[...,None] - torch.unsqueeze(Texp, dim=-2))
        
        prediction_of_future_temperature_changing_amount = torch.abs(torch.mean(Tt_plus1_given_a, dim=-1)[...,None] - 
                                                                     torch.mean(Tt, dim=-1)[...,None])
        
        if(self.solution=='modify_coefficients'):
            #Equally, it is score = |Tt_p1 - Texp| - alpha*|Tt - Texp| - beta*|Tt_p1 - Tt|.
            #Choose alpha=1.1 and beta=0.9 here.
            #Choose the action that has less impact to the environment will having the biggest advantage of error.
            score = torch.mean(current_Err - 1.1*Err - 0.9*prediction_of_future_temperature_changing_amount, dim=-1)
    
            #Ban the candidate which lager then current average indoor temperature, 
            #since to set the temperature set point higher then current temperature means to turn off the HVAC.
            score = torch.where((action_set[..., 0] <= torch.unsqueeze(
                    torch.mean(Tt, dim=-1), dim=-1)), score, torch.ones_like(score) * -1000.)
    
            #Finally, Find the candidate with highest score value.
            idx = torch.argmax(score, dim=-1)
    
            #Select a_model.
            TSPmodel = action_set[idx]
        
        elif(self.solution=='two_stages'):
            
            long_term_converge_tendency = torch.reshape(LS_R, [in_.shape[0], action_set.shape[0], 5]) 
            longterm_delta_T = torch.mean(torch.abs(long_term_converge_tendency),-1) #B, 14
            #Long term expected tempterature converge tendency
            #To be clear, it predict how will the temperature keep changing after 18 minute util it converge suppose that we keep this setting for a long time.
            #The smaller this value is, the more stable temperature the model expected to have after the corresponding setting.

            Tt_plus1_given_a = torch.reshape(LS_r, [in_.shape[0], action_set.shape[0], 5]) 
            #short term tempterature
            #To be clear, it predict the indoor temperature after 18 minute under this setting after 18 minute (same with the output of naive version). 
            Err = torch.abs(torch.mean(Tt_plus1_given_a, dim=-1) - Texp) #B, 14 - B, 1 = B, 14
            #Expected error after setting the corresponding actions ([20.5, 21, ...., 27], totally 14 candidates.)
            
            #Ban the candidate which lager then current average indoor temperature, 
            #since to set the temperature set point higher then current temperature means to turn off the HVAC.
            Err = torch.where((action_set[..., 0] <= torch.unsqueeze(
                  torch.mean(Tt, dim=-1), dim=-1)), Err, torch.ones_like(Err) * 1000.)
            
            ###stage 1: choose the candidates with top 3 smallest error.###
            top_3_min_error_idx_dimension0 = torch.range(0, in_.shape[0]-1)[...,None].repeat(1, 3)
            top_3_min_error_idx_dimension1 = torch.argsort(Err,-1)[...,:3].long() #B, 3
            top_3_corresponding_delta_T = longterm_delta_T[(top_3_min_error_idx_dimension0.long(),
                                                            top_3_min_error_idx_dimension1.long())] #B, 3
            
            ###stage 2: choose the most stable candidates out of the top 3 smallest error candidates, respected to longterm_delta_T.###
            top1_min_delta_T_idx_dimension0 = torch.range(0, in_.shape[0]-1) #B
            top1_min_delta_T_idx_dimension1 = top_3_corresponding_delta_T.argmin(-1) #B
            final_selection = top_3_min_error_idx_dimension1[(top1_min_delta_T_idx_dimension0.long(),
                                                              top1_min_delta_T_idx_dimension1.long())] #B
        
            #Select a_model.
            TSPmodel = action_set[final_selection]
        
        else:
            raise NotImplementedError("Not a supported method!")
            
        #Mix a_model with a_expert according to the confident alpha.
        TSPmix = alpha*TSPmodel + (1-alpha)*TSPexpert

        if GPU_USAGE:
            TSPmix = TSPmix.cpu()
        TSPmix = TSPmix.numpy()
        if debug == True:
            return TSPmix, TSPmodel, score
        else:
            return TSPmix


    def gen_data_v2(self, in_):
        in_t = np.copy(in_)[:-1]
        in_t_plus_5 = np.copy(in_)[1:]
        extend_list = [np.copy(in_t_plus_5) for _ in range(205, 270, 5)]
        for item, t in zip(extend_list, range(205, 270, 5)):
            tsp = t/10
            item[:, 0] = tsp
            Tsp_list = in_t[:, 0]
            bigger_index = np.where(Tsp_list > tsp)
            item[bigger_index, 1:6] = item[bigger_index, 1:6] - \
                0.6*np.abs(in_t[bigger_index, 0:1] - tsp)
            smaller_index = np.where(Tsp_list < tsp)
            item[smaller_index, 1:6] = item[smaller_index, 1:6] + \
                0.4*np.abs(in_t[smaller_index, 0:1] - tsp)
        return np.concatenate(extend_list, axis=0)

    def save(self, path):
        raise NotImplementedError("saving not supported!")

    def record(self, s, a, s_t_plus_1, r):
        self.MEMORY.record([s, a, s_t_plus_1, r])
        
        # update confident
        self.update_alpha(nearest_n=1)
        
        if self.CAPACITY != None:
            if (self.MEMORY.length() > self.CAPACITY):
                self.MEMORY.delete_first()

    def choose_action(self, St, Texp, TSPexpert):

        # User's configulations
        s = self.state_normalizer.normalize(St)
        Texp = np.ones_like(St[:, 0:1])*self.action_normalizer.normalize(Texp)
        TSPexpert = np.ones_like(St[:, 0:1])*self.action_normalizer.normalize(TSPexpert)

        # Prepare normalizers
        T_std = self.state_normalizer.scaler_group['t'].std
        T_mean = self.state_normalizer.scaler_group['t'].mean

        # Convert np to tensors
        action_set = np.expand_dims(self.action_normalizer.normalize(self.actions), axis=-1)
        action_set = torch.tensor(action_set, requires_grad=False)
        s = torch.tensor(s, requires_grad=False)
        Texp = torch.tensor(Texp, requires_grad=False)
        TSPexpert = torch.tensor(TSPexpert, requires_grad=False)

        # Device deployment
        if GPU_USAGE:
            action_set = action_set.cuda()
            s = s.cuda()
            Texp = Texp.cuda()
            TSPexpert = TSPexpert.cuda()

        # Choose the action
        normalized_action = self.Actor(in_=s, action_set=action_set, Texp=Texp, TSPexpert=TSPexpert, alpha=self.alpha)
        action = normalized_action * T_std + T_mean #De-normalize
        
        #Quantize the action
        Q_idx = np.argmin(np.abs(action - np.tile(np.expand_dims(self.actions, axis=0), [action.shape[0], 1])), axis=-1)
        Q_action = np.expand_dims(self.actions[Q_idx], axis=-1)
        
        return Q_action

    def get_batch(self):
        idx = [i for i in range(self.MEMORY.length())]
        random.shuffle(idx)
        return self.MEMORY.get_batch(idx[:self.batch_size])

    def update_alpha(self, nearest_n=5):
        
        # Evaluate the quality of current model to adjust the weight for mix-up control.
        
        # 1. Get the latest samples.
        testing_data = self.MEMORY.get_batch([i for i in range(-nearest_n,0,1)])
        
        # 2. Process the nessary parameters.
        s = torch.from_numpy(self.state_normalizer.normalize(testing_data[0]))
        a = torch.from_numpy(self.action_normalizer.normalize(testing_data[1]))
        s_t_plus_1 = torch.from_numpy(self.state_normalizer.normalize(testing_data[2]))
        r = torch.from_numpy(self.action_normalizer.normalize(testing_data[3]))
        
        if GPU_USAGE:
            s = s.cuda()
            a = a.cuda()
            s_t_plus_1 = s_t_plus_1.cuda()
            r = r.cuda()

        # 3. Load the configs for updating.
        BETA = self.BETA
        SIGMA = self.SIGMA
        GARMMA = self.GARMMA
        alpha = self.alpha

        # 4. Evaluate on those samples.
        with torch.no_grad():
            y = (r*(1-GARMMA) + GARMMA *
                 self.TD_Simulator(in_=torch.cat([a, s_t_plus_1], dim=-1), name="Target")).detach()

        
        input_x = torch.autograd.Variable(torch.cat([a, s], dim=-1),requires_grad=True)
        R_, r_ = self.TD_Simulator(in_=input_x, name="Eval", return_Q=False)

        loss_r_before_mean = torch.sum(torch.square(r_ - r), dim=-1)
        loss_r = torch.mean(loss_r_before_mean)

        tmp = -100*((s_t_plus_1[:, :5]-s[:, :5])*(R_))  # cache
        R_constraint = torch.sum(torch.maximum(
            torch.zeros_like(tmp), tmp), dim=-1)
        loss_R_before_mean = torch.sum(torch.square(
            R_ + r_.detach() - y), dim=-1) + R_constraint
        loss_R = torch.mean(loss_R_before_mean)


        td_loss = float(loss_r + loss_R)
        
        # 5. Update the mixup weight alpha respected to the validation loss.
        self.alpha = float(
            BETA * math.exp(-td_loss/SIGMA) + (1-BETA)*alpha)
        

    def learn(self, iteration=200, show_message=True):
        ll = []
        el = []
        gc = []
        for _ in range(iteration):
            data = self.get_batch()
            s = torch.from_numpy(self.state_normalizer.normalize(data[0]))
            a = torch.from_numpy(self.action_normalizer.normalize(data[1]))
            s_t_plus_1 = torch.from_numpy(
                self.state_normalizer.normalize(data[2]))
            r = torch.from_numpy(self.action_normalizer.normalize(data[3]))
            if GPU_USAGE:
                s = s.cuda()
                a = a.cuda()
                s_t_plus_1 = s_t_plus_1.cuda()
                r = r.cuda()

            GARMMA = self.GARMMA

            with torch.no_grad():
                y = (r*(1-GARMMA) + GARMMA *
                     self.TD_Simulator(in_=torch.cat([a, s_t_plus_1], dim=-1), name="Target")).detach()


            input_x = torch.autograd.Variable(torch.cat([a, s], dim=-1), requires_grad=True)
            R_, r_ = self.TD_Simulator(in_=input_x, name="Eval", return_Q=False)
            grads_val = torch.autograd.grad(outputs=r_, inputs=input_x,
                                        grad_outputs=torch.ones_like(r_),
                                        create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]

            gradient_constraint_loss = 100*torch.mean(torch.sum(torch.clamp(-self.relation_label.to(grads_val.device)[None]*grads_val, 0),-1))


            loss_r_before_mean = torch.sum(torch.square(r_ - r), dim=-1)
            loss_r = torch.mean(loss_r_before_mean) + gradient_constraint_loss

            tmp = -100*((s_t_plus_1[:, :5]-s[:, :5])*(R_))  # cache
            R_constraint = torch.sum(torch.maximum(
                torch.zeros_like(tmp), tmp), dim=-1)
            loss_R_before_mean = torch.sum(torch.square(
                R_ + r_.detach() - y), dim=-1) + R_constraint
            loss_R = torch.mean(loss_R_before_mean)

            td_loss = float(loss_r + loss_R)

            T_std = self.state_normalizer.scaler_group['t'].std
            err = float(torch.mean(torch.sum(torch.abs(r_ - r), dim=-1)))
            
            loss = loss_r + loss_R
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.global_step += 1
            ll.append(td_loss)
            el.append(err)
            gc.append(gradient_constraint_loss.item())

            if self.global_step % self.update_target_per == 0:
                self.Target_share.load_state_dict(self.Eval_share.state_dict())
                self.Target_r.load_state_dict(self.Eval_r.state_dict())
                self.Target_R.load_state_dict(self.Eval_R.state_dict())
        if show_message:
            print('Global step:', self.global_step, ' loss:', sum(ll) /
                  len(ll), 'err:', sum(el)/len(el), 'gradient constraint:', sum(gc)/len(gc), 'alpha update:', self.alpha)
        return self.alpha
