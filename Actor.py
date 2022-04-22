import torch
import numpy as np
import math

class Actor():
     def __init__(self,
                  BETA=0.5, 
                  SIGMA=0.9, 
                  ALPHA = 0.5, 
                  use_confident = True, 
                  valid_actions = np.array([i/10 for i in range(185,271,5)]), 
                  solution = 'modify_coefficients'):
         
         self.BETA = BETA
         self.SIGMA = SIGMA
         self.ALPHA = ALPHA
         self.use_confident = use_confident
         self.valid_actions = valid_actions
         self.solution = solution
     
     def update_confidence(self, Trainer, Memory):
         if(self.use_confident):
             loss_value = Trainer.get_loss(Memory.get_last_n(n=1))
             loss_value = loss_value.item()
             self.ALPHA = self.BETA * math.exp(-loss_value/self.SIGMA) + (1-self.BETA)*self.ALPHA
         else:
             self.ALPHA = 1.0
     
     def Choose_Action_Base_on_Model(self, Model, Current_State, Target):
         #Current_State: Numpy array with shape [N,]
         #Target: Float value representing the target temperature wanted by the users.
         
         
         action_set = torch.from_numpy(self.valid_actions.astype(np.float32))
         Texp = Target*torch.ones([1])

         if(len(Current_State.shape)==1):
             Current_State = Current_State[None]
         Current_S = Current_State[...,Model.s_dim-1]
         Current_E = Current_State[...,Model.e_dim-1]
         
         St = torch.from_numpy(Current_S.astype(np.float32))
         Et = torch.from_numpy(Current_E.astype(np.float32))
         SEt = torch.cat([St, Et],-1)
         
         expanded_data = SEt[:, None].repeat(1, action_set.shape[0], 1)
         expanded_data = expanded_data.reshape([-1, SEt.shape[-1]], 1)
         
         expanded_action_set = action_set[None].repeat(SEt.shape[0], 1)
         expanded_action_set = action_set.reshape([-1, 1])

         ASEt = torch.cat([expanded_action_set, expanded_data], dim=-1)
         LS_R, LS_r = Model.Predict(ASEt.to(Model.device))
         LS_R, LS_r = LS_R.cpu(), LS_r.cpu()
         Tt = St
         
         if self.solution == 'modify_coefficients':
             Tt_plus1_given_a = LS_r + LS_R
             Tt_plus1_given_a = torch.reshape(Tt_plus1_given_a, [St.shape[0], action_set.shape[0], St.shape[-1]]) #long term tempterature

             current_Err = torch.abs(torch.mean(Tt, dim=-1) - torch.mean(Texp, dim=-1))[...,None]
             Err = torch.abs(torch.mean(Tt_plus1_given_a, dim=-1)[...,None] - torch.unsqueeze(Texp, dim=-2))
             prediction_of_future_temperature_changing_amount = torch.abs(torch.mean(Tt_plus1_given_a, dim=-1)[...,None] - torch.mean(Tt, dim=-1)[...,None])
            
             #Equally, it is score = |Tt_p1 - Texp| - alpha*|Tt - Texp| - beta*|Tt_p1 - Tt|.
             #Choose alpha=1.1 and beta=0.9 here.
             #Choose the action that has less impact to the environment will having the biggest advantage of error.
             score = torch.mean(current_Err - 1.1*Err - 0.9*prediction_of_future_temperature_changing_amount, dim=-1)
     
             #Ban the candidate which lager then current average indoor temperature, 
             #since to set the temperature set point higher then current temperature means to turn off the HVAC.
             action_set_idx = torch.where(action_set > torch.mean(Tt, dim=-1))
             if (len(action_set_idx[0]) > 0):
                 min_action_set = torch.min(action_set[action_set_idx])
                 score = torch.where((action_set[..., 0] <= min_action_set), score, torch.ones_like(score) * -1000.)
     
             #Finally, Find the candidate with highest score value.
             idx = torch.argmax(score, dim=-1)
             #Select a_model.
             TSPmodel = action_set[idx]
         else:
             raise NotImplementedError("Not a supported method!")
             
         return TSPmodel
     
     def __call__(self, Model, Current_State, Target, TSPexpert):
         
         TSP = self.Choose_Action_Base_on_Model(Model, Current_State, Target)        
         TSP = TSP.numpy()
         
         if(self.use_confident):
             # Mix a_model with a_expert according to the confident alpha.
             TSP = self.ALPHA*TSP + (1-self.ALPHA)*TSPexpert
             
             # Quantize to the valid setpoints.
             Qidx = np.argmax(-np.abs(TSP[:,None] - self.valid_actions[None]),-1) #argmin(abs dist.)
             TSP = self.valid_actions[Qidx][:,None]
         else:
            TSP = TSP[:,None]
             
         return TSP
     
     def Info(self):
         print('Strategy:', self.solution)
         print('Use Confident:', self.use_confident)
         print('Current Confident: ', self.ALPHA)
         print('BETA: ', self.BETA)


