import torch
import numpy as np

class Trainer():
    def __init__(self,Model,GARMMA,lr,batch_size=1000,update_target_per=100,global_step=0):
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
        self.GARMMA = GARMMA
        self.Model = Model
        for param in self.Model.Target_net.parameters():
            param.requires_grad = False
        self.optimizer = torch.optim.Adam(self.Model.Eval_net.parameters(), lr=lr)
        self.global_step = global_step
        self.update_target_per = update_target_per
        self.batch_size = batch_size

    def get_loss(self, data, return_err=False):
        s, a, s_t_plus_1, r = data
        s = s.to(self.Model.device).float().to(self.Model.device)
        a = a.to(self.Model.device).float().to(self.Model.device)
        s_t_plus_1 = s_t_plus_1.to(self.Model.device).float().to(self.Model.device)
        r = r.to(self.Model.device).float().to(self.Model.device)
        with torch.no_grad():
            y = (r*(1-self.GARMMA) + self.GARMMA *
                    self.Model(in_=torch.cat([a, s_t_plus_1], dim=-1), name="Target")).detach()

        input_x = torch.autograd.Variable(torch.cat([a, s], dim=-1), requires_grad=True)
        
        R_, r_, backcast = self.Model(in_=input_x, name="Eval", return_Q=False)
        
        grads_val = torch.autograd.grad(outputs=r_, inputs=input_x,
                                    grad_outputs=torch.ones_like(r_),
                                    create_graph=True, retain_graph=True, only_inputs=True,allow_unused=True)[0]

        gradient_constraint_loss = 100*torch.mean(torch.sum(torch.clamp(-self.relation_label.to(grads_val.device)[None]*grads_val, 0),-1)) # physical gradient constraint for r

        loss_r_before_mean = torch.sum(torch.square(r_ - r), dim=-1) #state forecasting loss
        loss_r = torch.mean(loss_r_before_mean) + gradient_constraint_loss

        tmp = -100*((s_t_plus_1[:, :len(self.Model.s_dim)]-s[:, :len(self.Model.s_dim)])*(R_))  
        R_constraint = torch.sum(torch.maximum(torch.zeros_like(tmp), tmp), dim=-1) # physical gradient constraint for R
        loss_R_TD_error = torch.sum(torch.square(R_ + r_.detach() - y), dim=-1) #long-term state residual changing forecasting loss (DQN-like TD error)
        loss_R = torch.mean(loss_R_TD_error +  R_constraint)

        loss_backcast = torch.mean(torch.square(backcast - s[..., :len(self.Model.s_dim)])) #recustruction loss

        loss = loss_r + loss_R + loss_backcast
        
        if(return_err):
            return loss, [r.detach(), r_.detach()]
        else:
            return loss
        
    def __call__(self, Memory):
        data_loader = Memory(self.batch_size)
        
        loss_list = []
        err_list = []
        for data in data_loader:
            loss, (r, r_) = self.get_loss(data, return_err=True)
        
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            self.global_step += 1
            if self.global_step % self.update_target_per == 0:
                self.Model.Target_net.load_state_dict(self.Model.Eval_net.state_dict())
                
            err = torch.mean(torch.sum(torch.abs(r_ - r), dim=-1)).item()
            
            loss_list.append(loss.item())
            err_list.append(err)
        
        return 'Loss: '+str(sum(loss_list)/len(loss_list))+'/Error: '+str(sum(err_list)/len(err_list))+' degree'
    
    def Info(self):
        print('Optimizer:', self.optimizer)
        print('Global_step:', self.global_step)
        print('Update_target_per:', self.update_target_per)
        print('Batch_size:', self.batch_size)