import torch
from torch import nn
import Normalizer as sc

class _encoder(nn.Module):
    def __init__(self, n, m, state, action):
        # state: index for input features represents state
        # action: similarly, but represents action
        super().__init__()
        assert m == len(state), f"m: {m} != len(state): {len(state)}"
        self.n = n
        self.m = m
        self.s_key = [key for key in list(range(n)) if key in state]
        self.a_key = [key for key in list(range(n)) if key not in state]
        self.state_encoder = nn.Sequential(
            nn.Linear(len(self.s_key), 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32,32),
            nn.LeakyReLU(0.2),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(len(self.a_key), 32),
            nn.LeakyReLU(0.2),
            nn.Linear(32,32),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self, x, concat=False):
        state, action = x[..., self.s_key], x[..., self.a_key]
        state, action = self.state_encoder(state), self.action_encoder(action)
        if concat:
            return torch.cat([state, action], dim=1)
        else:
            return state, action

class _predictor_core(nn.Module):
    def __init__(self):
        super().__init__()
        self.share_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
        )
        self.R_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
        )
        self.r_net = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.2),
        )
    def forward(self, x):
        tmp = self.share_net(x)
        tmp_R, tmp_r = self.R_net(tmp), self.r_net(tmp)
        return tmp_R, tmp_r

class Core(nn.Module):
    def __init__(self,
                 n=37,
                 m=5,
                 state_feature=list(range(1, 6)),
                 action_feature=list(range(1))):
        super().__init__()
        self.encoder = _encoder(n=n, m=m, state=state_feature, action=action_feature)
        self.predictor = _predictor_core()
        self.decoder = nn.Linear(32, m)
    
    def forward(self, x):
        abstract_state, abstract_action = self.encoder(x)

        # route 1, encoder -> decoder
        backcast = self.decoder(abstract_state)

        # route 2, encoder -> core -> R' -> decode
        #                          -> r' -> decoder    
        R_r_Delta, abstract_r = self.predictor(torch.cat([abstract_state, abstract_action], dim=1))
        abstract_R = R_r_Delta + abstract_r
        Rr_pred, r_pred = self.decoder(abstract_R), self.decoder(abstract_r)

        return Rr_pred, r_pred, backcast # R_pred represents delta


class MainModel(nn.Module):
    def __init__(self, a_dim=[0], s_dim=[1,2,3,4,5], e_dim=[i for i in range(6,37,1)]):
        super().__init__()
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.e_dim = e_dim
        self.Target_net = Core(action_feature=a_dim, state_feature=s_dim)
        self.Eval_net = Core(action_feature=a_dim, state_feature=s_dim)
        self.Normalizer = sc.Null_Normalizer()
        
        for param in self.Target_net.parameters():
            param.requires_grad = False
    @property
    def device(self):
        return next(iter(self.parameters())).device
    
    def Set_Normalizer(self, Normalizer):
        del(self.Normalizer)
        self.Normalizer = Normalizer

    def Predict(self, in_):
        LS_R, LS_r, _ = self.forward(in_=in_, name="Eval", return_Q=False)
        return LS_R, LS_r

    def forward(self, in_, name, return_Q=True):
        
        in_ = self.Normalizer.normalize(in_)
        A = in_[:,self.a_dim]
        S = in_[:,self.s_dim]
        E = in_[:,self.e_dim]
        A = torch.where(A < torch.mean(in_[:,self.s_dim],-1)[...,None], 
                        A, 
                        torch.mean(in_[:,self.s_dim],-1)[...,None] + A - A.detach())
        in_ = torch.cat([A, S, E], -1).float()
        
        if name == "Target":
            nRr, nr, nbackcast = self.Target_net(in_)
        elif name == "Eval":
            nRr, nr, nbackcast = self.Eval_net(in_)
        else:
            raise RuntimeError(f"Network name {name} not available")
        
        backcast = self.Normalizer.denormalize(nbackcast, self.s_dim)
        Rr = self.Normalizer.denormalize(nRr, self.s_dim)
        
        if return_Q == True:
            return Rr
        else:
            r = self.Normalizer.denormalize(nr, self.s_dim)
            R = Rr - r
            return R, r, backcast
    
    def Info(self):
        print(self)