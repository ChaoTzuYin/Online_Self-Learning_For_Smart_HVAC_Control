#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 07:07:40 2021

@author: user
"""
import torch
import torch.nn as nn
from TDVAE.model.tdvae_model import tdvae

class TDVAE_Core(nn.Module):
    def __init__(self, distribution_dim=16, num_layer_block=2):
        super(TDVAE_Core, self).__init__()
        self.state_abstraction = tdvae(S_dim=32,
                                       A_dim=32,
                                     belief_state_dim=64, 
                                     MLP_hidden_dim=64,
                                     distribution_dim=16,
                                     num_layer_block=2,
                                     condition_dim=32,
                                     backbone_stack_layer=1)
        
        self.out_linear = nn.Sequential(nn.Linear(distribution_dim*num_layer_block, 32),
                                        nn.LeakyReLU(0.2))
    def forward(self, X):
        _, z_list, _, tdvae_loss_list = self.state_abstraction(X[...,32:].permute(0,2,1), X[...,:32].permute(0,2,1))
        xt_p1_pred = self.out_linear(torch.cat(z_list,1).permute(0,2,1))
        if(self.training):
            return xt_p1_pred, tdvae_loss_list
        else:
            return xt_p1_pred    


class Discriminator(torch.nn.Module):
    def __init__(self,mode='div'):
        super().__init__()
        self.mode = mode
        self.main = nn.Sequential(nn.Linear(64, 256),
								  nn.LeakyReLU(),
								  nn.Linear(256, 512),
								  nn.LeakyReLU(),
								  nn.Linear(512, 1))
        self.d_optimizer = torch.optim.Adam(params=self.parameters(),
                                         lr=5e-4,
                                         betas=(0.5, 0.999))
            
    def forward(self, x):
        return self.main(x)

    def compute_gradient(self, x):
        inp_vec = torch.autograd.Variable(x,
                                          requires_grad=True) # B,(HW+L),C
        out = self.main(inp_vec)
        out_gradient = torch.autograd.grad(
            outputs=out,
            inputs=inp_vec,
            grad_outputs=torch.ones_like(out),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return out_gradient

    def fit(self, real_vectors, fake_vectors):
        # Set discriminator gradients to zero.
        self.zero_grad()
        
        b_length = min(real_vectors.shape[0],fake_vectors.shape[0])
        
        real_vectors = torch.autograd.Variable(real_vectors.data[:b_length],requires_grad=True)
        fake_vectors = torch.autograd.Variable(fake_vectors.data[:b_length],requires_grad=True)
               
        # Train with real
        real_output = self.forward(real_vectors)
        errD_real = torch.mean(real_output)
        D_x = real_output.mean().item()

        # Train with fake
        fake_output = self.forward(fake_vectors)
        errD_fake = -torch.mean(fake_output)
        D_G_z1 = -fake_output.mean().item()

        # Calculate W-div gradient penalty
        gradient_penalty = self.calculate_gradient_penalty(real_vectors, fake_vectors, 2, 6)

        # Add the gradients from the all-real and all-fake batches
        errD = errD_real + errD_fake + gradient_penalty
        errD.backward()
        # Update D
        self.d_optimizer.step()        
        
        return D_x, D_G_z1
    
    def D_loss(self, x):
        errG = torch.mean(self.forward(x))
        return errG

    def calculate_gradient_penalty(self, real_data, fake_data, k=2, p=6, l=10):
        alpha = torch.rand([real_data.shape[0]]+[1]*(len(real_data.shape)-1)).to(real_data.device)
        mixup_vectors = fake_data*alpha + real_data*(1-alpha)
        noise_gradient = self.compute_gradient(mixup_vectors)
        if self.mode=='div':
            noise_gradient_gradient_norm = noise_gradient.view(noise_gradient.size(0), -1).pow(2).sum(1) ** (p / 2)
            gradient_penalty = torch.mean(noise_gradient_gradient_norm) * k / 2
        else:
            noise_gradient_gradient_norm = noise_gradient.view(noise_gradient.size(0), -1).pow(2).sum(1) ** (1 / 2)
            gradient_penalty = torch.mean((noise_gradient_gradient_norm-1)**2) * l
        return gradient_penalty

class Encoder(nn.Module):
    def __init__(self, a_dim, s_dim):
        super().__init__()       
        self.a_net = nn.Sequential(nn.Linear(a_dim, 32),
                                   nn.LeakyReLU(0.2))
        self.s_net = nn.Sequential(nn.Linear(s_dim, 32),
                                   nn.LeakyReLU(0.2))
    def forward(self, A, S):
        return torch.cat([self.a_net(A), self.s_net(S)],-1)

class Decoder(nn.Module):
    def __init__(self, out_dim):
        super().__init__()       
        self.net = nn.Sequential(nn.Linear(32, out_dim))
    def forward(self, X):
        return self.net(X)

class Core(nn.Module):
    def __init__(self):
        super().__init__()       
        self.net = nn.Sequential(nn.Linear(64, 64),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(64, 64),
                                 nn.LeakyReLU(0.2),
                                 nn.Linear(64, 32),
                                 nn.LeakyReLU(0.2))
    def forward(self, X):
        return self.net(X)
    
    