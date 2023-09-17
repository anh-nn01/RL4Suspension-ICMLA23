import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import *
from model import *


"""**********************************************************
    Trainer class which:
        1. Initialize Actor/Critic weights
        2. Perform backpropagation given (s{t}, a{t}, r{t}, s{t+1})
        3. Save checkpoints: 
            ckpt_Q_origin -> Q-Network; 
            ckpt_mu_origin -> Policy Network
**********************************************************"""
class Trainer:
    def __init__(self, ckpt_Q_origin, ckpt_mu_origin):
        self.ckpt_Q_origin = ckpt_Q_origin
        self.ckpt_mu_origin = ckpt_mu_origin

        """######################################################################
            1. Initialize Critics (Q-Networks)
                Q_origin: instantly updated weights
                Q_target: delayed weights (updated periodically)
                    * NOTE: Q_target are not back-propagated -> disable require_grads for efficiency
            --> "Twin-delayed" to increase training stability & convergence
        ######################################################################"""
        self.Q_origin = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
        self.Q_origin = self.Q_origin.to(device)
        self.Q_origin.requires_grad_(True)

        self.Q_target = QNet(input_dim+output_dim, Q_hidden_dim, 1).to(device)
        self.Q_target.requires_grad_(False) # Target network's weights are copied but not back-propagated

        """###################################################################### 
            2. Initialize Actors (Policy Networks)
                Q_origin: instantly updated weights
                Q_target: delayed weights (updated periodically)
                    * NOTE: Q_target are not back-propagated -> disable require_grads for efficiency
            --> "Twin-delayed" to increase training stability & convergence
        ######################################################################"""
        self.mu_origin = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
        self.mu_origin = self.mu_origin.to(device)
        self.mu_origin.requires_grad_(True)

        self.mu_target = PolicyNet(input_dim, mu_hidden_dim, output_dim).to(device)
        self.mu_target.requires_grad_(False) # Target network's weights are copied but not back-propagated

        print('Learning rate:', lr)
        print('Device:', device)
        print('Q-Network:', self.Q_origin)
        print('Policy Network:', self.mu_origin)

        """###################################################################### 
            3. Initialize Optimizer
        ######################################################################"""
        self.optimizer_Qnet = torch.optim.Adam(self.Q_origin.parameters(), lr=lr)
        self.optimizer_munet = torch.optim.Adam(self.mu_origin.parameters(), lr=lr/10)

    """**********************************************************
        Optimization framework for Twin-Delayed (TD) Critic & Actor
        :states      - st
        :actions     - at
        :rewards     - rt
        :next_states - s{t+1}
        :dones       - bool

        Purpose:
            * Update weights of Q_orig and mu_orig using back-propagation
            * Q_target & mu_target are NOT updated (soft delayed updated separately)
    **********************************************************"""
    def optimize(self, states, actions, rewards, next_states, dones):
        """ 1. Tensor conversion of (st, at, rt, s{t+1}, done)"""
        states = torch.tensor(np.array(states), dtype=torch.float).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float).to(device)
        # actions = actions.unsqueeze(dim=1)
        rewards = torch.tensor(rewards, dtype=torch.float).to(device)
        rewards = rewards.unsqueeze(dim=1)
        next_states = torch.tensor(next_states, dtype=torch.float).to(device)
        dones = torch.tensor(dones, dtype=torch.float).to(device)
        dones = dones.unsqueeze(dim=1)

        """ 2. Critic Loss Optimization"""
        self.optimizer_Qnet.zero_grad()
        qvals = self.Q_origin(states, actions)                            # Q_orig(st, at)
        a_next_tgt = self.mu_target(next_states)                          # a{t+1}_tgt
        qvals_next_tgt = self.Q_target(next_states, a_next_tgt)           # Q_tgt(s{t+1}, a{t+1}_tgt)
        qvals_tgt = rewards + (1.0 - dones) * gamma * qvals_next_tgt # Q_tgt(st, at) = rt + gamma*Q_tgt(s{t+1}, a{t+1})

        """ 2.1. Update Critic -> Minimize MSE between
                `Q_orig(st, at)` and `rt + gamma*Q_tgt(s{t+1}, a{t+1})`
                    where:
                        Q_orig, Q_tgt    is given
                        mu_orig, mu_tgt  is given
                        st, at, rt, s{t+1} is given
                        a{t+1} = mu_tgt(s{t+1})
        """
        loss_Q = F.mse_loss(qvals, qvals_tgt, reduction="none")
        loss_Q.sum().backward()
        self.optimizer_Qnet.step()


        """ 3. Actor Loss Optimization"""
        """ 3.1. Freeze Critic (Q_origin) before Actor (mu_origin) optimization"""
        for p in self.Q_origin.parameters():
            p.requires_grad = False

        """ 3.2. Update Actor -> Maximize the reward"""
        self.optimizer_munet.zero_grad()
        a_next = self.mu_origin(states)            # a{t+1} = mu_orig(s{t})
        q_tgt_max = self.Q_origin(states, a_next)  #
        loss_actor = (-q_tgt_max) - 0.1*abs(a_next)
        loss_actor.sum().backward()          # (q_tgt_max) or (-q_tgt_max) ?
        self.optimizer_munet.step()

        """ 3.1. Unfreeze Critic (Q_origin) """
        for p in self.Q_origin.parameters():
            p.requires_grad = True # enable grad again


    """**********************************************************
        Update lr of Q-Network and Policy Network to lr/10
    **********************************************************"""
    def reduce_lr(self):
        for g in self.optimizer_Qnet.param_groups:
            g['lr'] = g['lr'] / 10
        for g in self.optimizer_munet.param_groups:
            g['lr'] = g['lr'] / 10

    """**********************************************************
        Soft-update Delayed Target models:
            Q_target = tau*Q_target + (1-tau)*Q_orig
            mu_target = tau*mu_target + (1-tau)*mu_orig
    **********************************************************"""
    def update_target(self):
        for var, var_target in zip(self.Q_origin.parameters(), self.Q_target.parameters()):
            var_target.data = tau * var_target.data + (1.0 - tau) * var.data
        for var, var_target in zip(self.mu_origin.parameters(), self.mu_target.parameters()):
            var_target.data = tau * var_target.data + (1.0 - tau) * var.data

    """**********************************************************
        Pick up action with Ornstein-Uhlenbeck noise
    **********************************************************"""
    def pick_sample(self, s, ou_action_noise):
        with torch.no_grad():
            s = np.array(s)
            s_batch = np.expand_dims(s, axis=0)
            s_batch = torch.tensor(s_batch, dtype=torch.float).to(device)
            action_det = self.mu_origin(s_batch)
            action_det = action_det.squeeze(dim=1)
            noise = ou_action_noise()
            action = action_det.cpu().numpy() + noise
            # action = np.clip(action, -1.0, 1.0)
            # return float(action.item())
            return action.astype(float)[0]
        
    """**********************************************************
        Save Actor-Critic weights as new checkpoints
    **********************************************************"""
    def save_checkpoints(self):
        torch.save(self.Q_origin, self.ckpt_Q_origin)
        torch.save(self.mu_origin, self.ckpt_mu_origin)
        


