import numpy as np
import torch

seed = 99

"""***************************
        Car Settings
***************************"""
m1 = 450 #kg
m2 = 45 #kg
m = [m1,m2]
cb = 1500    #N/(m/s)
kb = 15000    #N/m
kw = 150000   #N/m

"""***************************
        Model Settings
***************************"""
input_dim = 6
output_dim = 2
Q_hidden_dim = 32
mu_hidden_dim = 16

"""***************************
        Optimization Settings
***************************"""
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
gamma = 0.95 # in Q_tgt(st, at) = rt + gamma*Q_tgt(s{t+1}, a{t+1})
lr = 0.001  # learning rate of both Q_origin & mu_origin
tau = 0.99  # used for soft update: Q_tgt = tau*Q_target + (1-tau)*Q_origin
bs = 512
num_episodes = 700

"""**************************
    ROAD PROFILE CONDITIONS
**************************"""
MAX_BUMP = 0.09 # in meters
t_stop = 10   # Total episode length
dt = 0.01     # Time Steps
TSPAN = [0, t_stop]
TIME = np.arange(0, t_stop, dt) # simulation time series