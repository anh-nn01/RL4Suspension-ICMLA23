import random
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from scipy.integrate import solve_ivp
from tqdm.notebook import tqdm
from road_generator import *
from config import *
from model import *
from trainer import *

"""**********************************************************
    Replay buffer class to enrich past experience
    ->  avoid biased toward recent experience
        (hence forgot past interactions / rare situations)
**********************************************************"""
class replayBuffer:
    def __init__(self, buffer_size: int):
        self.buffer_size = buffer_size
        self.buffer = []
        self._next_idx = 0

    def add(self, item):
        if len(self.buffer) > self._next_idx:
            self.buffer[self._next_idx] = item
        else:
            self.buffer.append(item)
        if self._next_idx == self.buffer_size - 1:
            self._next_idx = 0
        else:
            self._next_idx = self._next_idx + 1

    def sample(self, batch_size):
        indices = [random.randint(0, len(self.buffer) - 1) for _ in range(batch_size)]
        states   = [self.buffer[i][0] for i in indices]
        actions  = [self.buffer[i][1] for i in indices]
        rewards  = [self.buffer[i][2] for i in indices]
        n_states = [self.buffer[i][3] for i in indices]
        dones    = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, n_states, dones

    def length(self):
        return len(self.buffer)

"""**********************************************************
    Ornstein-Uhlenbeck noise implemented by OpenAI
    Copied from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
**********************************************************"""
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

"""**********************************************************
    Map Actor Network's output to Cs (System's Damping) action space
    Range: [-600, +600] N
**********************************************************"""
def get_cs(a):
    ca = 600*a
    return cb + ca

"""**********************************************************
    Map Actor Network's output to Ks (System's Stiffness) action space
    Range: [-2500, +5000] N
**********************************************************"""
def get_ks(a):
    # positive and negative stiffness has different range
    ka = 5000*a if a > 0 else 2500*a
    return kb + ka

"""**********************************************************
    Execute training process
**********************************************************"""
def execute_training(ckpt_Q_origin, ckpt_mu_origin):
    """ Initialize Trainer (model weights & optimizers)"""
    trainer = Trainer(ckpt_Q_origin=ckpt_Q_origin, 
                      ckpt_mu_origin=ckpt_mu_origin)
                      
    """ Replay buffer """
    # buffer = replayBuffer(buffer_size=1000000)
    buffer = replayBuffer(buffer_size=100000)

    """ Training history"""
    reward_records = []
    best_score = -99999
    rl_cs = []
    rl_ks = []

    """ Episode Iteration"""
    for episode in tqdm(range(num_episodes)):
        """--------------------------------------------------------------
            1. Randomly Generate new road profile each episode
        --------------------------------------------------------------"""
        # road_profile = generate_road(TIME, MAX_BUMP)
        road_profile = list(RoadProfile().get_profile_by_class("E", t_stop, dt)[1][1:])

        """--------------------------------------------------------------
            ODE (Ordinary Differential Equation) of Quarter Car model
            :t  - timesteps
            :y0 - initial state [xb, xw, d/dt(xb), d/dt(xw)]
            :m  - tuple containing (m_body, m_wheel)
            :cs - constant related to ...
            :ks - spring stiffness
            :kw - tire stiffness

            :return [d/dt(xb), d/dt(xw), d2/dt(xb), d2/dt(xw)]
        --------------------------------------------------------------"""
        def odefun(t, y0, m, cs, kw, ks):
            m1=m[0];    # Body mass in kg
            m2=m[1];    # Wheel mass in kg

            """ Road condition at time step t"""
            t_idx = min(round(t/dt), len(TIME)-1)
            xr = road_profile[t_idx]
            # print(t)

            """ 1. Displacement & Velocity of body & wheel """
            xb = y0[0]                # x_body        (x1)
            xw = y0[1]                # x_wheel       (x2)
            dxb = y0[2]               # d(x_body)/dt  (dx1/dt)
            dxw = y0[3]               # d(x_wheel)/dt (dx2/dt)
            """ 2. Acceleration of body & wheel """
            d2xb = - ks/m1*(xb-xw) - cs/m1*(dxb-dxw) # + kw/m1*xr - kw/m1*xb
            d2xw = ks/m2*(xb-xw) + cs/m2*(dxb-dxw) + kw/m2*(xr-xw)

            return [dxb,dxw,d2xb,d2xw]
        
        """--------------------------------------------------------------
                Exploration factor 
        --------------------------------------------------------------"""
        if episode < 100:
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.5)
        elif episode < 200:
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.3)
        elif episode < 500:
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.2)
        else:
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.05)
        
        """--------------------------------------------------------------
            2. Initialize the first state of the quarter car model & RL model
                * state_ODE \in R^4: [x_body, x_wheel, d/dt(x_body), d/dt(x_wheel)]
                * state_RL  \in R^5: [d/dt(x_body), d/dt(x_wheel), d/dt(road_profile)]
        --------------------------------------------------------------"""
        s_ode = [road_profile[0], road_profile[0], 0, 0]  # [xb, xw, dxb, dxw] e R4 -> state representation of the quarter car model
        s_ode_prev = s_ode
        s_rl = None # [xb, xw, dxb, dxw, d2xb] e R5  -> state representation of the RL model
        done = False
        total_reward = 0
        temp_xb_time = []
        temp_dxb_time = []
        
        """--------------------------------------------------------------
                3. Each Episode optimization
        --------------------------------------------------------------"""
        for t_idx, t in enumerate(TIME):
            """####################################
                I. State & Action: S{t} & A{t} COMPUTATION
                    (current State & Action)
            ####################################"""
            """ 1.1. Get current state s{t} from ODE """
            xb, xw, dxb, dxw = np.array(s_ode)[0:4]
            dxr = (road_profile[t_idx] - road_profile[t_idx-1]) / dt
            xb_prev, xw_prev, dxb_prev, dxw_prev = np.array(s_ode_prev)[0:4]
            dxr_prev = (road_profile[max(t_idx-1, 0)] - road_profile[max(t_idx-2, 0)]) / dt
            
            # s_rl = [xb, xw, dxb, dxw, dxr] # rescale
            # s_rl = [dxb, dxw, dxr] # rescale
            s_rl = [dxb, dxw, dxr, dxb_prev, dxw_prev, dxr_prev] # rescale
            """ 1.2. Take action a{t} from RL's s{t}"""
            a = trainer.pick_sample(s_rl, ou_action_noise)
            a_cs, a_ks = a
            """ 1.3. Compute d2/dt (x_body)"""
            d2xb = (-get_cs(a_cs) * (dxb-dxw) - get_ks(a_ks) * (xb-xw)) / m1
            
            """ 1.4. Record curves"""
            temp_xb_time.append(xb)
            temp_dxb_time.append(dxb)
            rl_cs.append(get_cs(a_cs))
            rl_ks.append(get_ks(a_ks))
            
            
            """####################################
                II. S{t+1} TRANSITION (from S{t} & A{t})
            ####################################"""
            # s_next, r, done, _ = env.step(a)
            """ 2.1. Solve ODE for s{t+1} """
            # del xb, xw, dxb, dxw # delete these variables for debugging purpose
            yout = solve_ivp(odefun, [t, t+dt], s_ode, args=(m, get_cs(a_cs), kw, get_ks(a_ks) ), dense_output=True)
            # yout.y is the solution of the ODE -> s{t+1}
            xb_next, xw_next, dxb_next, dxw_next = yout.y[:,-1]
            if t_idx < len(TIME)-1:
                dxr_next = (road_profile[t_idx+1] - road_profile[t_idx]) / dt
            """ 2.2. Record next states"""
            s_ode_next = [xb_next, xw_next, dxb_next, dxw_next]
            # s_rl_next = [xb, xw, dxb]
            # s_rl_next = [xb, xw, dxb, dxw, dxr]
            # s_rl_next = [dxb_next, dxw_next, dxr_next]
            s_rl_next = [dxb_next, dxw_next, dxr_next, dxb, dxw, dxr]
            """ 2.3. Select a_next"""
            ou_action_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(output_dim), sigma=np.ones(output_dim) * 0.0)
            a_next = trainer.pick_sample(s_rl_next, ou_action_noise)
            a_cs_next, a_ks_next = a_next
            d2xb_next = (-get_cs(a_cs_next) * (dxb_next-dxw_next) - get_ks(a_ks_next) * (xb_next-xw_next)) / m1
            """ 2.4. Check final state"""
            done = False if (t_idx < len(TIME)-1) else True
            
            
            """####################################
                III. REWARDS, BUFFER, & MODEL UPDATES
            ####################################"""
            """ 3.1. Reward function"""
            # r = -0.7*abs(xb) - 0.3*(dxb)**2
            # r = -0.9*abs(xb) - 0.1*abs(dxb) # - abs(xb-xw)
            #  r = -0.1*abs(dxb) # -> TRAINED 1 USING THIS (with buffer size 1M)
            r = -1/10*abs(dxb_next) # -> TRAINED 2,3,4 USING THIS (with buffer size 100K)
            # r = -1/10*abs(dxb_next) - 1/100*abs(d2xb_next) # -> TRAINED 2,3,4 USING THIS (with buffer size 100K)
            # r = -1/10*abs(xb) - 0.25e2*(abs(xb)**3)
            # r = -10*abs(d2xb)
            # r = -0.25e2*(abs(xb)**3)
            # r = -1000*(abs(xb)**3)
            # r = -1*(0.01*(abs(d2xb)**2))
            # r = (-1/10*(abs(xb))) - (0.001*(abs(d2xb)**2))
            # print(r)
            
            total_reward += r
            
            """ 3.2. buffer experience"""
            buffer.add([s_rl, a, r, s_rl_next, float(done)])
            
            """ 3.3. Update model based on buffered experience (st, at, rt, s{t+1}, done) """
            if buffer.length() >= bs:
                states, actions, rewards, n_states, dones = buffer.sample(bs)
                trainer.optimize(states, actions, rewards, n_states, dones)
                trainer.update_target()

            """####################################
                IV. STATE TRANSITION
            ####################################"""
            s_ode_prev = s_ode
            s_ode = s_ode_next

        # Output total rewards in episode (max 500)
        print(f"Run episode {episode} with rewards {total_reward}")
        reward_records.append(total_reward)
        np.save('Reward.npy', np.array(reward_records))

        if best_score < total_reward:
            best_score = total_reward
            print(f'New best score: {best_score}')
            rl_xb_time = temp_xb_time
            rl_dxb_time = temp_dxb_time
            trainer.save_checkpoints()

if __name__ == '__main__':
    ckpt_Q_origin = '../checkpoints/Q_origin_Sep16_2023.pt'
    ckpt_mu_origin = '../checkpoints/mu_origin_Sep16_2023.pt'
    print('Start Training Process...')

    execute_training(ckpt_Q_origin, ckpt_mu_origin)
