"""
This script loads the fully trained neural network
calculated by train_flywheel.py

The program then shows an animation of the 
flywheel with the option of saving the animation.

We finally have the option to produce energy 
and phase plots. 

Author: Brendan Halliday
Institute: Queen's University 
Date: April 13th, 2024
"""

#%%
"Import modules"
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import train_flywheel_swing as fly
import torch
from flywheel_swingup_ballance import SwingUpFlyWheelEnv
from flywheel_ballance import TopBalanceFlyWheelEnv
from itertools import count
from generate_video import Movie_Maker

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

font_size = 20
# set font for good looking graphs
font = {'family'  : 'serif',
        'serif'   : ['Computer Modern Roman'],
        'style'   : 'normal',
        'weight'  : 'bold',
        'size'    : font_size}

mpl.rc('font', **font)
plt.rcParams['figure.dpi'] = 300 # resolution on screen
plt.rcParams.update({'font.size': font_size})
#%%
"Define Subroutines"
def select_action(policy, state):
    
    # do not compute gradient in contect of backpropagation 
    # this helpd with memeory management
    with torch.no_grad():

        # t.max(1) will return the largest column value of each row.
        # second column on max result is index of where max element was
        # found, so we pick action with the larger expected reward.
        
        # policy_net(state) returns a action
        # tensor and we then choose 
        # the index of the action tensor with
        # the max probability or the action
        # with the largest expected reward
        # 
        # -> .max(1) returns a tensor with max
        # value along each row (in other words, it returns the
        # largest column value in each row)
        # -> .indices returns the indices of the max row 
        # elements
        # ->  .view(1,1) reshapes the tensor as a 1 by 1 tensor
        return policy(state).max(1).indices.view(1, 1)
    

def plot_energy_and_phase(energy, angle, velocity, 
                          save=False, fig_size=(4,10), 
                          save_type='png', quality=300,
                          save_title='phase_energy'):
    """
    Plots energy as a for the fully
    trained animation
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, 
                                   figsize=fig_size, 
                                   constrained_layout=True)
    t = np.arange(0, len(energy))/60.
    ax2.plot(t, energy)
    ax1.plot(angle, velocity)

    ax1.set_xlabel(r'$\theta$ [rad]')
    ax1.set_ylabel('$\omega$ [rad/s]')
    ax1.grid()

    ax2.set_xlabel('time [s]')
    ax2.set_ylabel('Energy [J]')
    ax2.set_xlim([t[0], t[-1]])
    ax2.grid()

    if save:
        plt.savefig(save_title +'.'+ save_type, dpi = quality, 
                        bbox_inches = "tight", format=save_type)

    plt.show()
    

#%%
"Load in weights and initiate the trained model"
situation = {'balance': True,
             'swing': False}

sit = situation['balance']

# save an animation
save = False

if sit:
    PATH = 'weights.pt'
    L, R, m1, m2 = 0.7, 0.3, 0.4, 0.4 # SI Units
    env = TopBalanceFlyWheelEnv(L, R, m1, m2)

else:
    PATH = 'weights_swing.pt'
    L, R, m1, m2 = 0.21, 0.085, 0.115, 0.526 # SI Units
    max_tau = 0.215 # Nm
    env = SwingUpFlyWheelEnv(L, R, m1, m2)


state, info = env.reset()
# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# load in model with saved weights
model = fly.DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load(PATH))
env.init_render()
state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)


if save:
    video = Movie_Maker((500, 500))

ENERGY = []
W = []
THETA = []

for t in count():
    env.clock.tick(60)
    action = select_action(model, state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated
        # render current state
    env.render()
    ENERGY.append(env.H())
    THETA.append(env.theta1)
    W.append(env.w1)
    
    if save:
        video.png(env.window)
    if done:
        env.close()
        break
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        # Move to the next state
        state = next_state
if save:
    video.mp4()


#%%
"Plot energy and phase plots"
plot_energy_and_phase(ENERGY, THETA, W, save=False)