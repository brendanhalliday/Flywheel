"""
This script loads the fully trained neural network
calculated by train_flywheel.py

Author: Brendan Halliday
Institute: Queen's University 
Date: April 13th, 2024
"""

#%%
"Import modules"
import train_flywheel_swing as fly
import torch
from flywheel_swingup_ballance import SwingUpFlyWheelEnv
from itertools import count
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
    

#%%
"Load in weights and initiate the trained model"
PATH = 'weights_swing.pt'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
L, R, m1, m2 = 0.7, 0.3, 0.4, 0.4 # SI Units
env = SwingUpFlyWheelEnv(L, R, m1, m2)
state, info = env.reset()
# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

model = fly.DQN(n_observations, n_actions).to(device)
model.load_state_dict(torch.load(PATH))
env.init_render()

# if GPU is to be used

state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

for t in count():
    env.clock.tick(60)
    action = select_action(model, state)
    observation, reward, terminated, truncated, _ = env.step(action.item())
    reward = torch.tensor([reward], device=device)
    done = terminated or truncated
        # render current state
    env.render()
    if done:
        env.close()
        break
    else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        # Move to the next state
        state = next_state