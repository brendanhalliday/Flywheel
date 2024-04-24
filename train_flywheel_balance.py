"""
This scipt trains the inverted flywheel pendulum
to balance

The algorthm was inspired by the Pytorch tutorial
written by Adam Paszke and Mark Towers
for training the cart-pole. The code has been
adapted by Brendan Halliday for the flywheel.

Author: Brendan Halliday
Institute: Queen's University 
Date: April 24th, 2024
"""

# %%
"""Import libraries"""

import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

font_size = 20
# set font for good looking graphs
font = {'family'  : 'serif',
        'serif'   : ['Computer Modern Roman'],
        'style'   : 'normal',
        'weight'  : 'bold',
        'size'    : font_size}

matplotlib.rc('font', **font)

from flywheel_ballance import TopBalanceFlyWheelEnv

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# %%
"""Initialize the environment"""
# env = gym.make("CartPole-v1")#, render_mode="human")

L, R, m1, m2 = 0.7, 0.3, 0.4, 0.4 # SI Units
env = TopBalanceFlyWheelEnv(L, R, m1, m2)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
"""Initialize transition variable and Replay Buffer"""
# transition stores state, action pairs with next state reward pairs
# Transition literally stores the transition
    # -> current state
    # -> action
    # -> next state
    # -> associated reward
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


# replay memory is a cyclic buffer of bounded size that holds the transitions observed recently 
class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    # sampler samples the buffer for selecting a random batch for training
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

# now we define our module
# %%
"""Define Neural Net class"""
class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
# %%
""""""
# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.95
EPS_END = 0.05 # was 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)

# next we load the same weights and biases from 
# policy net into target_net
# this initially makes the two networks identical
# Target net is used for stability
target_net.load_state_dict(policy_net.state_dict())

# initialize the optimizer based on
# the parameters(weights and biases) of 
# policy_net and the learing rate <LR>
optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)

# Initialize the cyclical memory buffer for 
# storing transitions
memory = ReplayMemory(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random() # choose random value between 0 and 1

    # make a threshold value that is EPS_START when steps_done = 0 and EPS_END when steps_done = large
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    # the above function makes exploration
    # more probable at lower steps done

    # update steps done
    steps_done += 1


    if sample > eps_threshold:
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
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        # return a pytorch tensor, which is like a numpy array
        # with a random action tensor
        return torch.tensor([[env.action_space.sample()]], 
                            device=device, dtype=torch.long)


episode_durations = []


def plot_durations(show_result=False):
    """
    This function should plot
    duration vs episode number
    """
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.grid()
        pass
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# %%
"""Optimization Model"""    
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    # once memory is that same size as batch size,
    # we can start sampling
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), 
                                          device=device, 
                                          dtype=torch.bool)
    
    # torch.cat (concatenates a tensor by adding rows to the tensor)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    # convert to tensors
    state_batch  = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), 
    # then we select the columns of actions taken.
    # These are the actions which would've been 
    # taken for each batch state according to 
    # policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss() # this is the type of loss function
                                  # similar to least squares in linear regression  
    # compute loss
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model---------------------

    # optimizer is the AdamW optimizer  
    # zero_grad() sets gradients to zero before recalculating gradients
    # we do this because Pytorch accumulates gradients on subsequent backward pass
    # otherwise, the new calculated gradients will be an accumulation
    # of old gradients
    optimizer.zero_grad()

    # computes the gradient of loss function
    loss.backward()
    
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)

    # performs update of weights and biases using calculated gradient (performs gradient decsent)
    optimizer.step()

#%%
"""Training Loop"""
show_process = False # if true, the training process will be shown in real time
save = True # save weights
if __name__ == "__main__":
    if torch.cuda.is_available():
        num_episodes = 600
    else:
        num_episodes = 100

    for i_episode in range(num_episodes):
        # Initialize the environment and get its state

        state, info = env.reset()
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        if i_episode != num_episodes - 1:

            for t in count():
                action = select_action(state)
                observation, reward, terminated, truncated, _ = env.step(action.item())
                reward = torch.tensor([reward], device=device)

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory
                memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τθ + (1 − τ)θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()

                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1 - TAU)
                target_net.load_state_dict(target_net_state_dict)

                if terminated:
                    episode_durations.append(t + 1)
                    if show_process:
                        plot_durations()
                    break
                    break

        else:
            print('Complete')
            plot_durations(show_result=True)
            plt.ioff()
            plt.savefig('balance_training.png', dpi = 300, bbox_inches = "tight", format='png')
            plt.show()
            state, info = env.reset()
            env.init_render()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                env.clock.tick(60)
                action = select_action(state)
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
    

#%%
"Save weights for reloading the model later"
if save:
    torch.save(policy_net.state_dict(), 'weights.pt')
