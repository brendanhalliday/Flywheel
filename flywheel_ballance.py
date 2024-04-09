"""
Final Project PHYS 879 
Author : Brendan S. Halliday
Queen's University

This program defines a new game environment class
for the inverted flywheel pendulum
All other programs for the final project should import this
file as it is working comfortably

"""

#%%
"Import libraries"
import pygame
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import random

#%%
"Define New Gym environment"
class TopBalanceFlyWheelEnv(gym.Env):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }
    def __init__(self, L, R, m1, m2, max_tau=0.8, max_w1=6*np.pi, 
                 max_w2=418.9, g = 9.81, fps= 60, init_time=30, 
                 random_init_position=False, env_config={}):
        """
        Initializes the flywheel pendulum class
        
        Required positional arguments:

            L           : Length of uniform rod
            R           : Radius of solid disk flywheel
            m1          : Mass of rod
            m2          : Mass of flywheel
            
        Optional Keyword Arguments:

            max_tau     : maximum torque output of flywheel motor
            max_w1      : maximum angular velocity of pendulum
            max_w2      : maximum angular velocity of flywheel
            g           : acceleration due to gravity
            fps         : framerate
            time        : time in seconds of one episode

        """
        # store game fps
        self.fps = fps

        # Pixel size of the game window
        self.window_size = 500  

        self.random_position = random_init_position

        if random_init_position:
            choice = random.random()
            if choice >= 0.5:
                init_var = -np.random.normal(0.1, 0.1)
            else:
                init_var = np.random.normal(0.1, 0.1)
        else:
            init_var = 0.1

        self.theta1      = np.pi + init_var
        self.theta2      = 0.
        self.w1          = 0.
        self.w2          = 0.

        # self.state is [sin(theta1), cos(theta1), w1, tanh(w2)] initial
        self.state = np.array([np.sin(self.theta1), 
                               np.cos(self.theta1), 
                               self.w1,
                               np.tanh(self.w2)])

        # define target state 
        self.target = np.array([0., -1., 0., 0.])

        # set trial_length in sec
        self.init_time = init_time
        self.trial_time = init_time

        # set lengths and masses
        self.rod_len     = L
        self.wheel_r     = R
        self.rod_mass    = m1
        self.wheel_mass  = m2

        # define maximum torque, and maximum angular velocities of 
        # both the rod and flywheel to bound energy
        self.max_tau = max_tau
        self.max_w1  = max_w1
        self.max_w2  = max_w2

        # define moments of inertia
        I2 = (1/2) * m2 * (R**2)
        I1 = (1/12) * m1 * (L**2)

        # define lengths
        l2 = L
        l1 = L/2

        self.I1, self.I2, self.l1, self.l2  = I1, I2, l1, l2
        self.m1, self.m2 = m1, m2
        self.g = g

        # define useful constants for calculating moments of inertia
        # see ipynb file for calculations and why this makes sense
        # compute useful constants once so the update equations need not compute them each time
      
        A = I1 + I2 + m1 * (l1**2) + m2 * (l2**2) 
        B = l1 * m1 + l1 * m2
        C = I1 + m1 * (l1**2) + m2 * (l2**2)
        self.const = g*B/C
        # torque terms in the equation for \ddot{\theta}_{i}
        self.torque_1 = max_tau/C
        self.torque_2 = max_tau*A/(I2*C)

        # Define the low and high values for each continuous variable
        low  = np.array([-1.0, -1.0, -max_w1, -1.0])
        high = np.array([1.0,   1.0,  max_w1, 1.0])

        # Create a Box space for the observation with four continuous variables
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)

        # We have 3 actions, corresponding to "torque_left", "no torque", "torque_right"
        self.action_space = spaces.Discrete(3)

        # The following dictionary maps abstract actions from `self.action_space` to
        # the direction we will walk in if that action is taken.
        # I.e. 0 corresponds to "no_right", 1 to "torque_ccw", -1 to "torque_cw"
        
        self._action_to_direction = {0:-1,
                                     1: 0, 
                                     2: 1}

    def init_render(self):
        """
        Render the game environment
        """
        import pygame
        pygame.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()

    def reset(self):
        """
        Reset the environment to initial state
        """
        if self.random_position:
            choice = random.random()
            if choice >= 0.5:
                init_var = -np.random.normal(0.1, 0.1)
            else:
                init_var = np.random.normal(0.1, 0.1)
        else:
            init_var = 0.1

        self.theta1      = np.pi + init_var
        self.theta2      = 0.
        self.w1          = 0.
        self.w2          = 0.

        self.state = np.array([np.sin(self.theta1), 
                               np.cos(self.theta1), 
                               self.w1,
                               np.tanh(self.w2)])
        
        self.trial_time = self.init_time
        info = {}
        return self.state, info
    
    def derivatives(self, y, t, const=1, torque_1=1, torque_2=1, direction=0):
        """
        Derivatives for the the reaction wheel
        pendulum

        y                     : previous values for state space vector
        const                 : g*B/C              - precalculated
        torque_1              : torque term for w1 - precalculated
        torque_2              : torque term for w2 - precalculated
        
        if <direction> == 0   : acceleration of flywheel set to 0
        elif <direction> == 1 : accelerate fly wheel clockwise
        else                  : accelerate fly wheel counter clockwise

        """

        theta, omega  =  np.hsplit(y, 2)
        dy            =  np.hsplit(np.zeros((len(y))), 2)
        st1           =  np.sin(theta[0])

        dy[0]         =  omega
        
        dy[1][0]      = -const*st1 - direction*torque_1
        dy[1][1]      = const*st1 + direction*torque_2
        
        return np.hstack(dy)
    
    def leapfrog_updater(self, f, y, t, dt, **kwargs):
        """ 
        Vectorized leapfrog method algorithm.
        (this can be generalized to N dimensions)

        Arguments:
        f : derivatives function
        y : data structure( will be lenth

        t : point in time
        h : time step value
        **kwargs: variable number of key word arguments
        """
        # define half step
        hh = dt / 2

        # save initial values
        ri, vi = np.hsplit(y, 2)

        rf = ri + hh * np.hsplit(f(y, t, **kwargs), 2)[0]         # 1 : r1 at h/2 using v0
        rfvi = np.hstack((rf, vi))

        vf = vi + dt  * np.hsplit(f(rfvi, t + hh, **kwargs), 2)[1] # 2 : v1 using a(r) at h/2
        rivf = np.hstack((ri, vf))

        rf = rf + hh * np.hsplit(f(rivf, t + dt, **kwargs), 2)[0]  # 3 : r1 at h using v1

        return np.hstack((rf, vf))
    
    def set_rod_speed(self, w1_new):
        """
        Solves/ limits the rod speed
        """
        # check rod speed
        if np.abs(self.w1) > self.max_w1:

            if self.w1 < 0:
                self.w1 = -self.max_w1
            else:
                self.w1 = self.max_w1
        else:
            self.w1 = w1_new
    

    def KE(self):
        """
        """
        w1, w2 = self.w1, self.w2
        I1, I2, l1, l2  = self.I1, self.I2, self.l1, self.l2 
        m1, m2 = self.m1, self.m2

        return 0.5*(I1*w1**2 + I2*(w1 + w2)**2 + m1*(l1*w1)**2  + m2*(l2*w1)**2) 
        

    def PE(self):
        """
        """
        l1, l2 = self.l1, self.l2 
        m1, m2, g = self.m1, self.m2, self.g
        t1 = self.theta1

        return -g*(l1*m1 + l2*m2)*np.cos(t1)


    def H(self):
        """
        Computes the total current energy
        H = T + U
        """
        return self.KE() + self.PE()

    def step(self, action): 
        """
        This function is used to iterate through 1 step in
        the algorithm
        """

        h = 1/self.fps
        t = 0

        # perform calculations
        # direction can be -1, 0, or 1 for torque direction
        direction = self._action_to_direction[action]

        # initialize dynamic variables of the flywheel pendulum
        y = np.array([self.theta1, self.theta2, self.w1, self.w2])

        # update the dynamic variables 
        y_new = self.leapfrog_updater(self.derivatives, y, t, h, const=self.const, 
                                      torque_1=self.torque_1, torque_2=self.torque_2, 
                                      direction=direction)

        # update angles 
        self.theta1, self.theta2 = y_new[0], y_new[1]

        # update speed but limit cap the speed if they exceed max_w2 or max_w1
        if np.abs(self.w2) > self.max_w2:
            # check sign of speed
            if self.w2 < 0:
                self.w2 = -self.max_w2
            else:
                self.w2 = self.max_w2

            # check rod speed
            self.set_rod_speed(y_new[2])

        else:
            # update flywheel speed
            self.w2 = y_new[3]

            # check rod speed
            self.set_rod_speed(y_new[2])

        # update observation state
        # the reason we don't use the y_new array is because the theta1, and theta2 are unbounded
        self.state = np.array([np.sin(self.theta1), 
                               np.cos(self.theta1), 
                               self.w1,
                               np.tanh(self.w2)])

        # Trial done?
        self.trial_time -= h

        if self.trial_time <= 0.:
            terminated = True
        else:
            if self.state[1] >= 0:
                terminated = True
            else:
                terminated = False
        # terminate the trial if a forbidden state has been reached. 
        
        # Give the computer a little treat
        if not terminated:
            if self.state[1] < np.cos(np.pi + 12*np.pi/180):
                reward = (1/2)*(1 - self.state[1])
            else:
                reward = -1
        else:
            reward = 0

        # pass aditional info
        info = {}

        return self.state, reward, terminated, False, info
    
    def render(self):
        """
        This function is to render the game for the human to play
        It renders the graphics given the angular position
        and velocities of the inverted flywheel pendulum
        """
        # set window colour
        self.window.fill((0,0,0))

        # set length ratios for the rod and wheel
        # such that it looks proportional to screen
        leng_ratio = self.rod_len/self.wheel_r # length of rod
        L = 150
        R = L*(1/leng_ratio)


        # theta2 is called since it is not contained in the 
        # self.state array
        t2 = self.theta2

        # center of the pygame window
        center = self.window_size/2

        # self.state[0] = sin(theta1)
        # self.state[1] = cos(theta1)
        p1 = (center, center)
        p2 = (L*self.state[0] + center, L*self.state[1] + center)
        p3 = (R*np.sin(t2) + L*self.state[0] + center, R*np.cos(t2) + L*self.state[1] + center)

        # Draw flywheel
        pygame.draw.circle(self.window, (0, 200, 200), p2, R)
        
        # draw rod
        pygame.draw.line(self.window, (0,100,100), p1, p2, 10)

        # draw flywheel line
        pygame.draw.line(self.window, (0,50,50), p2, p3, 4)
        
        # display all
        pygame.display.update()

    # def close():
    #     """
    #     Define a close function to close
    #     the Pygame render
    #     """
    #     import pygame
    #     pygame.display.quit()
    #     pygame.quit()

def pressed_to_action(keytouple):
    """
    Converts key strokes to actions
    """
    action_torque = 1

    if keytouple[pygame.K_LEFT]      == True:  # left  is -1
        action_torque = 0

    if keytouple[pygame.K_RIGHT]     == True:  # right is +1
        action_torque = 2
    
    return action_torque


def main_function(environment, fps=60):
    """
    The main function used for gaming
    """
    run = True
    while run:
        environment.clock.tick(fps)
        # ─── CONTROLS ───────────────────────────────────────────────────────────────────
        # end while-loop when window is closed
        get_event = pygame.event.get()
        for event in get_event:
            if event.type == pygame.QUIT:
                run = False
        # get pressed keys, generate action
        get_pressed = pygame.key.get_pressed()
        action = pressed_to_action(get_pressed)
        # calculate one step
        environment.step(action)
        # render current state
        environment.render()
    pygame.quit()

#%%
if __name__ == '__main__':

    L, R, m1, m2 = 0.7, 0.3, 0.4, 0.4 # SI Units
    environment = TopBalanceFlyWheelEnv(L, R, m1, m2)
    environment.init_render()
    main_function(environment)

#%%
