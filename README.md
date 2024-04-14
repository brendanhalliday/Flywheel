Run train_flywheel_balance.py to train the flywheel yourself.

This is a reinforced learning algorithm that uses a 
Deep Q-Network (DQN) to approximate the Quality function
to find the optimal policy.

This program learns to balance the inverted flywheel pendulum from
a off centered upright position. This means that the rod is not 100% 
vertical in it's initial position. It must learn to counter the torque
applied by gravity from this starting psotion to balance the rod.

Run train_flywheel_swing.py to trian the swing up and balance version
of the code. 

To run an animation of the trianed model, run load_trained.py


Contact Brendan for details

To install dependencies, run the following:
pip install -r /path/to/requirements.txt
