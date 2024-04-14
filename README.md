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

Message for Nickolaos (and future Brendan):
    If you need to use more packages, update the requirements file:

    1. activate virtual envoronment. 
    2. cd to inverted_flywheel_pendulum
    3. python -m pip freeze > requirements.txt
    4. Commit
    5. Push
