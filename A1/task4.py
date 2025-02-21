"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the MultiBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, set_pulled, reward): This method is called 
        just after the give_pull method. The method should update the 
        algorithm's internal state based on the arm that was pulled and the 
        reward that was received.
        (The value of arm_index is the same as the one returned by give_pull 
        but set_pulled is the set that is randomly chosen when the pull is 
        requested from the bandit instance.)
"""

import numpy as np

# START EDITING HERE
import math
# You can use this space to define any helper functions that you need
# END EDITING HERE


class MultiBanditsAlgo:
    def __init__(self, num_arms, horizon):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        # START EDITING HERE
        self.success = np.zeros((num_arms,2))   # No. of times each arm returns 1 for each set 
        self.failures = np.zeros((num_arms,2))  # No. of times each arm returns 0 for each set 
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        sampled_mean = np.random.beta(self.success+1,self.failures+1)   # Sample from corresponding Beta distributions
        return np.argmax(sampled_mean.mean(axis=1))                     # Return the the arm index where average is best.
        # END EDITING HERE
    
    def get_reward(self, arm_index, set_pulled, reward):
        # START EDITING HERE
        self.success[arm_index,set_pulled]  += reward
        self.failures[arm_index,set_pulled] +=1 - reward
        # END EDITING HERE

