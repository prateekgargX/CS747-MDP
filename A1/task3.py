"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the FaultyBanditsAlgo class. Here are the method details:
    - __init__(self, num_arms, horizon, fault): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)
"""

import numpy as np

# START EDITING HERE
# END EDITING HERE

class FaultyBanditsAlgo:
    def __init__(self, num_arms, horizon, fault):
        # You can add any other variables you need here
        self.num_arms = num_arms
        self.horizon = horizon
        self.fault = fault                  # probability that the bandit returns a faulty pull
        # START EDITING HERE
        self.success = np.zeros(num_arms)   # No. of times each arm returns 1
        self.failures = np.zeros(num_arms)  # No. of times each arm returns 0
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        sampled_mean = np.random.beta(self.success+1,self.failures+1)   # Sample from corresponding Beta distributions
        return np.argmax(sampled_mean)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        P_E_r = (0.5*self.fault)/(0.5*self.fault + (((self.failures[arm_index]+1)*(1-reward) + (self.success[arm_index]+1)*reward)/(self.failures[arm_index]+self.success[arm_index]+2))*(1 - self.fault))
        E_r = np.random.binomial(1,P_E_r)
        if E_r <1:
            self.success[arm_index]  += reward
            self.failures[arm_index] +=1 - reward
        #END EDITING HERE
