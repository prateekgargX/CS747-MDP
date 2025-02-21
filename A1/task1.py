"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

from hashlib import new
from itertools import count
import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
def kl_bern(p,q):
    """return KL divergence between 2 bernoulli distribtuions parameterised by p and q
        p and q: vectors of values
    """
    eps = 0.00000001 # For numerical stability
    p[p==0] += eps
    p[p==1] -= eps
    q[q==0] += eps
    q[q==1] -= eps
    return p*np.log(p/q) + (1-p)*np.log((1-p)/(1-q))

def rhs(count,t, c=0):
    """return the rhs calculated to find UCBs for KL-UCB algorithm
        compatible with vector of counts
    """
    te = math.log(t)
    if te == 0: te = 0.00000001 # For numerical stability
    return (te + c*math.log(te))/count

def get_ucb_kl(counts,values,total_counts,prec=0.00000001,c=0, max_iter=30):
    """returns ucb for KL-UCB algorithm using vectorized Newton raphson method
    """
    num_arms = len(counts)
    rhss = rhs(counts,total_counts,c)   # calculating rhs for each arm
    q = np.zeros(num_arms)              # placeholder for solutions
    l = np.zeros(num_arms)
    l+= values                          # Lower estimate of the solutions
    u = np.ones(num_arms)               # Upper estimate of the solutions

    for _ in range(max_iter):
        q = (u+l)/2                                                 # Candidate Solutions
        kls = kl_bern(values,q)                                     # KL-div for each arm
        NEG_MASK = kls < rhss                                       # indices where lower estimate needs to be updated
        l[NEG_MASK] = q[NEG_MASK]                                   # updating lower estimate
        u[np.logical_not(NEG_MASK)] = q[np.logical_not(NEG_MASK)]   # updating upper estimate
        NOT_UPDATE_MASK = abs(u-l) < prec                           # checking if every value is within required precision
        if all(NOT_UPDATE_MASK) : break                             # Terminate the loop
    return q

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)    # no of times each arm has been pulled
        self.values = np.zeros(num_arms)    # empirical average of each arm
        self.total_counts = 0               # total no. of pulls(to be used in the calculation of ucb)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.total_counts+=1                    # total count is increased everytime an arm is pulled
        for i in range(self.num_arms):
            if self.counts[i] == 0 : return i   # play each arm atleast once(ucb==inf)
        # Calculate the ucb for each arm
        ucb = self.values + np.sqrt((2*math.log(self.total_counts))/self.counts)
        return np.argmax(ucb)
        # END EDITING HERE  
        
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1                             # increment the arm counter by 1
        n = self.counts[arm_index]                              # to be used to update empirical average
        value = self.values[arm_index]                           
        new_value = ((n - 1) / n) * value + (1 / n) * reward    # Update emperical average 
        self.values[arm_index] = new_value                      
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE
        self.counts = np.zeros(num_arms)    # no of times each arm has been pulled
        self.values = np.zeros(num_arms)    # empirical average of each arm
        self.total_counts = 0               # total no. of pulls(to be used in the calculation of ucb)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.total_counts+=1                  # total count is increased everytime the arm is pulled
        for i in range(self.num_arms):
            if self.counts[i] == 0 : return i # play each arm atleast once(==ucb==1)
        ucb = get_ucb_kl(self.counts,self.values,self.total_counts,prec=0.0001,c=0,max_iter=10) # for task 1:0.0001, max_iter = 10
        return np.argmax(ucb)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        self.counts[arm_index] += 1                             # increment the arm counter by 1
        n = self.counts[arm_index]                              # to be used to update empirical average
        value = self.values[arm_index]                           
        new_value = ((n - 1) / n) * value + (1 / n) * reward    # Update emperical average 
        self.values[arm_index] = new_value                      
        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
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
        self.success[arm_index]  += reward
        self.failures[arm_index] +=1 - reward
        # END EDITING HERE
