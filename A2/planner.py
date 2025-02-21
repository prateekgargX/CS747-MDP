#! /usr/bin/python
import argparse
from asyncore import read
import numpy as np
import pulp as pl

parser = argparse.ArgumentParser()

def readMDPfile(file_path):
    """
    utility function to return mdp instance from prescribed file path
    returns: (S,A,T,R,gamma)
    """
    with open(file_path,'r') as file:
        lines = file.readlines()
    
    S = int(lines[0].split()[1])  # number of states
    A = int(lines[1].split()[1])  # number of actions

    gamma = float(lines[-1].split()[1]) # discount factor
    mdp_type = lines[-2].split()[1] # MDP type

    T = np.zeros((S,A,S)) # initializing the transition function with all zeros
    R = np.zeros((S,A,S)) # initializing the reward function with all zeros

    s_term = []
    # Extracting all the terminal state
    for s in lines[2].split()[1:]:
        s = int(s)
        if s >=0 :
            s_term.append(s)
            T[s,:,s] = 1  # Terminal states only transitions to themselves 
    
    # Extracting reward, transition for all (s,a,s')
    for line in lines[3:-2]:
        line_l = line.split()
        s0 = int(line_l[1])
        ac = int(line_l[2])
        s1 = int(line_l[3])
        T[s0,ac,s1] = float(line_l[5])
        R[s0,ac,s1] = float(line_l[4])

    return S,A,T,R,gamma, mdp_type, s_term

def readPolicyfile(file_path):
    """
    utility function to return policy from prescribed file path
    returns: policy pi
    """
    with open(file_path,'r') as file:
        lines = file.readlines()

    return np.array(lines, dtype=int)

def BellmanOptimalityOp(T,R, gamma):
    """
        returns Bellman Optimality operator for a given transition function T, reward function R, and  discount factor gamma
    """
    def bstar(F):
        return np.max(np.sum(T*(R + gamma*F.reshape((1,1,-1))), axis=2), axis=1).reshape((-1,1))
    return bstar

def BellmanOp(T,R, gamma, pi):
    """
        returns Bellman operator for a given transition function T, reward function R, discount factor gamma, and policy pi
    """
    def b_pi(F):
        return np.sum(T*(R + gamma*F.reshape((1,1,-1))), axis=2)[np.arange(T.shape[0]),pi].reshape((-1,1))
    return b_pi

def valueIteration(S,A,T,R,gamma):
    """
        returns optimal value function for a given MDP(S,A,T,R,gamma)
    """
    # Initialise with a value function which is all zeros, takes care of terminal state values as well.
    V = np.zeros((S,1))
    # if a policy is specified, we apply Bellman Operator otherwise we apply Bellman Optimality Operator to get value 
    bstar = BellmanOptimalityOp(T,R, gamma)

    MAX_ITER =100000 # Maximum number of iterations to run algorithm for, for gamma<1 only 400-500 iterations are used.
    tol = 0.0000001 # norm of difference of consecutive vectors to assume convergence

    for i in range(MAX_ITER):
        V_new = bstar(V) # apply the operator
        if np.linalg.norm(V_new-V)< tol and np.log(np.linalg.norm(V_new)/tol)/(0.0000001+np.log(1/gamma)) < i:
            # Check if converged and if converged, whether the error from the optimal value is within tol
            V = V_new
            break
        V = V_new
    
    return V

def policyEval(S,A,T,R,gamma,pi,s_term):
    """
        returns value function for a given policy pi
    """
    T_pi = T[np.arange(S),pi]
    R_pi = R[np.arange(S),pi]
    
    V = np.zeros((S,1))
    
    MASK = np.ones(S, dtype=bool)
    MASK[s_term] = False
    V[MASK] = np.linalg.solve((np.identity(S) - gamma*T_pi)[MASK][:,MASK], np.sum(T_pi*R_pi, axis=1, keepdims=True)[MASK])
    return V

def howardPolicyIteration(S,A,T,R,gamma,s_term):
    """
        return optimal policy pi for a given MDP(S,A,T,R,gamma)
    """
    
    np.random.seed(42)
    pi = np.random.randint(low=0,high=A,size=S) # initialise with a random policy
    while True:
        V_pi = policyEval(S,A,T,R,gamma, pi, s_term) 
        temp_mat = gamma*V_pi.reshape((1,1,-1))
        Q = np.sum(T*(R + temp_mat), axis=2)
        improvable_matrix = Q-V_pi>0.0000001

        if (np.logical_not(improvable_matrix)).all():
            return pi
        
        im_S,im_A = np.where(improvable_matrix)

        improvable_states = np.unique(im_S)

        for s in improvable_states:
            pi[s]=np.random.choice(im_A[im_S==s])
        
def linearProgramming(S,A,T,R,gamma,s_term):
    """
        return optimal value function for a given MDP(S,A,T,R,gamma)
    """
    v = pl.LpVariable.dicts ("s", (range (S)))
    prob = pl.LpProblem ("ValueLP", pl.LpMaximize) # Maximise the objective function
    prob += sum ([-v [i] for i in range (S)]) # the objective function
    
    # Inequality constraints
    for s0 in range (S):
        for ac in range(A):
            prob += v[s0] >= sum (T[s0,ac,s1]*(R[s0,ac,s1]+ gamma*v[s1]) for s1 in range(S))

    # Equality constraints for terminal states
    for s in s_term:
            for a in range(A):
                prob += v[s] == 0

    prob.solve(pl.PULP_CBC_CMD(msg=0))
    
    V = np.zeros ((S,1))
    for i in range (S):
        V[i] = v[i]. varValue
    return V

if __name__ == "__main__":

    parser.add_argument("--mdp", type=str, required=True)
    parser.add_argument("--algorithm",type=str, default="hpi")
    parser.add_argument("--policy", type=str, required=False)
     
    args = parser.parse_args()

    mdp_file_path = args.mdp 
    algo = args.algorithm
    pol_eval = args.policy != None
    pol_file_path = args.policy 

    S,A,T,R,gamma,mdp_type,s_term = readMDPfile(mdp_file_path)

    if pol_eval:
        pol = readPolicyfile(pol_file_path)
        V = policyEval(S,A,T,R,gamma, pol, s_term)       
        for v,a in zip(V,pol):
            print("{:.6f} {}".format(v.item(),a.item()))
    else:
        if algo == "vi":
            V = valueIteration(S,A,T,R,gamma)
            pi_optimal = np.argmax(np.sum(T*(R + gamma*V.reshape((1,1,-1))), axis=2), axis=1).reshape((-1,1))
        elif algo == "hpi":
            pi_optimal = howardPolicyIteration(S,A,T,R,gamma,s_term)
            V = policyEval(S,A,T,R,gamma, pi_optimal, s_term) 
        elif algo == "lp":
            V = linearProgramming(S,A,T,R,gamma,s_term)
            pi_optimal = np.argmax(np.sum(T*(R + gamma*V.reshape((1,1,-1))), axis=2), axis=1).reshape((-1,1))

        for v,a in zip(V,pi_optimal):
            print("{:.6f} {}".format(v.item(),a.item()))
    

