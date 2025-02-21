#! /usr/bin/python
import argparse
from turtle import right
import numpy as np

parser = argparse.ArgumentParser()

def DEBUG(*x):
    print("DEBUG: ",x)
    raise

def strState2pos(s):
    B1 = int(s[0:2]) 
    B2 = int(s[2:4])
    R  = int(s[4:6])
    P  = int(s[6:7])

    return B1,B2,R,P

def pos2xy(p):
    return (p-1)%4+1, (p-1)//4+1

def pos2StateInd(B1,B2,R,P):
    if B1 == 0 or B2 == 0:
        return 0
    if P == 0:
        return 8193
    return 32*16*(B1-1) + 32*(B2-1) + 2*(R - 1)+ P

def xy2pos(x,y):
    return x+4*(y-1)

def validxy(x,y):
    return x>0 and x<5 and y>0 and y<5

def middleOf2Points(B1,B2):
    x1,y1 = pos2xy(B1)
    x2,y2 = pos2xy(B2)
    points = []
    if x1==x2:
        sy = min(y2,y1)
        ey = max(y2,y1)
        for y in range(sy,ey+1):
            points.append(xy2pos(x1,y))
    elif y1==y2:
        sx = min(x2,x1)
        ex = max(x2,x1)
        for x in range(sx,ex+1):
            points.append(xy2pos(x,y1))
    elif x1 - y1 == x2 - y2:
        a = x1 - y1
        sx = min(x2,x1)
        ex = max(x2,x1)
        for x in range(sx,ex+1):
            points.append(xy2pos(x,x-a))
    elif x1 + y1 == x2 + y2:
        a = x1 + y1
        sx = min(x2,x1)
        ex = max(x2,x1)
        for x in range(sx,ex+1):
            points.append(xy2pos(x,a-x))
    return points

if __name__ == "__main__":

    parser.add_argument("--opponent", type=str, required=True)
    parser.add_argument("--p",type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
     
    args = parser.parse_args()

    opponent_file_path = args.opponent
    p = args.p
    q = args.q

    with open(opponent_file_path,'r') as file:
        lines = file.readlines()


    S = 8194  # number of states
    A = 10  # number of actions

    gamma = 1 # discount factor
    mdptype = "episodic" # MDP type

    T_f = np.zeros((S,A,S)) # initializing the transition function with all zeros
    R_f = np.zeros((S,A,S)) # initializing the reward function with all zeros

    R_f[:,9,8193] = 1
    
    s_term = [0,8193]

    for s in s_term:
        T_f[s,:,s] = 1
        R_f[s,:,s] = 0

    for line in lines[1:]:
        line_l = line.split(" ")
        strState = line_l[0]
        p_l = float(line_l[1])
        p_r = float(line_l[2])
        p_u = float(line_l[3])
        p_d = float(line_l[4])

        p_list = [p_l,p_r,p_u,p_d]

        B1_s0,B2_s0,R_s0,P_s0 = strState2pos(strState)
        
        B1_s0_x,B1_s0_y = pos2xy(B1_s0)
        B2_s0_x,B2_s0_y = pos2xy(B2_s0)
        R_s0_x,R_s0_y = pos2xy(R_s0)
        s0 = pos2StateInd(B1_s0,B2_s0,R_s0,P_s0)
        R_s1_list = [xy2pos(R_s0_x-1,R_s0_y),xy2pos(R_s0_x+1,R_s0_y),xy2pos(R_s0_x,R_s0_y-1),xy2pos(R_s0_x,R_s0_y+1)]
        
        valid_p_R_s1_list = [(p_list[i],R_s1_list[i]) for i in range(len(p_list)) if p_list[i] > 0]
        for a in range(10):
            if a < 4:
                # B1 moves
                if a == 0: B1_s1_x,B1_s1_y = B1_s0_x-1,B1_s0_y
                if a == 1: B1_s1_x,B1_s1_y = B1_s0_x+1,B1_s0_y
                if a == 2: B1_s1_x,B1_s1_y = B1_s0_x,B1_s0_y-1
                if a == 3: B1_s1_x,B1_s1_y = B1_s0_x,B1_s0_y+1
                B1_s1 = xy2pos(B1_s1_x,B1_s1_y)
                B2_s1 = B2_s0
                P_s1 = P_s0
            if 4 <= a < 8:
                # B2 moves
                if a == 4: B2_s1_x,B2_s1_y = B2_s0_x-1,B2_s0_y
                if a == 5: B2_s1_x,B2_s1_y = B2_s0_x+1,B2_s0_y
                if a == 6: B2_s1_x,B2_s1_y = B2_s0_x,B2_s0_y-1
                if a == 7: B2_s1_x,B2_s1_y = B2_s0_x,B2_s0_y+1
                B2_s1 = xy2pos(B2_s1_x,B2_s1_y)
                B1_s1 = B1_s0
                P_s1 = P_s0    
            if a == 8:
                # attempt to pass ball to teammate
                P_s1 = 2 if P_s0 == 1 else 1
                B1_s1 = B1_s0
                B2_s1 = B2_s0
            if a == 9:
                # attempt a shot to the goal
                B1_s1 = -1
                B2_s1 = -1
                P_s1 = 0

            for p_,R_s1 in valid_p_R_s1_list:
                s1 = pos2StateInd(B1_s1,B2_s1,R_s1,P_s1)    
                a_valid = True
                if 0 <= a < 4:
                    if validxy(B1_s1_x,B1_s1_y):
                        if P_s1 == 1:
                            if B1_s1 == R_s1 or (B1_s1 == R_s0 and B1_s0 == R_s1): # tackling
                                p_term = 0.5 + p
                            else:
                                p_term = 2*p
                        else:
                            p_term = p
                    else:
                        p_term = 1
                        a_valid = False
                elif 4 <= a < 8:
                    if validxy(B2_s1_x,B2_s1_y):
                        if P_s1 == 2:
                            if B2_s1 == R_s1 or (B2_s1 == R_s0 and B2_s0 == R_s1): # tackling
                                p_term = 0.5 + p
                            else:
                                p_term = 2*p
                        else:
                            p_term = p
                    else:
                        p_term = 1
                        a_valid = False
                elif a == 8:
                    # passing
                    if R_s1 in middleOf2Points(B1_s1,B2_s1):
                        p_term = 1 - 0.5*(q - 0.1*max(abs(B1_s0_x-B2_s0_x),abs(B1_s0_y-B2_s0_y)))
                    else:  
                        a_temp = 0.1*max(abs(B1_s0_x-B2_s0_x),abs(B1_s0_y-B2_s0_y))
                        p_term = 1 - (q - a_temp)
                else:
                    # shooting the goal
                    x_dist = B1_s0_x if P_s0 == 1 else B2_s0_x
                    if R_s1 in [8,12]:
                        p_term = 1 - 0.5*(q - 0.2*(4 - x_dist))
                    else:
                        p_term = 1 - (q - 0.2*(4 - x_dist))
                if a_valid:
                    T_f[s0,a,s1] += p_*(1-p_term)
                T_f[s0,a,0]  += p_*p_term


    print("numStates",S)
    print("numActions",A)
    print("end 0 8193")

    for s0 in range(0, S):
        for a in range(0, A):
            s1_distribution_s0_a = T_f[s0,a]
            s1_list = list(np.nonzero(s1_distribution_s0_a)[0])            
            for s1 in s1_list:
                print("transition",s0,a,s1,R_f[s0,a,s1],s1_distribution_s0_a[s1])

    print("mdptype",mdptype)
    print("discount ",gamma)