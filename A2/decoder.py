#! /usr/bin/python
import argparse
import numpy as np

parser = argparse.ArgumentParser()

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

if __name__ == "__main__":

    parser.add_argument("--value-policy", type=str, required=True)
    parser.add_argument("--opponent", type=str, required=True)

    args = parser.parse_args()

    value_po_file_path = args.value_policy
    opponent_file_path = args.opponent
    

    with open(opponent_file_path,'r') as file:
        opponent_lines = file.readlines()

    with open(value_po_file_path,'r') as file:
        value_po_lines = file.readlines()

    for opp_line in opponent_lines[1:]:
        line_l = opp_line.split(" ")

        strState = line_l[0]
        B1,B2,R,P = strState2pos(strState)
        indState = pos2StateInd(B1,B2,R,P)
        v_a = value_po_lines[indState].rstrip().split()
        print(strState, v_a[1], v_a[0])
        # raise
    