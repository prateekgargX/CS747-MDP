import os
from signal import pause
import sys
import random 
import json
import math
import utils
import time
import config
import numpy as np
random.seed(73)

def anglexy(x,y):
    """
    returns angle(-1,1) of point x,y
    """
    if x<=0 and y<=0:
        theta = np.arctan(x/y)
    elif x<=0 and y>0:
        theta = np.pi - np.arctan(-x/y)
    elif x>0 and y>0:
        theta =  - np.pi + np.arctan(x/y)
    elif x>0 and y<=0:
        theta = - np.arctan(-x/y)
    return theta/np.pi

def length_xy(v):
    return np.linalg.norm(v)

class Agent:
    def __init__(self, table_config) -> None:
        self.table_config = table_config
        self.prev_action = None
        self.curr_iter = 0
        self.state_dict = {}
        self.holes =[]
        self.ns = utils.NextState()
        self.num_last_ball = None

    def set_holes(self, holes_x, holes_y, radius):
        for x in holes_x:
            for y in holes_y:
                self.holes.append((x[0], y[0]))
        self.ball_radius = radius


    def action(self, ball_pos=None):
        vec_w = np.array(ball_pos["white"]) # getting the coordinates of cue ball
       
        # removing the cue ball from position list
        del ball_pos["white"]
        del ball_pos[0]

        ball_vecs = np.array(list(ball_pos.values())) # Getting the vectors of balls
        hole_vecs = np.array(self.holes)              # Getting the vectors of holes
        
        hole2ball_vecs = ball_vecs[:,None,:] - hole_vecs[None,:,:] # pairwise vectors between ball and holes 
        
        hit_points_vecs = ball_vecs[:,None,:]+2*self.ball_radius*(hole2ball_vecs/np.linalg.norm(hole2ball_vecs, axis=-1, keepdims=True))
        cue2hit_points_vecs = hit_points_vecs - vec_w[None,None,:]
        cue2ball_vecs = ball_vecs[:,None,:] - vec_w[None,None,:]
        # print(.shape)

        DOT_PRODUCT_TOL = 0
        cos_similarity = (-hole2ball_vecs * cue2ball_vecs).sum(axis=-1)/(np.linalg.norm(hole2ball_vecs, axis=-1) * np.linalg.norm(cue2ball_vecs, axis=-1))
        hittable_holes = cos_similarity > DOT_PRODUCT_TOL

        valid_holes_dict = {}
        best_b_so_far = 0
        best_h_so_far = 0
        best_dist_so_far = 100000
        LAMBDA = 10
        for b in range(ball_vecs.shape[0]):
            ball_h = []
            ball_h_dist = []
            for h in range(hole_vecs.shape[0]):
                if hittable_holes[b,h]: 
                    ball_h.append(h)
                    ball_h_dist.append(np.linalg.norm(hole2ball_vecs[b,h])+LAMBDA*(1 - cos_similarity[b,h]))
            if len(ball_h_dist) > 0:
                bmindist = min(ball_h_dist)
                ball_h_i = ball_h_dist.index(bmindist)
                if bmindist < best_dist_so_far:
                    best_dist_so_far = bmindist
                    best_b_so_far = b
                    best_h_so_far = ball_h_i
                valid_holes_dict[b] = ball_h_i 
            else:
                valid_holes_dict[b] = None
        EPS = 0.9
        coin_flip = np.random.binomial(1,EPS)
        if coin_flip == 1:
            b = best_b_so_far
            h = best_h_so_far 
        else:
            b = np.random.randint(ball_vecs.shape[0])
            h = valid_holes_dict[b] 
            if h == None:
                h = np.random.randint(hole_vecs.shape[0])

        b_vec = ball_vecs[b]
        h_vec = hole_vecs[h]
        
        hit_vec = b_vec+2*self.ball_radius*(b_vec - h_vec)/np.linalg.norm((b_vec - h_vec))
        target_x,target_y =hit_vec - vec_w
        angle = anglexy(target_x,target_y)
        
        FORCE_FACTOR = 1.1
        force = FORCE_FACTOR*(length_xy(hit_vec - vec_w) + (length_xy(h_vec - b_vec)/config.ball_coeff_of_restitution))/(960*1.414)
        
        return (angle, force)
