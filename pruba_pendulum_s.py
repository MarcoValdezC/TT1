# -*- coding: utf-8 -*-
"""
Created on Fri Oct  1 11:30:01 2021

@author: marco
"""

from platypus import *
import random
import numpy as np 
import math
import matplotlib.pyplot as plt
import serial  
from drawnow import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# def belegundu(vars):
#     x = vars[0]
#     y = vars[1]
#     return [-2*x + y, 2*x + y], [-x + y - 1, x + y - 7]


# def limobjetive(ise,iadu):
#     if(ise>=10):
#         ie=5
#     else:
#         ie=ise
#     if(iadu>=0.8):
#         ia=0.8
#     else:
#         ia=iadu
#     return np.array([ie,ia])


# def limcontro(u):
#     if(u>=0):
#         if(u>=1.4):
#             ur=1.4
#         elif(u<=0.0035):
#             ur=0.0035
#         else:
#             ur=u
#     else:
#         if(u>=-0.0035):
#             ur=-0.0035
#         elif(u<=-1.4):
#             ur=-1.4
#         else:
#             ur=u
    
#     return ur

def pendulum_s(r):
    

    '''State variables'''
    x = np.zeros((n, 2))

    '''Control vector'''
    u = np.zeros((n, 1))
    
    
    ise=0
    ise_next=0
    iadu=0
    iadu_next=0
    
    '''Initial conditions'''
    x[0, 0] = 0  # Initial pendulum position (rad)
    x[0, 1] = 0  # Initial pendulum velocity (rad/s)
    ie_th = 0

    '''State equation'''
    xdot = [0, 0]

    '''Dynamic simulation'''
    for o in range(n - 1):
        '''Current states'''
        th = x[o, 0]
        th_dot = x[o, 1]
        e_th =np.pi-th
        e_th_dot = 0 - th_dot
        
        '''Controller'''
        Kp =r[0]
        Kd =r[1]
        Ki =r[2]
        
        u[o,0]= (Kp * e_th + Kd * e_th_dot + Ki * ie_th)
        # if(u>=0):
        #     if(u>=1.4):
        #         u[o,0]=1.4
        #     elif(u<=0.0035):
        #         u[o,0]=0.0035
        #     else:
        #         ur=u
        # else:
        #     if(u>=-0.0035):
        #         u[o,0]=-0.0035
        #     elif(u<=-1.4):
        #         u[o,0]=-1.4
        #     else:
        #         u[o,0]=u
    

 
        
        
        '''System dynamics'''
        xdot[0] = th_dot
        xdot[1] = (u[o] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)
        
        '''Integrate dynamics'''
        x[o + 1, 0] = x[o, 0] + xdot[0] * dt
        x[o + 1, 1] = x[o, 1] + xdot[1] * dt
        ie_th = ie_th + e_th * dt
        
        ise=ise_next+(e_th**2)*dt
        iadu=iadu_next+ (abs(u[o,0]-u[o-1,0]))*dt
        if(ise>=0.1):
            ie=0.1
        else:
            ie=ise
        if(iadu>=0.8):
            ia=0.8
        else:
            ia=iadu
   
        ise_next=ise
        iadu_next=ia
       
    
    return [ise_next], [ise_next]

problem = Problem(3, 2)
problem.types[:] = [Real(0, 10), Real(0, 10), Real(0,10)]

problem.function =pendulum_s

algorithm = (NSGAII(problem))
algorithm.run(50)

nondominated_solutions = nondominated(algorithm.result)

plt.scatter([s.objectives[0] for s in nondominated_solutions],
            [s.objectives[1] for s in nondominated_solutions])
#plt.xlim([0, 10])
#plt.ylim([0, 5])
plt.xlabel("$f_1(x)$")
plt.ylabel("$f_2(x)$")
plt.show()