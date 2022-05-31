# -*- coding: utf-8 -*-
"""
Created on Tue Sep 28 10:01:35 2021

@author: marco
"""



import numpy as np


def limobjetive(ise,iadu):
    if(ise>=0.1):
        ie=0.1
    else:
        ie=ise
    if(iadu>=0.8):
        ia=0.8
    else:
        ia=iadu
    return np.array([ie,ia])


def limcontro(u):
    if(u>=1.4):
        ur=1.4
    elif(u<=0.0035):
        ur=0.0035
    else:
        ur=u
    return ur
    
def pendulum_s(r):
    '''Time Parameters'''
    dt = 0.005  # Tiempo de muestreo (5ms)
    ti = 0.0  # Tiempo inicial de la simulación (0s)
    tf = 10.0  # Tiempo inicial de la simulación (10s)
    n = int((tf - ti) / dt) + 1  # Número de muestras
    t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
    
    '''Dynamics Parameters'''
    m = 0.5  # Masa del pendulo (kg)
    l = 1.0  # Longitud de la barra del péndulo (m)
    lc = 0.3  # Longitud al centro de masa del péndulo (m)
    b = 0.05  # Coeficiente de fricción viscosa pendulo
    g = 9.81  # Aceleración de la gravedad en la Tierra
    I = 0.006  # Tensor de inercia del péndulo

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
        
        u = Kp * e_th + Kd * e_th_dot + Ki * ie_th
        u[o]=limcontro(u)
        
        '''System dynamics'''
        xdot[0] = th_dot
        xdot[1] = (u[o] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)
        
        '''Integrate dynamics'''
        x[o + 1, 0] = x[o, 0] + xdot[0] * dt
        x[o + 1, 1] = x[o, 1] + xdot[1] * dt
        ie_th = ie_th + e_th * dt
        
        ise=ise_next+(e_th**2)*dt
        iadu=iadu_next+ (abs(u[o]-u[o-1]))*dt
        lim=limobjetive(ise,iadu)
        ise_next=lim[0]
        iadu_next=lim[1]
       
    
    return np.array([ise_next, iadu_next])

k=np.array([0,1,2])

pen=pendulum_s(k)