# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:39:36 2021

@author: marco
"""

import numpy as np 
import matplotlib.pyplot as plt

def dx(theta,thetap):
    u=1.0
    m=1.0
    lc=0.3
    b=0.2
    g=9.81
    I=1.0
    derivate=np.array([[thetap ],[(u-m*g*(lc)*np.sin(theta)-b*thetap)/(m*lc**2+I)]])
    return derivate
h=float(input("Tamaño de paso: "))
s=float(input("Valor maximo de la evaluacion:"))
#Número de muestras
n=(s/h)+1
x=np.zeros(shape=(2,int(n)))
t=np.zeros(int(n))
for i in range(1,int(n)):
    t[i]=t[i-1]+h
    x[:,[i]]=+0.5*dx(x[0,i-1],x[1,i-1])

plt.plot(t,x[0,:])

    

