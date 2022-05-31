# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 13:52:34 2021

@author: marco
"""

import numpy as np 
import matplotlib.pyplot as plt
def yprima(x,y):
    derivate=x+2*y
    return derivate
h=float(input("Tamaño de paso: "))
s=float(input("Valor maximo de la evaluacion:"))
#Número de muestras
n=(s/h)+1
x=np.zeros(int(n))
y=np.zeros(int(n))
x0=0
y0=0
t=np.zeros(int(n))
x[0]=x0
y[0]=y0
for i in range(1,int(n)):
    t[i]
    x[i]=x[i-1]+h
    y[i]=y[i-1]+h*yprima(x[i-1],y[i-1])
    print(x[i],y[i])
plt.scatter(x,y)
0.25