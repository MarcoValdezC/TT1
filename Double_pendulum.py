# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 20:04:59 2021

@author: marco
"""

import os
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

'''Time parameters''' #Parametros temporales
dt = 0.005  # Tiempo de muestreo (5ms)
ti = 0.0  # Tiempo inicial de la simulación (0s)
tf =2*np.pi  # Tiempo final de la simulación (12.25s)
n = int((tf - ti) / dt) + 1  # Número de muestras
t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)

'''Dynamic parameters''' #Parametros dinamicos
m1 = 0.5  # Masa de la barra 1(kg)
m2= 0.5 #Masa de la barra 2 (kg)
l1 = 1.0  # Longitud de la barra 1 (m)
lc1 = 0.5  # Longitud al centro de masa de la barra 2 (m)
l2= 1.0 #.0Longitud de la baraa 2 (m)
lc2=0.3 #Longitud al centro de masa de la barra 2(m)
b1 = 0.05  # Coeficiente de fricción viscosa de la barra 1
b2= 0.02 #Coeficiente de fricción viscosa de la barra 2
g = 9.81  # Aceleración de la gravedad en la Tierra
I1 = 0.006  # Tensor de inercia del péndulo 1
I2= 0.004 #Tensor de inercia del péndulo 2

''' Cinematica inversa'''

r=2
ro=r*np.cos(3*t+np.pi/2)

'''Ecuaciones paramétricas de rosa de 3 pétalos'''
Xp = ro*np.cos(t)
Yp = ro*np.sin(t)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculo del Modelo Cinematico Inverso de Posicion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#Variable Articular 2
teta_rad_inv2 =np.arccos((Xp**2+Yp**2-(np.multiply(l1,l1)+np.multiply(l2,l2)))/(2*l1*l2))
teta_grad_inv2=teta_rad_inv2*180/np.pi


#Variable Articular 1 
alfa=np.arctan2(Xp,Yp)
beta=np.arccos((np.multiply(l1,l1)+np.multiply(l2,l2)-(np.multiply(Xp,Xp)+np.multiply(Yp,Yp)))/(2*l1*l2))
gamma=np.arcsin((l2*np.sin(beta))/np.sqrt(np.multiply(Xp,Xp)+np.multiply(Yp,Yp)))

if (teta_rad_inv2<0).all():
    teta_rad_inv1=alfa+gamma
else:
    teta_rad_inv1=alfa-gamma

teta_grad_inv1=teta_rad_inv1*180/np.pi
'''Cinematica Diferencial inversa'''
dx=-r*(3*np.sin(3*t)*np.cos(t)+np.cos(3*t)*np.sin(t))
dy=-r*(3*np.sin(3*t)*np.sin(t)-np.cos(3*t)*np.cos(t))

t1_dot=((np.sin(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*dx)-((np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*dy)
t2_dot=-(((l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*dx)+(((l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*dy)
'''State variables'''#Variables de estado
x = np.zeros((n, 4))

'''Control vector'''#Señales de control
u = np.zeros(( 2,n))


'''Initial conditions'''#Condiciones iniciales
x[0, 0] =np.pi/2# Initial pendulum position 1 (rad)
x[0, 1] =0# Initial pendulum position 2 (rad)
x[0, 2]=0 # Initial pendulum velocity (rad/s)
x[0, 3]=0 # Initial pendulum velocity (rad/s)

ie_th1 = 0
ie_th2 = 0

'''State equation'''#Ecuacion de estado
xdot = [0, 0, 0, 0]

'''Dynamic simulation'''
for i in range(n - 1):
    '''Current states'''
    th1 = x[i, 0]
    th2 = x[i, 1]
    th1_dot=x[i,2]
    th2_dot=x[i,3]
    
    '''Controller'''
    e_th1 = teta_rad_inv1[i-1] - th1
    e_th1_dot = t1_dot[i]- th1_dot
    
    
    e_th2 = teta_rad_inv2[i-1] - th2
    
    
    e_th2_dot =t2_dot[i] - th2_dot

    Kp = 1
    Kd = 0.01
    Ki = 0
    
    Kp2 = 1
    Kd2 = 0.01
    Ki2 = 0

    u[0,i] = Kp * e_th1 + Kd * e_th1_dot + Ki * ie_th1
    u[1,i] = Kp2 * e_th2 + Kd2 * e_th2_dot + Ki2 * ie_th2
    
    '''Propiedades del modelo dinámico'''
    #Efecto inercial
    M=np.array([[(m1*lc1**2)+I1+I2+m2*((l1**2)+(lc2**2)+(2*l1*lc2*np.cos(th2))),(m2*lc2**2)+I2+m2*l1*lc2*np.cos(th2)],[(m2*lc2**2)+I2+m2*l1*lc2*np.cos(th2), (m2*lc2**2)+I2]])
   #Fuerzas centrípeta y de Coriolis
    C=np.array([[-2*m2*l1*lc2*th2_dot*np.sin(th2) + m1*b1 ,-m1*l1*lc2*np.sin(th2)*th2_dot],[m2*l1*lc2*th1_dot*np.sin(th2) , m2*b2]])
    #Aporte gravitacional
    gra=np.array([[m1*lc1*np.sin(th1)+m2*((l1*np.sin(th1))+lc2*np.sin(th1+th2))],[m2*lc2*np.sin(th1+th2)]])
    v=np.array([[th1_dot],[th2_dot]])
    C2=np.dot(C,v)
    ua=np.array([[u[0,i]],[u[1,i]]])
    aux1=ua-C2-gra
    Minv=linalg.inv(M)
    aux2=np.dot(Minv,aux1)
    '''System dynamics'''
    xdot[0] = th1_dot
    xdot[1]= th2_dot
    xdot[2]=aux2[0,:]
    xdot[3]=aux2[1,:]
    '''Integrate dynamics'''
    x[i + 1, 0] = x[i, 0] + xdot[0] * dt
    x[i + 1, 1] = x[i, 1] + xdot[1] * dt
    x[i + 1, 2] = x[i, 2] + xdot[2] * dt
    x[i + 1, 3] = x[i, 3] + xdot[3] * dt
    
    ie_th1 = ie_th1 + e_th1 * dt
    ie_th2 = ie_th2 + e_th2 * dt

print(x[:, 0])
print(x[:, 1])
X2=l1*np.sin(x[:,0])+l2*np.sin(x[:, 0]+x[:, 1])
Y2=-l1*np.cos(x[:, 0])-l2*np.cos(x[:, 0]+x[:, 1])
'''Plotting results'''  
plt.figure(figsize=(12, 10))
plt.subplot(321)
plt.plot(X2,Y2, 'k', lw=1)
plt.legend([r'$\theta_1$'], loc=1)
plt.ylabel('Pendulum position')
plt.xlabel('Time')

plt.subplot(322)
plt.plot(t, x[:, 1], 'b', lw=1)
plt.legend([r'$\theta_2$'], loc=1)
plt.ylabel('Pendulum position')
plt.xlabel('Time')

plt.subplot(323)
plt.plot(t, x[:, 2], 'c', lw=2)
plt.legend([r'$\dot{\theta_1}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(324)
plt.plot(t, x[:, 3], 'g', lw=2)
plt.legend([r'$\dot{\theta_2}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(325)
plt.plot(t, u[0, :], 'b', lw=1)
plt.legend([r'$\upsilon_1$'], loc=1)
plt.ylabel('V')
plt.xlabel('Time')

plt.subplot(326)
plt.plot(t, u[1, :], 'b', lw=1)
plt.legend([r'$\upsilon_2$'], loc=1)
plt.ylabel('V')
plt.xlabel('Time')

'''Plotting results'''
x0=np.zeros(len(t))
y0=np.zeros(len(t))

x1=l1*np.sin(x[:,0])
y1=-l1*np.cos(x[:,0])

x2=l2*np.sin(x[:,0])+ l2*np.sin(x[:,0]+x[:,1])
y2=-l2*np.cos(x[:,0])- l2*np.cos(x[:,0]+x[:,1])

fig = plt.figure(figsize=(8,6.4))
ax = fig.add_subplot(111,autoscale_on=False,\
                     xlim=(-2.8,2.8),ylim=(-2.2,2.2))
ax.set_xlabel('position')

line, = ax.plot([],[],'o-',color='blue',lw=4,\
                markersize=6,markeredgecolor='k',\
                markerfacecolor='k')
line1, = ax.plot([],[],'o-',color='blue',lw=4,\
                markersize=6,markeredgecolor='k',\
                markerfacecolor='k')
time_template = 't= %.1fs'
time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)


def init():
    line.set_data([],[])
    line1.set_data([],[])
    time_text.set_text('')
    
    return line, time_text, line1,

def animate(i):
   
    line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
    line1.set_data([x1[i],x2[i]],[y1[i],y2[i]])
    time_text.set_text(time_template % t[i])
    
    return line, time_text, line1,

ani_a = animation.FuncAnimation(fig, animate, \
         np.arange(1,len(t)), \
         interval=10,blit=False,init_func=init)

# requires ffmpeg to save mp4 file
#  available from https://ffmpeg.zeranoe.com/builds/
#  add ffmpeg.exe to path such as C:\ffmpeg\bin\ in
#  environmen-t variables
#ani_a.save('Pendulum_Control.mp4',fps=30)

plt.show()

