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
from scipy.integrate import odeint 

import time
import math
import pylab as py


from matplotlib import animation, rc
from IPython.display import HTML
from matplotlib import pyplot as plt

'''Time parameters''' #Parametros temporales
dt = 0.005  # Tiempo de muestreo (5ms)
ti = 0.0  # Tiempo inicial de la simulación (0s)
tf =20  # Tiempo final de la simulación (12.25s)
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
gravi = 9.81  # Aceleración de la gravedad en la Tierra
I1 = 0.006  # Tensor de inercia del péndulo 1
I2= 0.004 #Tensor de inercia del péndulo 2

''' Cinematica inversa'''

r=0.2
#ro=r*np.cos(3*t)

'''Ecuaciones paramétricas de circunferencia'''
Xp =1.4+ r*np.cos(t)
Yp =0.2+ r*np.sin(t)
'''Ecuaciones paramétricas de rosa de 3 petalos
Xp =1.4+ ro*np.cos(t)
Yp =0.2+ ro*np.sin(t)   '''

#Ecuaciones pametricas lemniscata


'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Calculo del Modelo Cinematico Inverso de Posicion
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

#Variable Articular 2
cosq2=(Xp**2+Yp**2-l1**2-l2**2)/(2*l1*l2)
teta_rad_inv2=np.arctan2((1-cosq2**2)**(1/2),cosq2)

teta_rad_inv1= np.arctan2(Xp,-Yp)-np.arctan2(l2*np.sin(teta_rad_inv2),(l1+l2*np.cos(teta_rad_inv2)));

#teta_rad_inv2 =np.arccos((Xp**2+Yp**2-(l1**2+l2**2))/2*l1*l2)
teta_grad_inv2=teta_rad_inv2*180/np.pi


#Variable Articular 1 
'''alfa=np.arctan2(Xp,Yp)
beta=np.arccos((np.multiply(l1,l1)+np.multiply(l2,l2)-(np.multiply(Xp,Xp)+np.multiply(Yp,Yp)))/(2*l1*l2))
gamma=np.arcsin((l2*np.sin(beta))/np.sqrt(np.multiply(Xp,Xp)+np.multiply(Yp,Yp)))
'''

#teta_rad_inv1=np.arctan2(-Yp,Xp)-np.arctan2(l2*np.sin(teta_rad_inv2),l1+l2*np.cos(teta_rad_inv2))
teta_grad_inv1=teta_rad_inv1*180/np.pi
'''Cinematica Diferencial inversa'''
#Rosa de 3 petalos 
#dx=-r*(3*np.sin(3*t)*np.cos(t)+np.cos(3*t)*np.sin(t))
#dy=-r*(3*np.sin(3*t)*np.sin(t)-np.cos(3*t)*np.cos(t))
#Circunferencia 
dx=-r*np.sin(t)
dy=r*np.cos(t)

t1_dot=((np.sin(teta_rad_inv1+teta_rad_inv2))/(l1*np.sin(teta_rad_inv2))*dx)-((np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*dy)
t2_dot=-(((l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*dx)+(((l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*dy)
'''Cinematica Aceleración Inversa'''
#Aceleración circunferencia
ddx=-r*np.cos(t)
ddy=-r*np.sin(t)

#Aceleración rosa de 3 petalos
#ddx=-r*(10*np.cos(3*t)*np.cos(t)-6*np.sin(3*t)*np.sin(t))
#ddy=-r*(10*np.cos(3*t)*np.sin(t)+6*np.sin(3*t)*np.cos(t))

#Jacobiano inverso 
#Jinv= [[(np.sin(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2)), (-np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))],[-(l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2),(l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2)]]
#Jt=np.array([[-r*np.cos(t)-(t1_dot**2*(-l1*np.sin(teta_rad_inv1)-l2*np.sin(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.sin(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.sin(teta_rad_inv1+teta_rad_inv2)))],[-r*np.sin(t)-(t1_dot**2*(l1*np.cos(teta_rad_inv1)-l2*np.cos(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.cos(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.cos(teta_rad_inv1+teta_rad_inv2)))]])

t1_ddot=((np.sin(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2))*(-r*np.cos(t)-(t1_dot**2*(-l1*np.sin(teta_rad_inv1)-l2*np.sin(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.sin(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.sin(teta_rad_inv1+teta_rad_inv2)))))+(((-np.cos(teta_rad_inv1+teta_rad_inv2)/l1*np.sin(teta_rad_inv2)))*(-r*np.sin(t)-(t1_dot**2*(l1*np.cos(teta_rad_inv1)-l2*np.cos(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.cos(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.cos(teta_rad_inv1+teta_rad_inv2)))))
t2_ddot=((-(l1*np.sin(teta_rad_inv1)+l2*np.sin(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*(-r*np.cos(t)-(t1_dot**2*(-l1*np.sin(teta_rad_inv1)-l2*np.sin(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.sin(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.sin(teta_rad_inv1+teta_rad_inv2)))))+(((l1*np.cos(teta_rad_inv1)+l2*np.cos(teta_rad_inv1+teta_rad_inv2))/l1*l2*np.sin(teta_rad_inv2))*(-r*np.sin(t)-(t1_dot**2*(l1*np.cos(teta_rad_inv1)-l2*np.cos(teta_rad_inv1+teta_rad_inv2))-2*t1_dot*t2_dot*l2*np.cos(teta_rad_inv1+teta_rad_inv2)-t2_dot**2*(l2*np.cos(teta_rad_inv1+teta_rad_inv2)))))

'''State variables'''#Variables de estado
x = np.zeros((n, 4))

'''Control vector'''#Señales de control
u = np.zeros(( 2,n))


'''Initial conditions'''#Condiciones iniciales
x[0, 0] =np.pi/2# Initial pendulum position 1 (rad)
x[0, 1] =0# Initial pendulum position 2 (rad)
x[0, 2]=0 # Initial pendulum velocity (rad/s)
x[0, 3]=0 # Initial pendulum velocity (rad/s)

th1_ddot=np.zeros(n)

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
    M=np.array([[(m1*lc1**2)+I1+I2+m2*((l1**2)+(lc2**2)+(2*l1*lc2*np.cos(th2))),(m2*lc2**2)+I2+m2*l1*lc2*np.cos(th2)],[(m2*lc2**2)+I2+m2*l1*lc2*np.cos(th2), (m2*lc2**2)+I2]])
    #Fuerzas centrípeta y de Coriolis
    C=np.array([[-2*m2*l1*lc2*th2_dot*np.sin(th2) +b1 ,-m2*l1*lc2*np.sin(th2)*th2_dot],[m2*l1*lc2*th1_dot*np.sin(th2) , b2]])
    #Aporte gravitacional
    gra=np.array([[m1*lc1*gravi*np.sin(th1)+m2*gravi*(l1*np.sin(th1)+lc2*np.sin(th1+th2))],[m2*lc2*gravi*np.sin(th1+th2)]])
    
    e_th1 = teta_rad_inv1[i] - th1
    e_th1_dot =t1_dot[i]- th1_dot
    
    e_th2 = teta_rad_inv2[i]- th2
    e_th2_dot =t2_dot[i]- th2_dot

    Kp =1.973480764	
#9.91102555#10#5#10 #3.60614907877409#5.7255997347206#10
    Kd =3.773281867	#9.91102555#10#0.973324679473922#5 #0.503359674635035#1.96901831751399#5
    #Ki = 
    
    Kp2 =2.429151687	#9.91102555#10#4.93017386912806#5#3.60614907877409#5.7255997347206#5
    Kd2 =1.832728859#9.91102555#5#0.347734270091561#0.1#0.503359674635035#0.5554397672254#0.1
    #Ki2 = 0

    u[0,i] = Kp * e_th1 + Kd * e_th1_dot +M[0,0]*t1_ddot[i]+M[0,1]*t2_ddot[i]+C[0,0]*t1_dot[i]+C[0,1]*t2_dot[i]+gra[0,0]
    u[1,i] = Kp2 * e_th2 + Kd2 * e_th2_dot +M[1,0]*t1_ddot[i]+M[1,1]*t2_ddot[i]+C[1,0]*t1_dot[i]+C[1,1]*t2_dot[i]+gra[1,0]
    
    '''Propiedades del modelo dinámico'''
    #Efecto inercial
   
    v=np.array([[th1_dot],[th2_dot]])
    C2=np.dot(C,v)
    ua=np.array([[u[0,i]],[u[1,i]]])
    aux1=ua-C2-gra
    Minv=linalg.inv(M)
    aux2=np.dot(Minv,aux1)
    xdot[0] = th1_dot
    xdot[1]= th2_dot
    xdot[2]=aux2[0,:]
    th1_ddot[i]=xdot[3]
    xdot[3]=aux2[1,:]
    '''Integrate dynamics'''
    x[i + 1, 0] = x[i, 0] + xdot[0] * dt
    x[i + 1, 1] = x[i, 1] + xdot[1] * dt
    x[i + 1, 2] = x[i, 2] + xdot[2] * dt
    x[i + 1, 3] = x[i, 3] + xdot[3] * dt
    
    ie_th1 = ie_th1 + e_th1 * dt
    ie_th2 = ie_th2 + e_th2 * dt
    ise=0
    iadu=0
    ise_next=0
    iadu_next=0
    
    ise=ise_next+(e_th1**2)*dt+(e_th2**2)*dt
    iadu=iadu_next+ (abs(u[0,i]-u[0,i-1]))*dt+(abs(u[1,i]-u[1,i-1]))*dt
    g=0
    if(ise>=10):
        ie=10
        g+=1
    else:
        ie=ise
        g+=0
    if(iadu>=10):
        ia=10
        g+=1
    else:
        ia=iadu
        g+=0
    if(g==2):
        print(g)
   
    ise_next=ie
    iadu_next=ia
    # print(ise_next)
    # print(iadu_next)



#print(x[:, 0])
#print(x[:, 1])



X2=l1*np.sin(x[:,0])+l2*np.sin(x[:, 0]+x[:, 1])
Y2=-l1*np.cos(x[:, 0])-l2*np.cos(x[:, 0]+x[:, 1])
'''Plotting results'''  
plt.figure(figsize=(20, 20))
plt.subplot(441)
plt.plot(t, x[:, 0], 'k', lw=1)
plt.legend([r'$\theta_1$'], loc=1)
plt.ylabel('Pendulum position ')
plt.xlabel('Time')

plt.subplot(442)
plt.plot(t, teta_rad_inv1, 'k', lw=1)
plt.legend([r'$\theta_1d$'], loc=1)
plt.ylabel('Pemdulum position d')
plt.xlabel('Time')

plt.subplot(443)
plt.plot(t, x[:, 1], 'b', lw=1)
plt.legend([r'$\theta_2$'], loc=1)
plt.ylabel('Pendulum position d')
plt.xlabel('Time')

plt.subplot(444)
plt.plot(t, teta_rad_inv2, 'r', lw=1)
plt.legend([r'$\theta_2d$'], loc=1)
plt.ylabel('Pendulum position')
plt.xlabel('Time')

plt.subplot(445)
plt.plot(t, x[:, 2], 'c', lw=2)
plt.legend([r'$\dot{\theta_1}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(446)
plt.plot(t, t1_dot, 'g', lw=2)
plt.legend([r'$\dot{\theta_1d}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(447)
plt.plot(t, x[:, 3], 'g', lw=2)
plt.legend([r'$\dot{\theta_2}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(448)
plt.plot(t, t2_dot, 'c', lw=2)
plt.legend([r'$\dot{\theta_2d}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(449)
plt.plot(t, u[0, :], 'b', lw=1)
plt.legend([r'$u_1$'], loc=1)
plt.ylabel('u')
plt.xlabel('Time')

plt.subplot(4,4,10)
plt.plot(t, u[1, :], 'b', lw=1)
plt.legend([r'$\upsilon_2$'], loc=1)
plt.ylabel('V')
plt.xlabel('Time')

plt.subplot(4,4,11)
plt.plot(t, th1_ddot, 'g', lw=2)
plt.legend([r'$\ddot{\theta_1}$'], loc=1)
plt.ylabel('Pendulum  acceleration')
plt.xlabel('Time')

plt.subplot(4,4,12)
plt.plot(t, t1_ddot, 'k', lw=2)
plt.legend([r'$\ddot{\theta_1d}$'], loc=1)
plt.ylabel('Pendulum  acceleration')
plt.xlabel('Time')

plt.subplot(4,4,13)
plt.plot(t, t2_ddot, 'r', lw=2)
plt.legend([r'$\ddot{\theta_2d}$'], loc=1)
plt.ylabel('Pendulum acceleration')
plt.xlabel('Time')

plt.subplot(4,4,14)
plt.plot(X2, Y2, 'r', lw=2)

plt.plot(Xp,Yp, 'k', lw=1)
plt.legend([r'Trayectoria', r'Trayectoria deseada'],loc=1)
plt.ylabel('y')
plt.xlabel('x')

plt.subplots_adjust(left=0.125,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.2, 
                    hspace=0.35)

'''Plotting results'''
x0=np.zeros(len(t))
y0=np.zeros(len(t))

x1=l1*np.sin(x[:,0])
y1=-l1*np.cos(x[:,0])

x2=l1*np.sin(x[:,0])+ l2*np.sin(x[:,0]+x[:,1])
y2=-l1*np.cos(x[:,0])- l2*np.cos(x[:,0]+x[:,1])

fig = plt.figure(figsize=(8,6.4))
ax = fig.add_subplot(111,autoscale_on=False,\
                     xlim=(-2.8,2.8),ylim=(-2.2,2.2))
ax.set_xlabel('position')

'''line, = ax.plot([],[],'o-',color='blue',lw=4,\
                markersize=6,markeredgecolor='k',\
                markerfacecolor='k',markevery=10000)'''
#line,  = ax.plot([], [], 'o-',markerfacecolor = 'b',lw=4,markersize = 15,markeredgecolor = 'k',color = 'k',markevery=10000)
'''line1, = ax.plot([],[],'o-',color='blue',lw=4,\
                markersize=6,markeredgecolor='k',\
                markerfacecolor='k',markevery=10000)'''
#line1, = ax.plot([], [], 'o-',color = 'g',lw=4, markersize = 15, markeredgecolor = 'k',markerfacecolor = 'r',markevery=10000)
line, = ax.plot([], [], 'o-',color = 'g',markersize = 3, markerfacecolor = 'k',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Earth
line1, = ax.plot([], [], 'o-',color = 'r',markersize = 8, markerfacecolor = 'b',lw=2, markevery=100000, markeredgecolor = 'k')   # line for Jupiter
line2, = ax.plot([], [], 'o-',color = 'k',markersize = 8, markerfacecolor = 'r',lw=1, markevery=1000000, markeredgecolor = 'k')  


#line4, = ax.plot([], [], color='k', linestyle='-', linewidth=5)
time_template = 't= %.1fs'
time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)


def init():
    line.set_data([],[])
    line1.set_data([],[])
    line2.set_data([], [])
    time_text.set_text('')
    
    return line, time_text, line1,

def animate(i):
    #trail1 =  16
    trail2 = 1100   
    
    line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
    line1.set_data([x1[i],x2[i]],[y1[i],y2[i]])
    line2.set_data(x2[i:max(1,i-trail2):-1], y2[i:max(1,i-trail2):-1])

    
    #line.set_data(x0[i:max(1,i-trail1):-1], y0[i:max(1,i-trail1):-1])   # marker + line of first weight
    #line1.set_data(x1[i:max(1,i-trail2):-1], y1[i:max(1,i-trail2):-1])
   
    time_text.set_text(time_template % t[i])
    
    return line, time_text, line1,

ani_a = animation.FuncAnimation(fig, animate, \
         np.arange(1,len(t)), \
         interval=1,blit=False,init_func=init)

# requires ffmpeg to save mp4 file
#  available from https://ffmpeg.zeranoe.com/builds/
#  add ffmpeg.exe to path such as C:\ffmpeg\bin\ in
#  environmen-t variables
#ani_a.save('Pendulum_Control.mp4',fps=30)

plt.show()

