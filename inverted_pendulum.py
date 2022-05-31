import os

import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt

'''Time parameters'''
dt = 0.005  # Tiempo de muestreo (5ms)
ti = 0.0  # Tiempo inicial de la simulación (0s)
tf = 10.0  # Tiempo inicial de la simulación (10s)
n = int((tf - ti) / dt) + 1  # Número de muestras
t = np.linspace(ti, tf, n)  # Vector con los intsntes de tiempo (en Matlab 0:0.005:10)

'''Dynamic parameters'''
m = 0.5  # Masa del pendulo (kg)
M = 0.7  # Masa del carro (kg) 
l = 1.0  # Longitud de la barra del péndulo (m)
lc = 0.3  # Longitud al centro de masa del péndulo (m)
b1 = 0.05  # Coeficiente de fricción viscosa pendulo
b2 = 0.06  #Coeficiente de friccion del carro
g = 9.81  # Aceleración de la gravedad en la Tierra
I = 0.006  # Tensor de inercia del péndulo

'''State variables'''
z = np.zeros((n, 4))

'''Control vector'''
u = np.zeros((n, 2))

'''Initial conditions'''
z[0, 0] =0 # Initial car position (rad)
z[0, 1] = 0 # Initial speed velocity (rad/s)
z[0, 2] = -np.pi/2 # Initial pendulum position 
z[0, 3] = 0  # Initial pendulum speed (rad)
ie_th = 0
ie_x=0

'''State equation'''
zdot = [0,0,0,0]

'''Dynamic simulation'''
for i in range(n - 1):
    
    '''Current states'''
    x     = z[i, 0] #Posicion del carro
    x_dot = z[i, 1] #Velocidad del carro
    th    = z[i, 2] #Posición del péndulo
    th_dot= z[i, 3] #Velocidad del péndulo

    '''Controller'''
    e_x      = 2-x
    e_x_dot  = 0.1-x_dot
    e_th     = np.pi/2- th
    e_th_dot = 0 - th_dot

    Kp = 5
    Kd = 3
    Ki = 1
    
    Kp1 = 5
    Kd1 = 3
    Ki1 = 1

    u[i,1] = Kp * e_th + Kd * e_th_dot + Ki * ie_th
    u[i,0] = Kp1 * e_x + Kd1 * e_x_dot + Ki1 * ie_x

    '''System dynamics'''
    zdot[0] = x_dot
    zdot[1] = (u[i,0]+m*lc*th_dot**2*(np.sin(th)) -m*lc*th_dot*np.cos(th)-b1*x_dot) / M+m
    zdot[2] = th_dot
    zdot[3] = (u[i,1]-m*lc*x_dot*np.cos(th)-m*g*lc*np.sin(th)-b2*th_dot)/(I+m*lc**2)
    
    '''Integrate dynamics'''
    z[i + 1, 0] = z[i, 0] + zdot[0] * dt
    z[i + 1, 1] = z[i, 1] + zdot[1] * dt
    z[i + 1, 2] = z[i, 2] + zdot[2] * dt
    z[i + 1, 3] = z[i, 3] + zdot[3] * dt
    ie_th = ie_th + e_th * dt

u[n - 1] = u[n - 2]

print(z[:, 0])

'''Plotting results'''
plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.plot(t, z[:, 0], 'k', lw=1)
plt.legend([r'$\theta$'], loc=1)
plt.ylabel('Car position')
plt.xlabel('Time')

plt.subplot(222)
plt.plot(t, z[:, 1], 'b', lw=1)
plt.legend([r'$\dot{\theta}$'], loc=1)
plt.ylabel('Car speed')
plt.xlabel('Time')

plt.subplot(223)
plt.plot(t, z[:, 2], 'r', lw=2)
plt.legend([r'$u$'], loc=1)
plt.ylabel('Pendulum position')
plt.xlabel('Time')

plt.subplot(224)
plt.plot(t, z[:, 3], 'r', lw=2)
plt.legend([r'$u$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.show()

'''Animation'''
x1 = z[:,0]
y1 = np.zeros(len(t))

#suppose that l = 1
x2 = l*np.cos(z[:,2])+x1
y2 = l*np.sin(z[:,2])

fig = plt.figure(figsize=(8,6.4))
ax = fig.add_subplot(111,autoscale_on=False,\
                     xlim=(-2.5,5),ylim=(-2.2,2.2))
ax.set_xlabel('position')
ax.get_yaxis().set_visible(True)

mass1, = ax.plot([],[],linestyle='None',marker='s',\
                 markersize=10,markeredgecolor='k',\
                 color='green',markeredgewidth=2)

line, = ax.plot([],[],'o-',color='green',lw=4,\
                markersize=6,markeredgecolor='k',\
                markerfacecolor='k')
time_template = 't= %.1fs'
time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)


def init():
    mass1.set_data([],[])
    line.set_data([],[])
    time_text.set_text('')
    
    return line, mass1, time_text

def animate(i):
    mass1.set_data([x1[i]],[y1[i]])
    line.set_data([x1[i],x2[i]],[y1[i],y2[i]])
    time_text.set_text(time_template % t[i])
    return mass1,line, time_text

ani_a = animation.FuncAnimation(fig, animate, \
         np.arange(1,len(t)), \
         interval=100,blit=False,init_func=init)

# requires ffmpeg to save mp4 file
#  available from https://ffmpeg.zeranoe.com/builds/
#  add ffmpeg.exe to path such as C:\ffmpeg\bin\ in
#  environmen-t variables
#ani_a.save('Pendulum_Control.mp4',fps=30)

plt.show()
