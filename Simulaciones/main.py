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
l = 1.0  # Longitud de la barra del péndulo (m)
lc = 0.3  # Longitud al centro de masa del péndulo (m)
b = 0.05  # Coeficiente de fricción viscosa pendulo
g = 9.81  # Aceleración de la gravedad en la Tierra
I = 0.006  # Tensor de inercia del péndulo

'''State variables'''
x = np.zeros((n, 2))

'''Control vector'''
u = np.zeros((n, 1))

'''Initial conditions'''
x[0, 0] = 0  # Initial pendulum position (rad)
x[0, 1] = 0  # Initial pendulum velocity (rad/s)
ie_th = 0

'''State equation'''
xdot = [0, 0]

'''Dynamic simulation'''
for i in range(n - 1):
    '''Current states'''
    th = x[i, 0]
    th_dot = x[i, 1]

    '''Controller'''
    e_th = np.pi / 2 - th
    e_th_dot = 0 - th_dot

    Kp = 5
    Kd = 1
    Ki = 3

    u[i] = Kp * e_th + Kd * e_th_dot + Ki * ie_th

    '''System dynamics'''
    xdot[0] = th_dot
    xdot[1] = (u[i] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)

    '''Integrate dynamics'''
    x[i + 1, 0] = x[i, 0] + xdot[0] * dt
    x[i + 1, 1] = x[i, 1] + xdot[1] * dt
    ie_th = ie_th + e_th * dt

u[n - 1] = u[n - 2]

print(x[:, 0])

'''Plotting results'''
plt.figure(figsize=(12, 10))
plt.subplot(221)
plt.plot(t, x[:, 0], 'k', lw=1)
plt.legend([r'$\theta$'], loc=1)
plt.ylabel('Pendulum position')
plt.xlabel('Time')

plt.subplot(222)
plt.plot(t, x[:, 1], 'b', lw=1)
plt.legend([r'$\dot{\theta}$'], loc=1)
plt.ylabel('Pendulum speed')
plt.xlabel('Time')

plt.subplot(223)
plt.plot(t, u[:, 0], 'r', lw=2)
plt.legend([r'$u$'], loc=1)
plt.ylabel('Control signal')
plt.xlabel('Time')

plt.show()

'''Animation'''
# Simulation

point = 0
while point < n:
    plt.figure()
    x1 = l * np.sin(x[point, 0])
    y1 = -l * np.cos(x[point, 0])

    plt.plot(x1, y1, 'bo', markersize=20)
    plt.plot([0, x1], [0, y1])
    plt.xlim(-l - 0.5, l + 0.5)
    plt.ylim(-l - 0.5, l + 0.5)
    plt.xlabel('x-direction')
    plt.ylabel('y-direction')
    filenumber = point
    filenumber = format(filenumber, "05")
    filename = "image{}.png".format(filenumber)
    plt.savefig(filename)
    plt.close()

    point = point +20

os.system("ffmpeg -f image2 -r 20 -i image%05d.png -vcodec mpeg4 -y movie.avi")
