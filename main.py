import os
import matplotlib.animation as animation
import numpy as np
import matplotlib.pyplot as plt
import xlsxwriter

'''Función de límite de la señal de control'''
def limcontro(u):
    if(u>=0):
        if(u>2.94):
            ur=2.94
        elif(u<=2.94):
            ur=u
        
    else:
        
        if(u>=-2.94):
            ur=u
        else:
            ur=-2.94
    
    return ur

'''Time parameters'''
dt = 0.005  # Tiempo de muestreo (5ms)
ti = 0.0  # Tiempo inicial de la simulación (0s)
tf = 10.0  # Tiempo inicial de la simulación (10s)
n = int((tf - ti) / dt) + 1  # Número de muestras
# Vector con los intsntes de tiempo (en Matlab 0:0.005:10)
t = np.linspace(ti, tf, n)  

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
ise=0#Inicial Jise actual 
ise_next=0#Inicial Jise siguiente
iadu=0#Inicial Jiadu actual
iadu_next=0#Inicial Jiadu siguiente 

'''State equation'''
xdot = [0, 0] #Ecuación de estado

'''Dynamic simulation'''
for i in range(n - 1):
    '''Current states'''
    th = x[i, 0] #Posición angular del péndulo
    th_dot = x[i, 1] #Velocidad angular del péndulo

    '''Controller'''
    e_th =np.pi-th #Error de posicón
    e_th_dot = 0 - th_dot #Error de velocidad

    Kp =6#9.00809903857079 #Ganancia proporcional
    Kd =1#0.74331509706173#Ganancia derivativa 
    Ki =2# Ganancia integral

    u[i] = limcontro(Kp * e_th + Kd * e_th_dot + Ki * ie_th) #Ley de control

    '''System dynamics'''
    xdot[0] = th_dot
    xdot[1] = (u[i] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)

    '''Integrate dynamics'''
    x[i + 1, 0] = x[i, 0] + xdot[0] * dt
    x[i + 1, 1] = x[i, 1] + xdot[1] * dt
    ie_th = ie_th + e_th * dt
    ise=ise_next+(e_th**2)*dt
    iadu=iadu_next+ (abs(u[i]-u[i-1]))*dt
       
    ise_next=ise
    iadu_next=iadu

u[n - 1] = u[n - 2]

# print(x[:, 0]) #Posición del péndulo
# print(ise) #Valor final de ISE
# print(iadu) #Valor final de IADU
#print(t,  x[:, 0])
#datos= [[t],[x[:, 0]]]
archivo=xlsxwriter.Workbook('datos_pendulo7.xlsx')
hoja=archivo.add_worksheet()
for item in range(len(t)):
    hoja.write(item,0,t[item])
    hoja.write(item,1,x[:, 0][item])
archivo.close()



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
plt.plot(t, u, 'r', lw=2)
plt.legend([r'$u$'], loc=1)
plt.ylabel('Control signal')
plt.xlabel('Time')

plt.show()

'''Animation'''

plt.rcParams['animation.html'] = 'html5'

x0=np.zeros(len(t))
y0=np.zeros(len(t))

x1=l*np.sin(x[:,0])
y1=-l*np.cos(x[:,0])

fig = plt.figure(figsize=(8,6.4))
ax = fig.add_subplot(111,autoscale_on=False,\
                     xlim=(-1.8,1.8),ylim=(-1.2,1.2))
ax.set_xlabel('x')
ax.set_ylabel('y')

line, = ax.plot([],[],'o-',color='orange',lw=4,\
                markersize=6,markeredgecolor='k',\
                markerfacecolor='k')
time_template = 't= %.1fs'
time_text = ax.text(0.05,0.9,'',transform=ax.transAxes)


def init():
    line.set_data([],[])
    time_text.set_text('')
    return line, time_text

def animate(i):
   
    line.set_data([x0[i],x1[i]],[y0[i],y1[i]])
    time_text.set_text(time_template % t[i])
    
    return line, time_text,

ani_a = animation.FuncAnimation(fig, animate, \
         np.arange(1,len(t)), \
         interval=40,blit=False,init_func=init)
plt.show()

