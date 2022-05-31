# -*- coding: utf-8 -*-
"""
Created on Sat Oct  9 16:08:31 2021

@author: marco
"""

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.visualization.scatter import Scatter
import numpy as np
from pymoo.core.problem import Problem

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
    


'''State equation'''
xdot = [0, 0]

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

class pendulum_s(Problem):

    def __init__(self, **kwargs):
        super().__init__(n_var=3, n_obj=2, n_constr=0, xl=[0, 0,0], xu=[10, 10,10], type_var=np.double, **kwargs)

    def evaluate_single_solution(self, r):
        # _x= p[0]
        # _y = p[1]

        # # Funcion objetivo
        # f1 = 4 * _x ** 2 + 4 * _y ** 2
        # f2 = (_x - 5) ** 2 + (_y - 5) ** 2

        # # Restriccion
        # # g1 = (_x - 5) ** 2 + _y ** 2 - 25
        # # g2 = -(_x - 8) ** 2 - (_y + 3) ** 2 + 7.7
        
        ise=0
        ise_next=0
        iadu=0
        iadu_next=0
    
        '''Initial conditions'''
        x[0, 0] = 0  # Initial pendulum position (rad)
        x[0, 1] = 0  # Initial pendulum velocity (rad/s)
        ie_th = 0

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
        
            u[o,0]= limcontro(Kp * e_th + Kd * e_th_dot + Ki * ie_th)
            '''System dynamics'''
            xdot[0] = th_dot
            xdot[1] = (u[o] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)
        
            '''Integrate dynamics'''
            x[o + 1, 0] = x[o, 0] + xdot[0] * dt
            x[o + 1, 1] = x[o, 1] + xdot[1] * dt
            ie_th = ie_th + e_th * dt
        
            ise=ise_next+(e_th**2)*dt
            iadu=iadu_next+ (abs(u[o,0]-u[o-1,0]))*dt
            if(ise>=2.0):
                ie=2.0
            else:
                ie=ise
            if(iadu>=0.8):
                ia=0.8
            else:
                ia=iadu
   
            ise_next=ie
            iadu_next=ia

        return [ise_next, iadu_next]

    def _evaluate(self, l, out, *args, **kwargs):
        all_F = []
        all_G = []
        for xi in l:
            F = self.evaluate_single_solution(xi)

            all_F.append(F)
            all_G.append(xi)

        out["F"] = np.array(all_F)
        out["G"] = np.array(all_G)
        #print(all_G)


problem = pendulum_s()
algorithm = NSGA2(pop_size=100)

res = minimize(problem,
               algorithm,
               ('n_gen', 100),
               seed=1,
               verbose=False)
X=res.X
plot = Scatter()
plot.add(problem.pareto_front(), plot_type="line", color="black", alpha=0.7)
plot.add(res.F, facecolor="none", edgecolor="red")
plot.show()