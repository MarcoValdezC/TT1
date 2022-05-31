# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:15:03 2021

@author: marco
"""

import random
import numpy as np 
import math
import matplotlib.pyplot as plt
from drawnow import *

#---------------------Parametros DE-----------------------------#
limit=[(0,10),(0,10),(0,10)]       #Limites inferior y superior
poblacion = 200                    # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones =10    #  Número de generaciones
D = 3   # Dimensionalidad O número de variables de diseño 
M = 2    # Numero de objetivos
AMAX = 30  # Numero maximo de soluciones en el archivo

#----------------------------------------------------------------


#---------------Función de dominancia------#
def dominates(_a, _b):
    for _j in range(M):               #Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:    
            return False              #Regresa False si a domina b, en este caso seleccionamos b
    return True                       #Regresa Trux si b domina a, en este caso seleccionamos a
#----------------------------------------------------------------------------------------------------
#Funcón de límite del actuador

def limcontro(u):
    if(u>2.94):
        ur=2.94
    elif(u>=-2.94 and u<=2.94):
        ur=u
    else:
        ur=-2.94
    
    return ur
#-----------------------------------------------------------------------------
#----------Problema de optimización---------
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
        
        u[o,0]= limcontro(Kp * e_th + Kd * e_th_dot + Ki * ie_th)
 
        
        
        '''System dynamics'''
        
        xdot[0] = th_dot
        xdot[1] = (u[o] - m * g * lc * np.sin(th) - b * th_dot) / (m * lc ** 2 + I)
        
        '''Integrate dynamics'''
        x[o + 1, 0] = x[o, 0] + xdot[0] * dt
        x[o + 1, 1] = x[o, 1] + xdot[1] * dt
        ie_th = ie_th + e_th * dt
        
        ise=ise_next+(e_th**2)*dt
        iadu=iadu_next+ (abs(u[o]-u[o-1]))*dt
        g=0
        if(ise>=20):
            ie=20
            g+=0
        else:
            ie=ise
            g+=1
        if(iadu>=0.8):
            ia=0.8
            g+=0
        else:
            ia=iadu
            g+=1
   
        ise_next=ie
        iadu_next=ia
        print(ise_next)
        print(iadu_next)
       
    
    return np.array([ise_next, iadu_next])
#----------------------------------------------------------------------------------------------------


#---------------Asegurar limites de caja-------------------------------------------------------------
def asegurar_limites(vec, limit):

    vec_new = []
    # ciclo que recorren todos los individuos
    for i in range(len(vec)):

        # Si el individuo sobrepasa el limite mínimo
        if vec[i] < limit[i][0]:
            vec_new.append(limit[i][0])

        # Si el individuo sobrepasa el limite máximom
        if vec[i] > limit[i][1]:
            vec_new.append(limit[i][1])

        # Si el individuo está dentro de los límites 
        if limit[i][0] <= vec[i] <= limit[i][1]:
            vec_new.append(vec[i])
        
    return vec_new
#---------------------------------------------------------------------------------------------------


#-------------Funcion main, DE----------------------------------------------------------------------

def main(function, limites, poblacion, f_mut, recombination, generaciones):
    
    #-----Poblacion------------------------------------------------------------#
    population =  np.zeros((generaciones,poblacion, D)) #poblacion actual
    population_next= np.zeros((generaciones,poblacion, D)) #poblacion siguiente 
    #---------------------------------------------------------------------------

    #------------------F(x)---------------------------------------------------#
    f_x = np.zeros((generaciones,poblacion, M))  # Valor de funcion objetivo de poblacion actual
    f_x_next = np.zeros((generaciones,poblacion, M))  # Valor de funcion objetivo de poblacion siguiente
    #---------------------------------------------------------------------------
    
    #---------------------------------------------------------------------------
    g_x = np.zeros((generaciones,poblacion))  # Valor de violacion de restricciones de poblacion actual
    g_x_next = np.zeros((generaciones,poblacion))  # Valor de violacion de restricciones de poblacion siguiente
    

    
    
    #--------------------Inicialización de la población-------------------------
    for i in range(0,poblacion): # cambiar tam_poblacion
        indv = []
        for j in range(len(limites)):
            indv.append(random.uniform(limites[j][0],limites[j][1]))
            #print(indv[0])
        population[0][i]=indv[0]
        population_next[0][i]=indv[0]
    
    #print(population[0,:])
    
    #-----------------------------------------------------------------------------------------------------
    
    #-------------Evaluación población 0------------------------------------------------------------------
    for i, xi in enumerate(population[0,:]):  # Evalua objetivos
        f_x[0][i] = function(xi)
    #------------------------------------------------------------------------------------------------------
    
    #---------------------Ciclo evolutivo------------------------------------------------------------------
    for i in range(0,generaciones-1):
        print ('Generación:',i) 
        for j in range(0, poblacion):
            
            #Mutacion 
            # Seleccionamos 4 posiciones de vector aleatorios, range = [0, poblacion)
            candidatos = range(0,poblacion)
            random_index = random.sample(candidatos, 4)
            
            r1 = random_index[0]
            r2 = random_index[1]
            r3 = random_index[2] 
            
            while r1 == j:
                t=random.sample(candidatos, 1)
                r1 = t[0]

            while r2 == r1 or r2 == j:
                t2=random.sample(candidatos,1)
                r2=t2[0]

            while r3 == r2 or r3 == r1 or r3 == j:
                t3 =random.sample(candidatos, 1)
                r3=t3[0]
        
        
            x_1 = population[i][r1]
            x_2 = population[i][r2]
            x_3 = population[i][r3]
            x_t = population[i][j]
            

            # Restamos x3 de x2, y creamos un nuevo vector (x_diff)
            x_diff =[x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # Multiplicamos x_diff por el factor de mutacion(F) y sumamos x_1
            v_mutante =   [x_1_i + f_mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_mutante = asegurar_limites(v_mutante, limites)
            
            #Vector hijo
            v_hijo =  np.copy(population[i][j])
            jrand = random.randint(0, D)
            
            for k in range(len(x_t)):
                crossover = random.uniform(0, 1)
                if crossover <= recombination or  k == jrand:
                    v_hijo[k]=v_mutante[k]
                else:
                    v_hijo[k]=x_t[k]
                    
            
            # Evalua descendiente
            f_ui = function(v_hijo)
            
            # Selecciona el individuo que pasa a la siguiente generacion
            if dominates(f_ui, f_x[i][j]):
                f_x_next[i][j] = np.copy(f_ui)
                population_next[i][j] = np.copy(v_hijo)
            elif dominates(f_x[i][j], f_ui):
                f_x_next[i][j] = np.copy(f_x[i][j])
                population_next[i][j] = np.copy(population[i][j])
            else:
                if random.uniform(0, 1) < 0.5:
                    f_x_next[i][j] = np.copy(f_ui)
                    population_next[i][j] = np.copy(v_hijo)
                else:
                    f_x_next[i][j] = np.copy(f_x[i][j])
                    population_next[i][j] = np.copy(population[i][j])

        # Una vez que termina la generacion actualizo x y f_x
        f_x[i+1] = np.copy(f_x_next[i])
        population[i+1] = np.copy(population_next[i])
        
        
        
        
       
# Filtrado no dominado
    print(i)
    f_x_fil = np.empty((0, M))  # Conjunto no dominado
    population_fil = np.empty((0, D))  # Conjunto no dominado
    population_fil2 = np.empty((0, D))  # Conjunto no dominado
    for i1, f_x_1 in enumerate(f_x[i,:]):
        sol_nd = True
        
        for i2, f_x_2 in enumerate(f_x[i,:]):
            if i1 != i2:
                if dominates(f_x_2, f_x_1):
                    sol_nd = False
                    
                    break
        if sol_nd:
     
            
            f_x_fil = np.append(f_x_fil, [f_x_1], axis=0)
            
            population_fil = np.append(population_fil, [population[i,i1,:]], axis=0)
          
           
    print(f_x_fil)
    print(population_fil)
    print(len(f_x_fil))
    
    #-------Guardar en archivo excel-----------------------------------------
  
    filename="fx1fill11.csv" 
    myFile=open(filename,'w') 
    myFile.write("kp,kd,ki,f1, f2 \n") 
    for l in range(len(f_x_fil)): 
        myFile.write(str(population_fil[l, 0])+","+str(population_fil[l, 1])+","+str(population_fil[l, 2])+","+str(f_x_fil[l, 0])+","+str(f_x_fil[l, 1])+"\n") 
    myFile.close()
    #------------Gráfica del Frente de Pareto-----------------------
    plt.figure(1)
    plt.title('Aproximacion al frente de Pareto')
    plt.scatter(f_x_fil[:, 0], f_x_fil[:, 1])
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
    
    return f_x_fil

#llamado de la función main de DE
var=main(pendulum_s, limit, poblacion, f_mut, recombination, generaciones)



