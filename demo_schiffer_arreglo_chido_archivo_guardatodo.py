# -*- coding: utf-8 -*-
"""
Created on Sun Jun 27 21:15:03 2021

@author: marco
"""

import random
import numpy as np
import math
import matplotlib.pyplot as plt
import serial  
from drawnow import *

#---------------------Parametros DE-----------------------------#
limit=[(-10,10)]                   #Limites inferior y superior
poblacion = 200                    # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones = 200                 #  Número de generaciones
D = 1    # Dimensionalidad O número de variables de diseño 
M = 2    # Numero de objetivos

#---------------------------------------------------------------------------------------------------

#---------------Función de dominancia------#
def dominates(_a, _b):
    for _j in range(M):               #Recorre el vector J de funciones objetivo
        if _b[_j] < _a[_j]:    
            return False              #Regresa False si a domina b, en este caso seleccionamos b
    return True                       #Regresa Trux si b domina a, en este caso seleccionamos a
#----------------------------------------------------------------------------------------------------

#-----------------------------Funciones objetivo-----------------------------------------------------
#Función de Schaffer 
def Schaffer(x):
    f1=x[0]**2
    f2=(x[0]-2)**2
    return np.array([f1, f2])
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
            #print(v_mutante)
            #Vector hijo
            v_hijo =  np.copy(population[i][j])
            jrand = random.randint(0, D)
            
            for k in range(len(x_t)):
                crossover = random.uniform(0, 1)
                if crossover <= recombination or  k == jrand:
                    v_hijo[k]=v_mutante[k]
                else:
                    v_hijo[k]=x_t[k]
                    
            #
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
        print(f_x[i,:])
        filename="datos"+str(i)+".csv" 
        myFile=open(filename,'w') 
        myFile.write("x, f1,f2 \n") 
        for w in range(len(population[i+1])): 
            myFile.write(str(population[i+1,w,0])+","+str(f_x[i+1,w, 0])+","+str(f_x[i+1,w, 1])+"\n") 
        myFile.close() 
       
# Filtrado no dominado
    f_x_fil = np.empty((0, M))  # Conjunto no dominado

    for i1, f_x_1 in enumerate(f_x[i+1,:]):
        sol_nd = True
        for i2, f_x_2 in enumerate(f_x[i+1,:]):
            if i1 != i2:
                if dominates(f_x_2, f_x_1):
                    sol_nd = False
                    break
        if sol_nd:
            # f_x_fil.append(f_x_1)
            f_x_fil = np.append(f_x_fil, [f_x_1], axis=0)
            #[i]=np.append(f_x_fil)
    print(len(f_x_fil))
    
    filename="fxfill.csv" 
    myFile=open(filename,'w') 
    myFile.write("f1, f2 \n") 
    for l in range(len(f_x_fil)): 
        myFile.write(str(f_x_fil[l, 0])+","+str(f_x_fil[l, 1])+"\n") 
    myFile.close() 
    
    plt.figure(1)
    plt.title('Aproximacion al frente de Pareto')
    plt.scatter(f_x_fil[:, 0], f_x_fil[:, 1])
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()
    
    return f_x_fil


var=main(Schaffer, limit, poblacion, f_mut, recombination, generaciones)



