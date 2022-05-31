import random
import numpy as np
import math
import matplotlib.pyplot as plt

#------------------Parametros de DE--------------#
limit=[(-10,10)]
poblacion = 1000                     # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones = 200                 #  Número de generaciones
D = 1  # Dimensionalidad
M = 2  # Numero de objetivos
DMIN = np.array([-10E0])  # Valor minimo de variables de disenio
DMAX = np.array([10E0])  # Valor maximo de variables de disenio




def dominates(_a, _b):
    for _j in range(M):
        if _b[_j] < _a[_j]:
            return False
    return True



#--- Funciones de optimización --------------------------------------------------+
#funciones mono objetivo 
def func1(x):
    # Esfera
    return sum([x[i]**2 for i in range(len(x))])

def func2(x):
    # Beale's function, límites=[(-4.5, 4.5),(-4.5, 4.5)], f(3,0.5)=0.
    term1 = (1.500 - x[0] + x[0]*x[1])**2
    term2 = (2.250 - x[0] + x[0]*x[1]**2)**2
    term3 = (2.625 - x[0] + x[0]*x[1]**3)**2
    return term1 + term2 + term3

def func3(x):
    #Booth function, límtes -10<=x,y<=10
    term4 = (x[0]+2*x[1]-7)**2
    term5 = (2*x[0]+x[1]-5)**2
    return term4+term5

def func4(x):
    #Matyias function
    return 0.26*(x[0]**2+x[1]**2)-0.48*x[0]*x[1]

def func5(x):
    #Holder table function
    term6= abs(1-((x[0]**2+x[1]**2)**(1/2))/np.pi)
    term7=-abs( np.sin(x[0])*np.cos(x[1])*math.exp(term6))
    return term7

#
def Schaffer(x):
    f1=x[0]**2
    f2=(x[0]-2)**2
    
    return np.array([f1, f2])


#------------------------------------------------------------------#


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


#--- MAIN ---------------------------------------------------------------------+

def main(function, limites, poblacion, f_mut, recombination, generaciones):
    #-----Poblacion--------#
    population =  np.zeros((poblacion, D)) #poblacion actual
    population_next= np.zeros((poblacion, D)) #poblacion siguiente 


    #------------------F(x)--------------------------------------------#

    f_x = np.zeros((poblacion, M))  # Valor de funcion objetivo de poblacion actual
    f_x_next = np.zeros((poblacion, M))  # Valor de funcion objetivo de poblacion siguiente

    #--- Inicializamos la población (Paso #1) ----------------+
    
    for i in range(0,poblacion): # cambiar tam_poblacion
        indv = []
        for j in range(len(limites)):
            indv.append(random.uniform(limites[j][0],limites[j][1]))
            #print(indv[0])
        population[i]=indv[0]
        population_next[i]=indv[0]
    
    #gen_best=np.zeros(generaciones+1)
   
    #--- Evaluación --------------------------------------------+
    #f_x=function(population)
    for i, xi in enumerate(population):  # Evalua objetivos
        #print(f'Individuo {i}: {xi}')
        f_x[i] = function(xi)
    
    print(f_x)
    for i in range(1,generaciones):
        print ('Generación:',i) 
        for j in range(0, poblacion):

    #--- Mutacion (Paso 2) ---------------------------------+
    
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
        
        
            x_1 = population[r1]
            x_2 = population[r2]
            x_3 = population[r3]
            x_t = population[j]
            

            # Restamos x3 de x2, y creamos un nuevo vector (x_diff)
            x_diff =[x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # Multiplicamos x_diff por el factor de mutacion(F) y sumamos x_1
            v_mutante =   [x_1_i + f_mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            
            
            #v_mutante = asegurar_limites(v_mutante, limites)
            #print(f'r1 = {r1}, r2 = {r2}, r3 = {r3}, i = {i}')
            #print('vi = ', v_mutante)

            #--- Recombinación (paso 3) ----------------+

            v_hijo =  np.copy(population[j])
            jrand = random.randint(0, D)
            
            for k in range(len(x_t)):
                crossover = random.uniform(0, 1)
                if crossover <= recombination or  k == jrand:
                    v_hijo[k]=v_mutante[k]

                # else:
                #     v_hijo.append(x_t[k])
                    
            #--- Comparación y selección -------------+
            auw=0
            auq=0
            
            # Evalua descendiente
            
            f_ui = function(v_hijo)
            for n in range(M):
                if (f_ui[n] < f_ui[n]):
                    auw=auw+1
                else:
                    auq=auq+1
                    
            if(auw>auq):
                f_x_next[i] = np.copy(f_ui)
                population_next[i] = np.copy(v_hijo)
               
            elif (auq>auw):
                 f_x_next[j] = np.copy(f_x[j])
                 population_next[j] = np.copy(population[j])
            else:
                if random.uniform(0, 1) < 0.5:
                    f_x_next[j] = np.copy(f_ui)
                    population_next[j] = np.copy(v_hijo)
                else:
                    f_x_next[i] = np.copy(f_x[j])
                    population_next[j] = np.copy(population[j]) 
            
        f_x = np.copy(f_x_next)
        x = np.copy(population_next)
    # Filtrado no dominado
    f_x_fil = np.empty((0, M))  # Conjunto no dominado

    for i1, f_x_1 in enumerate(f_x):
        sol_nd = True
        for i2, f_x_2 in enumerate(f_x):
            if i1 != i2:
                if dominates(f_x_2, f_x_1):
                    sol_nd = False
                    break
        if sol_nd:
            # f_x_fil.append(f_x_1)
            f_x_fil = np.append(f_x_fil, [f_x_1], axis=0)
    
    plt.figure(1)
    plt.title('Aproximacion al frente de Pareto')
    plt.scatter(f_x_fil[:, 0], f_x_fil[:, 1])
    plt.xlabel('f1')
    plt.ylabel('f2')
    plt.show()

    #print(f_x_fil)
    
    return f_x_fil

#--- Parámetros ----------------------------------------------------------------+

function = Schaffer                # Test function
#limites = [(-10, 10),(-10, 10)]    # limites [(x1_min, x1_max), (x2_min, x2_max),...]

#-------------------------------------------------------------------------+

vari=main(function, limit, poblacion, f_mut, recombination, generaciones)







#--- Fin ---------------------------------------------------------------------