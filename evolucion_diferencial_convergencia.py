import random
import numpy as np
import math
import matplotlib.pyplot as plt

#--- Funciones de optimización ---------------------------------------------------+

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
def Schaffer(x):
    f1=x[0]**2
    f2=(x[0]-2)**2
    J=np.array([[f1],[f2]])
    return J


#------------------------------------------------------------------+


def asegurar_limites(vec, limit):

    vec_new = []
    # ciclo que recorren todos los individuos
    for i in range(len(vec)):

        # Si el individuo sobrepasa el limite mínimo
        if vec[i] < limites[i][0]:
            vec_new.append(limites[i][0])

        # Si el individuo sobrepasa el limite máximom
        if vec[i] > limites[i][1]:
            vec_new.append(limites[i][1])

        # Si el individuo está dentro de los límites 
        if limites[i][0] <= vec[i] <= limites[i][1]:
            vec_new.append(vec[i])
        
    return vec_new


#--- MAIN ---------------------------------------------------------------------+

def main(function, limites, poblacion, f_mut, recombination, generaciones):

    #--- Inicializamos la población (Paso #1) ----------------+
    
    population = []
    population_next=[]
    for i in range(0,poblacion): # cambiar tam_poblacion
        indv = []
        for j in range(len(limites)):
            indv.append(random.uniform(limites[j][0],limites[j][1]))
            print(indv[0])
            
        population.append(indv)
        population_next.append(indv)
    print (population)
    gen_best=np.zeros(generaciones+1)

    #--- Evaluación --------------------------------------------+

    #print(population)
    for i in range(1,generaciones+1):
        #print ('Generación:',i)

        gen_scores = [] 

        
        for j in range(0, poblacion):

    #--- Mutacion (Paso 2) ---------------------------------+
    
    # Seleccionamos 4 posiciones de vector aleatorios, range = [0, poblacion)
            candidatos = range(0,poblacion)
            random_index = random.sample(candidatos, 4)
        
            x_1 = population[random_index[0]]
            x_2 = population[random_index[1]]
            x_3 = population[random_index[2]]
            x_t = population[j]
            

            # Restamos x3 de x2, y creamos un nuevo vector (x_diff)
            x_diff = [x_2_i - x_3_i for x_2_i, x_3_i in zip(x_2, x_3)]

            # Multiplicamos x_diff por el factor de mutacion(F) y sumamos x_1
            v_mutante = [x_1_i + f_mut * x_diff_i for x_1_i, x_diff_i in zip(x_1, x_diff)]
            v_mutante = asegurar_limites(v_mutante, limites)

            #--- Recombinación (paso 3) ----------------+

            v_hijo = []
            for k in range(len(x_t)):
                crossover = random.random()
                if crossover <= recombination:
                    v_hijo.append(v_mutante[k])

                else:
                    v_hijo.append(x_t[k])
                    
            #--- Comparación y selección -------------+
            
            
            score_trial  = function(v_hijo)
            score_target = function(x_t)  #Arreglo de la poblacion original, score de la pob orig y sig.
            #print(score_trial[1])
            #print(len(score_trial))
            auw=0
            auq=0
            for n in range(len(score_trial)):
                if (score_trial[n]<=score_target[n]):
                    auw=auw+1
                else:
                    auq=auq+1
                    
            if(auw>auq):
                population_next[j] = v_hijo
                gen_scores.append(score_trial)
                #print ('   >',score_trial, v_hijo)
            elif (auq>auw):
                population_next[j]=population[j]
                #print ('   >',score_target, x_t)
                gen_scores.append(score_target)
            


            # if score_trial < score_target:
            #     population_next[j] = v_hijo
            #     gen_scores.append(score_trial)
            #     print ('   >',score_trial, v_hijo)

            # else:
            #     population_next[j]=population[j]
            #     #print ('   >',score_target, x_t)
            #     gen_scores.append(score_target)

        #--- --------------------------------+
        #print(gen_scores)
        # population=population_next.copy()
        # gen_avg = sum(gen_scores) / poblacion                         # current generation avg. fitness
        # gen_best[i] = min(gen_scores)                                  # Desempeño del mejor individio
        # gen_sol = population[gen_scores.index(min(gen_scores))]     # Solucion

        # print ('      > Promedio de generacion:',gen_avg)
        # print ('      > Mejor individuo:',gen_best[i])
        # print ('         > Mejor solucion:',gen_sol,'\n')

    return gen_sol, gen_best

#--- Parámetros ----------------------------------------------------------------+

function = Schaffer                # Test function
limites = [(-10, 10),(-10, 10)]    # limites [(x1_min, x1_max), (x2_min, x2_max),...]
limit=[(-10,10)]
poblacion = 50                     # Tamaño de la población, mayor >= 4
f_mut = 0.5                        # Factor de mutacion [0,2]
recombination = 0.7                # Tasa de  recombinacion [0,1]
generaciones = 200                 #  Número de generaciones

#-------------------------------------------------------------------------+

vari=main(function, limit, poblacion, f_mut, recombination, generaciones)
yy=vari[1]

x = np.linspace(0, generaciones, generaciones+1)

plt.figure(figsize=(20, 20))
plt.plot(x,yy, 'r', lw=2)
plt.legend([r'f(x)'], loc=1)
plt.ylabel('f(x) ')
plt.xlabel('Generaciones')


#--- Fin ---------------------------------------------------------------------