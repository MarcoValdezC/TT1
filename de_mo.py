import numpy as np
import matplotlib.pyplot as plt
from numpy import random

FR = 0.5  # Factor de escalamiento
CR = 0.5  # Factor de cruza
NP = 200  # Tamanio de poblacion
GMAX = 100  # Generacion maxima
D = 1  # Dimensionalidad
M = 2  # Numero de objetivos
DMIN = np.array([-10E0])  # Valor minimo de variables de disenio
DMAX = np.array([10E0])  # Valor maximo de variables de disenio

x = np.zeros((NP, D))  # Poblacion actual
x_next =  np.zeros((NP, D))  # Poblacion siguiente

f_x = np.zeros((NP, M))  # Valor de funcion objetivo de poblacion actual
f_x_next = np.zeros((NP, M))  # Valor de funcion objetivo de poblacion siguiente


def dominates(_a, _b):
    for _j in range(M):
        if _b[_j] < _a[_j]:
            return False
    return True


def F(_p):
    _x = _p[0]

    f1 = _x ** 2
    f2 = (_x - 2) ** 2

    return np.array([f1, f2])


'''
def F2(_p):
    _x = _p[0]
    _y = _p[1]

    f1 = 4 * _x ** 2 + 4 * _y ** 2
    f2 = (_x - 5) ** 2 + (_y - 5) ** 2

    g1 = (_x - 5) ** 2 + _y ** 2 <= 25
    g2 = (_x - 8) ** 2 + (_y + 3) ** 2 >= 7.7
    g = int(g1) + int(g2)

    return np.array([f1, f2]), g

# x = np.array([-1,0])
x = np.array([1, 1])
f_x, g_x = F2(np.array(x))
print(f'f_x = {f_x}, g_x = {g_x}')
exit(0)
'''

# Paso 1. Inicializacion
x = DMIN + np.random.rand(NP, D) * (DMAX - DMIN)  # Inicializa poblacion



for i, xi in enumerate(x):  # Evalua objetivos
    # print(f'Individuo {i}: {xi}')
    f_x[i] = F(xi)

#print(f_x)

# Paso 2. Ciclo evolutivo
for gen in range(GMAX):  # Para cada generacion
    for i in range(NP):  # Para cada individuo
        # Selecciona r1 != r2 != r3 != i
        r1 = i
        r2 = i
        r3 = i

        while r1 == i:
            r1 = random.randint(0, NP)

        while r2 == r1 or r2 == i:
            r2 = random.randint(0, NP)

        while r3 == r2 or r3 == r1 or r3 == i:
            r3 = random.randint(0, NP)

        # print(f'r1 = {r1}, r2 = {r2}, r3 = {r3}, i = {i}')

        # Genera individuo mutante
        vi = x[r1] + FR * (x[r2] - x[r3])
        
        #print('vi = ', vi)

        # Genera individuo descendiente
        ui = np.copy(x[i])

        jrand = random.randint(0, D)
        # print(jrand)

        for j in range(D):
            if random.uniform(0, 1) < CR or j == jrand:
                ui[j] = vi[j]
        # print('ui = ', ui)

        # Evalua descendiente
        f_ui = F(ui)
        #print(f_ui)

        # Selecciona el individuo que pasa a la siguiente generacion
        if dominates(f_ui, f_x[i]):
            f_x_next[i] = np.copy(f_ui)
            x_next[i] = np.copy(ui)
        elif dominates(f_x[i], f_ui):
            f_x_next[i] = np.copy(f_x[i])
            x_next[i] = np.copy(x[i])
        else:
            if random.uniform(0, 1) < 0.5:
                f_x_next[i] = np.copy(f_ui)
                x_next[i] = np.copy(ui)
            else:
                f_x_next[i] = np.copy(f_x[i])
                x_next[i] = np.copy(x[i])

    # Una vez que termina la generacion actualizo x y f_x
    f_x = np.copy(f_x_next)
    x = np.copy(x_next)

# print(f_x)

# Filtrado no dominado
f_x_fil = np.empty((0, M))  # Conjunto no dominado

for i1, f_x_1 in enumerate(f_x):
    sol_nd = True
    print(f_x_1)
    for i2, f_x_2 in enumerate(f_x):
        print(f_x_2)
        if i1 != i2:
            if dominates(f_x_2, f_x_1):
                sol_nd = False
                break
    if sol_nd:
        # f_x_fil.append(f_x_1)
        f_x_fil = np.append(f_x_fil, [f_x_1], axis=0)

print(len(f_x_fil))

plt.figure(1)
plt.title('Aproximacion al frente de Pareto')
plt.scatter(f_x_fil[:, 0], f_x_fil[:, 1])
plt.xlabel('f1')
plt.ylabel('f2')
plt.show()
