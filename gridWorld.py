# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 17:18:23 2016

@author: Edwwin
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

maze = np.array([['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', 'G', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', ' ', '*'],
       ['*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*']])


m,n = maze.shape
m -= 2  # ajuste parte superior e inferior
n -= 2  # ajuste laterales
Q = np.zeros((m,n,4))
actions = np.array([[0,1], [1,0], [0,-1], [-1,0]])  # cambios en la posicion x,y del agente

nSteps = 50000
alpha = 0.1
epsilon = 0.1
gamma = 1.00

s_t1 = np.array([1,1])  # posicion de inicio estado siguiente
a_t1 = 1 # indice primera accion a tomar
trials = []
steps = 0

for step in range(nSteps):
    
    here = maze[s_t1[0]+1, s_t1[1]+1]
    if here == 'G':
        # Llego a la celda objetivo G
        Q[s_t1[0],s_t1[1],a_t1] = 0
        if steps > 0:
            Q[s_t[0],s_t[1],a_t] += alpha * (0 + gamma * Q[s_t1[0],s_t1[1],a_t1] - Q[s_t[0],s_t[1],a_t])
        s_t1 = np.array([np.random.randint(0,m),np.random.randint(0,n)])
        trials.append(steps)
        steps = 0

    else:
        # Celda vacia
        steps += 1
    
        # Seleccion de accion siguiente
        if np.random.uniform() < epsilon:
            a_t1 = np.random.randint(0,len(actions))
        else:
            a_t1 = np.argmax(Q[s_t1[0],s_t1[1],:])

        if steps > 1:
            Q[s_t[0],s_t[1],a_t] += alpha * (-1 + gamma * Q[s_t1[0],s_t1[1],a_t1] - Q[s_t[0],s_t[1],a_t])

        s_t = s_t1
        a_t = a_t1
        s_t1 = s_t1 + actions[a_t1,:]
        if s_t1[0] > 9: s_t1[0] -= 1
        if s_t1[1] > 9: s_t1[1] -= 1
        if s_t1[0] < 0: s_t1[0] += 1
        if s_t1[1] < 0: s_t1[1] += 1


fig = plt.figure(1,figsize=(8,8))
ax = fig.add_subplot(3,1,1, projection='3d')

## Plot de Qmax para cada estado
(m,n,_) = Q.shape
gridsize = max(m,n)
xs = np.floor(np.linspace(0,m-0.5,gridsize))
ys = np.floor(np.linspace(0,n-0.5,gridsize))
xgrid,ygrid = np.meshgrid(xs,ys)
points = np.vstack((xgrid.flat,ygrid.flat))

# Politica derivada de Q
Qmaxs = [np.max( Q[s1,s2,:]) for (s1,s2) in zip(points[0,:],points[1,:])]
Qmaxs = np.asarray(Qmaxs).reshape(xgrid.shape)
surf = ax.plot_surface(xgrid,ygrid,Qmaxs,rstride=1,cstride=1,color='yellow')
ax.set_zlabel("Qmax")
ax.set_title("Min %d Max %d" % tuple(np.round((np.min(Qmaxs),np.max(Qmaxs)))))

# Numero de pasos por episodio
plt.subplot(3,1,2)
bestactions = np.argmax(Q,axis=2)
px,py = np.meshgrid(np.arange(m)-0.5, np.arange(n)-0.5)
pts = np.vstack((px.flat,py.flat)).T
arrowx = actions[:,0][bestactions]
arrowy = actions[:,1][bestactions]
plt.quiver(px,py,arrowx,arrowy)

plt.subplot(3,1,3)
plt.plot(trials)
plt.ylabel("steps")
plt.xlabel("episodes")

plt.show()
