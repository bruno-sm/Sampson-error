from sampson import *
from derivative import *
import numpy as np

# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 05:02:50 2019

@author: Usuario
"""

"""
***NEWTON Y GAUSS-NEWTON***

P:
    Vector de parámetros de R^N a R^M.
    
P_0:
    Inicialización.
    
f:
    f(P) = (H x_1, ..., H x_n) = (f_1(x_1,H), ... , f_n(x_n,H))

X:
    X = f(P), con X el vector medido que aproxima el valor de verdad.

e:
    Función coste.
    e_0 = f(P_0) - X

J:
    J es la Jacobiana (df/dP). Matriz nx9.
    
Asumimos que f es aproximada a P_0 por f(P_0 + Delta) = f(P_0) + J Delta.

P_1 = P_0 + Delta
f(P_1) - X = f(P_0) + J Delta - X = e_0 + J Delta
Luego buscamos minimizar ||e_0 + J Delta||.

TEORÍA Delta:
    Se obtiene de resolver: J^T J Delta = - J^T e_0.
    O también de usar la pseudo-inversa Delta = -J^+ e_0.  https://en.wikipedia.org/wiki/Moore%E2%80%93Penrose_inverse#Examples
    - Si (J^T J) es invertible, Delta = - (J^T J)^-1 J^T e_0 = - J^-1 J^T^-1 J^T e_0 = - J^-1 e_0
    - Sino, NO SÉ. ¿Hallar la pseudo-inversa resolviendo ecuaciones? Un horror.

PRÁCTICA Delta:
    Solución al 'linear least-squares problem':  J Delta_i = - e_i.

PRÁCTICA J:
    (df/dP) evaluada en P_i.

PRÁCTICA e_i:
    f(P_i) - X.
    
Reenfoque a mejor: Siendo g(P) una función arbitraria escalar.
Ahora se resume en:
    g_{pp} Delta = - g_p
Donde g_{pp} es la matriz Hessiana de g, y g_p es el gradiente de g.
    g(P) = 1/2 * ||e(P)||^2 = e(P)^T e(P)/2
con e(P) = f(P) - X.
También, se puede computar como

e_p = J.

NEWTON:
    g_{pp} = e_p^T e_p + e_{pp}^T e
    g_p = e_p^T e.

GAUSS-NEWTON:
    g_{pp} = e_p^T e_p + e_{pp}^T e = e_p^T e_p POR SER LINEAL.
    g_p = e_p^T e.
    
    g_{pp} = J^T J
    g_p = J^T e

GRADIENT DESCENT:
    g_{pp} = lambda (múltiplo de la matriz identidad)
    g_p = e_p^T e
    


*** LM ***

TEORÍA Delta:
    Se obtiene de resolver ahora: (J^T J + Lambda I) Delta = - J^T e, para algún valor de Lambda que varía de iteración a iteración.
Normalmente, el valor inicial de Lambda es 10^(-3)*(Media de la diagonal de J^T J).

Si el valor de Delta hallado reduce el error, entonces el incremento es aceptado y lambda = lambda/10.
Si el valor de Delta hallado aumenta el error, entonces lambda = lambda*10 y se repite.

Cuando se acepta un incremento, entonces LM da 1 iteración.

RAZONAMIENTO -> pág 601








-----------------------------

CHECA PABLO:

P:
    Vector de parámetros de R^N a R^M.
    
P_0:
    Inicialización.
    
f:
    f(P) = función de Sampson

X:
    X = valor al que se minimiza. X=0.

e:
    f(P) - X = f(P)

J:
    J es la Jacobiana (df/dP). Matriz 1x9

"""

# LM con f = función de coste.
def LM_fSampson(x1,x2,H,topeIter):
    P = np.asarray(H).reshape(-1)
    f = lambda P : sampson_error(x1,x2,np.reshape(P,(3,3)))
    for iter in range(1,topeIter):
        J = np.matrix([ partial_derivative(f, np.asarray(P)[0], i) for i in range(len(P)) ]) 
        JT = np.transpose(J)
        JTJ = np.dot(JT,J)
        I = np.identity(3)
        e = f(P)
        if(iter == 1):
            Lambda = 10^(-3) * np.average( np.diagonal(JTJ) )
        
        incremento = False
        while(not incremento):
            try:
                Delta = np.linalg.solve(JTJ + Lambda*I, -e*JT)
            except np.linalg.LinAlgError:
                print("Matriz singular.")
                return(-1)
                
            P_nuevo = np.asarray(P) + np.asarray(Delta)
            e_nuevo = f(P_nuevo)
            if(e_nuevo < e):
                incremento = True
                Lambda = Lambda/10
                P = P_nuevo
            else:
                incremento = False
                Lambda = Lambda*10
    
    return np.reshape(P,(3,3))