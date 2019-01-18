from sampson import *
from derivative import *
from ransac import *
import numpy as np

# -*- coding: utf-8 -*-

"""    

*** LM ***

TEORÍA Delta:
    Se obtiene de resolver ahora: (J^T J + Lambda I) Delta = - J^T e, para algún valor de Lambda que varía de iteración a iteración.
Normalmente, el valor inicial de Lambda es 10^(-3)*(Media de la diagonal de J^T J).

Si el valor de Delta hallado reduce el error, entonces el incremento es aceptado y lambda = lambda/10.
Si el valor de Delta hallado aumenta el error, entonces lambda = lambda*10 y se repite.

Cuando se acepta un incremento, entonces LM da 1 iteración.

-----------------------------

LM con f como función de Sampson:

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
def LM_fSampson(x1,x2,topeIter,topePasosSinIncremento):
    H = find_homography_with_ransac(x1, x2)
    P = np.asarray(H).reshape(-1)
    f = lambda P : sampson_error(x1,x2,np.reshape(P,(3,3)))
    iter = 1
    pasos_sin_incremento = 0
    while((iter < topeIter) and (pasos_sin_incremento < topePasosSinIncremento)):
        J = np.matrix([ partial_derivative(f, np.asarray(P), i) for i in range(len(P)) ]) 
        JT = np.transpose(J)
        JTJ = np.dot(JT,J)
        I = np.identity(9)
        e = f(P)
        if(iter == 1):
            Lambda = 10**(-3) * np.average( np.diagonal(JTJ) )
        
        incremento = False
        while(not incremento and (pasos_sin_incremento < topePasosSinIncremento)):
            try:
                Delta = np.linalg.solve(JTJ + Lambda*I, -e*JT)
                Delta = np.asarray(Delta).reshape(-1)
            except np.linalg.LinAlgError:
                print("Matriz singular.")
                return(-1)
            
            P_nuevo = np.asarray(P) + np.asarray(Delta)
            e_nuevo = f(P_nuevo)
            if(e_nuevo < e):
                incremento = True
                pasos_sin_incremento = 0
                Lambda = Lambda/10
                P = P_nuevo
            else:
                incremento = False
                pasos_sin_incremento = pasos_sin_incremento +1
                Lambda = Lambda*10
        iter = iter +1
    
    return (np.reshape(P,(3,3)),e)