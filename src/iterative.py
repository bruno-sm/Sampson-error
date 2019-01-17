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
    Función coste.

X:
    X = f(P), con X el vector medido que aproxima el valor de verdad.

e:
    e_0 = f(P_0) - X

J:
    J es la Jacobiana (df/dP).
    
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
    


*** ***
"""