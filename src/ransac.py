import numpy as np
import cv2
import math
from matplotlib import pyplot as plt
import functools


# Calcula una homografía entre varias correspondencias (los puntos de pts1 van en correspondencia con los de pts2)
def find_homography_with_ransac(pts1, pts2):
    # Pasamos los puntos a coordenadas homogeneas
    pts1 = [[p[0], p[1], 1.0] for p in pts1]
    pts2 = [[p[0], p[1], 1.0] for p in pts2]

    max_inliers_prop = 0

    # Resuelve la homografía con el mínimo número de puntos requeridos (m=4) escogidos al azar
    m = 4 # Mínimo número de puntos requeridos
    H = dlt(pts1[0:m], pts2[0:m])
    H_aux = H

    # Calculamos N para asegurar con una probabilidad p que al menos uno de los conjuntos de muestras aleatorias no incluya un outlier
    p = 0.99 # Probabilidad que queremos de no incluir un outlier
    v = 0.6 # Probabilidad a priori de que un punto sea un outlier
    N = int(np.log(1-p) / np.log(1 - pow(1 - v, m)))

    print("Iteraciones de RANSAC: {}".format(N))
    i = 0
    while i < N:
        # Resuelve la homografía con el mínimo número de puntos requeridos (m=4) escogidos al azar
        samples_idx = np.random.choice(range(0, len(pts1)), m, False)
        samples1 = [pts1[i] for i in samples_idx]
        samples2 = [pts2[i] for i in samples_idx]
        H_aux = dlt(samples1, samples2)
        # Determina cuantos de los puntos encajan con una tolerancia e
        e = 3
        inliers1 = []
        inliers2 = []
        for j in range(0, len(pts1)):
            # Calculamos la transformación del punto con la homografía obtenida
            trans_pt = H_aux * np.matrix(pts1[j]).transpose()
            trans_pt = np.asarray(trans_pt.transpose())[0]
            # Pasamos el punto transformado a coordenadas homogeneas
            trans_pt = trans_pt * (1/trans_pt[2])
            #print(np.linalg.norm(pts2[j] - trans_pt))
            if np.linalg.norm(pts2[j] - trans_pt) < e:
                inliers1.append(pts1[j])
                inliers2.append(pts2[j])

        inliers_prop = len(inliers1)/len(pts1)
        print("Proporción de inliers iteración {}: {}".format(i, inliers_prop))

        # Reestimamos la proporción de outliers
        v = min(v, 1 - inliers_prop)
        # Calculamos el número de inliers aceptable para la proporción de ouliers estimada
        T = (1 - v) * len(pts1)
        # Si el número de inliers supera la cota T terminamos
        if len(inliers1) > T:
            return dlt(np.array(inliers1), np.array(inliers2))

        # Si la proporción de inliers supera el máximo actual guardamos la homografía
        if inliers_prop > max_inliers_prop:
            H = dlt(np.array(inliers1), np.array(inliers2))
            max_inliers_prop = inliers_prop

        # Recalculamos N para la proporción de outliers estimada
        N = int(np.log(1-p) / np.log(1 - pow(1 - v, m)))
        print("Iteraciones de RANSAC recalculadas: {}".format(N))
        i = i + 1

    return H


# Resuelve la homografía para n correspondencias usando el algoritmo DLT
def dlt(pts1, pts2):
    A = build_complete_dlt_matrix(pts1, pts2)

    np.asarray(A)
    # Descomposición de A en valores singulares para obtener los vectores propios de A^T * A
    u, s, v = np.linalg.svd(A)

    # Nos quedamos con el vector propio con menor valor singular
    h = v[-1::][0]
    # Normalizamos el vector
    h = h * (1/np.linalg.norm(h))
    # Lo ponemos en forma de matriz 3x3
    h = np.reshape(h, (3, 3))
    return h


# Construye la matriz de DLT para un conjunto de correspondencias
# Los puntos tienen dimensión 2, suponiendo que su componente homogenea es 1
def build_complete_dlt_matrix(pts1, pts2):
    A = np.zeros((2*len(pts1), 9), np.float)

    for i, j in enumerate(range(0, 2*len(pts1), 2)):
        A_i = build_dlt_matrix(pts1[i], pts2[i])
        A[j] = A_i[0]
        A[j+1] = A_i[1]

    return A


# Contruye la matriz de DLT para una sola correspondencia
# Los puntos tienen dimensión 2, suponiendo que su componente homogenea es 1
def build_dlt_matrix(pt1, pt2):
    A = np.matrix([
        [0, 0, 0, -pt1[0], -pt1[1], -1, pt2[1]*pt1[0], pt2[1]*pt1[1], pt2[1]],
        [pt1[0], pt1[1], 1, 0, 0, 0, -pt2[0]*pt1[0], -pt2[0]*pt1[1], -pt2[0]]
        ])

    return A
