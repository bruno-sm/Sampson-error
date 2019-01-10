import numpy as np
from ransac import build_dlt_matrix


# Calcula el error de sampson de la homografía H con las correspondencias dadas por pts1 y pts2 
def sampson_error(pts1, pts2, H):
    return np.sum([local_sampson_error(pt1, pt2, H) for pt1, pt2 in zip(pts1, pts2)])


# Calcula el error de sampson de la homografía H con la correspondencia pt1 -> pt2 
def local_sampson_error(pt1, pt2, H):
    # Calculamos los componentes de la fórmula del error
    A = build_dlt_matrix(pt1, pt2)
    h = H.reshape((9, 1))
    ε = A*h

    x = pt1[0]
    y = pt1[1]
    u = pt2[0]
    v = pt2[1]

    # Jacobiano de Ah
    J = np.matrix([
        [h[6]*v - h[3], h[7]*v - h[4], 0, h[6]*x + h[7]*y + h[8]],
        [h[0] - h[6]*u, h[1] - h[7]*u, -h[6]*x - h[7]*y - h[8], 0]
        ], dtype='float')

    # Fórmula del error de Sampson
    return (ε.T * np.linalg.inv(J * J.T) * ε)[0, 0]
