import numpy as np
from sampson import *
from ransac import find_homography_with_ransac
from derivative import partial_derivative


def find_homography_with_newton(pts1, pts2):
    H = find_homography_with_ransac(pts1, pts2)
    f = lambda P : sampson_error(pts1, pts2, np.reshape(P, (3, 3)))
    P = np.reshape(H, (9, 1))
    ε = np.matrix([f(P)])
#    print("Error: {}".format(ε))

    Δ = np.zeros((1,1)) + np.inf
    while np.mean(Δ) > 0:
        J = np.matrix([partial_derivative(lambda v: f(np.matrix(v)), np.asarray(P.T)[0], i) for i in range(len(P))]) 
        Δ = -np.linalg.pinv(J) * ε
        P = P + Δ
        ε = np.matrix([f(P)]) 
#        print("Error: {}".format(ε))

    return np.reshape(P, (3, 3)), ε[0,0]
