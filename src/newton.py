import numpy as np
from sampson import *
from ransac import find_homography_with_ransac
from derivative import partial_derivative


def find_homography_with_newton(pts1, pts2):
    H = find_homography_with_ransac(pts1, pts2)
    f = lambda P : sampson_error(pts1, pts2, np.reshape(P, (3, 3)))
    vectorized_f = np.vectorize(
        lambda h1, h2, h3, h4, h5, h6, h7, h8, h9 : f(np.matrix([h1, h2, h3, h4, h5, h6, h7, h8, h9]).T)
    )
    P = np.reshape(H, (1, 9))
    ε = np.matrix([f(P)])

    Δ = np.zeros((1,1)) + np.inf
    while np.mean(Δ) > 0:
        J = np.matrix([partial_derivative(f, np.asarray(P)[0], i) for i in range(len(P))]) 
        Δ = -np.linalg.pinv(J) * ε
        print("Δ: {}".format(Δ))
        P = P + Δ
        ε = np.matrix([f(P)]) 
        print("Error: {}".format(np.sum(ε)))

    return np.reshape(P, (3, 3))
