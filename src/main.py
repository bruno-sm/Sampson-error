from ransac import *
from sampson import *


def main():
    pts1 = [[1, 2], [3, 4], [5, 6], [8, 10], [9,2], [3,5]]
    pts2 = [[3, 4], [1, 2], [5, 6], [11, 23], [12, 23], [2, 33]]
    h = find_homography_with_ransac(pts1, pts2)
    print([3,5,1])
    print("to")
    # Para multiplicar por la homografía el vector tiene que estar en forma de columna
    r = h * np.matrix([3,5,1]).transpose()
    # Después de transformar el punto con la homografía hay que normalizarlo de nuevo para
    # que la coordenada homogenea sea 1
    r = r * (1/r[2])
    # Devolvemos el punto a un array (forma de fila) 
    r = np.asarray(r.reshape((1,3)))[0]
    print(r)
    print("real")
    print([2,33,1])
    print("Local Sampson error:")
    print(local_sampson_error([3,5], r, h))
    print("Total Sampson error:")
    print(sampson_error(pts1, pts2, h))


if __name__ == "__main__":
    main()
