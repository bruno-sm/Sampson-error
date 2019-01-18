import cv2
import numpy as np
from gauss_newton import *
from iterative import *
import time


def main():
    imgs = [cv2.imread("imagenes/" + name, 1) for name in ["mosaico002.jpg", "mosaico003.jpg", "mosaico004.jpg"]]

    # Crea una imagen en negro de tamaño suficiente
    h = int(imgs[0].shape[0] * 3) 
    w = int(imgs[0].shape[1] * 5)
    canvas = np.zeros((w, h, 3), dtype=np.uint8)
    show_img((stitch_images(imgs, canvas, find_homography_with_gauss_newton), "Mosaico"))


# Une la imágenes en un mosaico y las introduce en el canvas
def stitch_images(imgs, canvas, homography_estimator):
    if len(imgs) == 1:
        print(imgs[0])
        return imgs[0]

    central_idx = int((len(imgs) - 1)/2)
    canvas_size = canvas.shape[:2]
    # Homografía que centra la imagen central
    traslation_to_center = np.matrix([
        [1.0, 0.0, (canvas_size[0]/2 - imgs[central_idx].shape[1]/2)],
        [0.0, 1.0, (canvas_size[1]/2 - imgs[central_idx].shape[0]/2)],
        [0.0, 0.0, 1.0]
        ])
    # Dibujamos la imagen central en el canvas
    canvas = cv2.warpPerspective(imgs[1], traslation_to_center, canvas_size, canvas, borderMode=cv2.BORDER_TRANSPARENT)

    H = traslation_to_center
    # Calculamos el mosaico desde la imagen central hacia la izquierda
    for i in reversed(range(central_idx)):
        H = H * homography_between_images(imgs[i], imgs[i+1], homography_estimator)
        canvas = cv2.warpPerspective(imgs[i], H, canvas_size, canvas, borderMode=cv2.BORDER_TRANSPARENT)

    H = traslation_to_center
    # Calculamos el mosaico desde la imagen central hacia la derecha
    for i in range(central_idx+1, len(imgs)):
        H = H * homography_between_images(imgs[i], imgs[i-1], homography_estimator)
        canvas = cv2.warpPerspective(imgs[i], H, canvas_size, canvas, borderMode=cv2.BORDER_TRANSPARENT)

    #Eliminamos los bordes sobrantes 
    gray = cv2.cvtColor(canvas,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,1,255,cv2.THRESH_BINARY)
    _,contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnt = contours[0]
    x,y,w,h = cv2.boundingRect(cnt)
    return canvas[y:y+h,x:x+w]


def homography_between_images(img1, img2, homography_estimator):
    # Obtenemos los key points y sus descriptores
    kp1, des1 = sift_kps_and_descriptors(img1)
    kp2, des2 = sift_kps_and_descriptors(img2)

    # Emparejamos por Lowe
    matches = lowe_matches(kp1, kp2, des1, des2)
    if len(matches) >= 4:
        match_idx = [(m[0].trainIdx, m[0].queryIdx) for m in matches]
        pts1 = np.float32([kp1[i].pt for (_, i) in match_idx])
        pts2 = np.float32([kp2[i].pt for (i, _) in match_idx])
        
        t1 = time.time()
        H, error = homography_estimator(pts1, pts2)
        t2 = time.time()
        print("Error: {}".format(error))
        print("Tiempo: " + str(t2-t1))
        return H

    raise "Error: No hay suficientes matches para calcular la homografía"


def sift_kps_and_descriptors(img):
    # Obtenemos los key points
    sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.06, edgeThreshold=10)
    return sift.detectAndCompute(img, None)


def lowe_matches(kp1, kp2, des1, des2):
    #FLANN_INDEX_KDTREE = 1
    #index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    #search_params = dict(checks=50)
    #flann = cv2.FlannBasedMatcher(index_params, search_params)
    bf = cv2.BFMatcher()
    raw_matches = bf.knnMatch(des1, des2, k=2)
    matches = [(m, n) for (m, n) in raw_matches if m.distance < 0.7*n.distance]
    return matches


def show_img(im):
    img, name = im
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows


if __name__ == "__main__":
    main()
