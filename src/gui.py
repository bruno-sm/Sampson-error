import sys
import os
import numpy as np
import qimage2ndarray as i2a
from main import stitch_images 
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.QtCore import Qt, QSize, pyqtSignal, pyqtSlot, QRectF
from gauss_newton import find_homography_with_gauss_newton
from iterative import *


class DragDropListWidget(QListWidget):
    dropped = pyqtSignal(list)
    urlList = []


    def __init__(self, parent):
        super(DragDropListWidget, self).__init__(parent)
        self.setAcceptDrops(True)
        self.setIconSize(QSize(72, 72))


    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
        else:
            event.ignore()


    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
        else:
            event.ignore()

    def dropEvent(self, event):
        if event.mimeData().hasUrls():
            event.setDropAction(Qt.CopyAction)
            event.accept()
            l = []
            for url in event.mimeData().urls():
                l.append(str(url.toLocalFile()))
                if os.path.exists(str(url.toLocalFile())):
                    self.urlList.append(str(url.toLocalFile()))
            self.dropped.emit(l)
        else:
            event.ignore()



class MosaicView(QLabel):
    def __init__(self):
        QLabel.__init__(self)


    def setPixmap(self, pixmap):
        self.setPixmap(pixmap)



class Window(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        
        self.showMaximized()    
        self.setWindowTitle("Compositor de mosaicos") 

        centralWidget = QWidget(self)          
        self.setCentralWidget(centralWidget)   

        self.algorithm = find_homography_with_gauss_newton 
        self.algorithmCombo = QComboBox(self)
        self.algorithmCombo.setObjectName(("algorithmComboBox"))
        self.algorithmCombo.addItem("Gauss-Newton")
        self.algorithmCombo.addItem("LM")
        self.algorithmCombo.activated[str].connect(self.changedAlgorithm)

        self.pictureList = DragDropListWidget(self)
        self.pictureList.dropped.connect(self.pictureDropped)

        self.pictureViewer = QLabel(self) 

        hbox = QHBoxLayout(self)
        hbox.addWidget(self.pictureList)
        hbox.addWidget(self.pictureViewer)

        vbox = QVBoxLayout(self)
        centralWidget.setLayout(vbox)
        vbox.addWidget(self.algorithmCombo)
        vbox.addLayout(hbox)


    @pyqtSlot(list)
    def pictureDropped(self, l):
        for url in l:
            if os.path.exists(url):
                icon = QIcon(url)
                item = QListWidgetItem(os.path.basename(url)[:20] + "...")
                item.setIcon(icon)
                item.setStatusTip(url)
                self.pictureList.addItem(item)

        self.show_mosaic()        


    @pyqtSlot(str)
    def changedAlgorithm(self, name):
        if name == "Gauss-Newton":
            self.algorithm = find_homography_with_gauss_newton 
        elif name == "LM":
            self.algorithm = lambda pts1, pts2: LM_fSampson(pts1, pts2, 10, 50)

        self.show_mosaic()


    def show_mosaic(self):
        if self.pictureList.urlList == []:
            return

        imgs = [i2a.rgb_view(QImage(url)) for url in self.pictureList.urlList]

        h = int(imgs[0].shape[0] * 3) 
        w = int(imgs[0].shape[1] * 5) 
        canvas = np.zeros((w, h, 3), dtype=np.uint8)
        mosaic = i2a.array2qimage(stitch_images(imgs, canvas, self.algorithm))
        self.pictureViewer.setPixmap(QPixmap.fromImage(mosaic))


if __name__ == '__main__':

    app = QApplication(sys.argv)

    w = Window()
    w.show()

    sys.exit(app.exec_())
