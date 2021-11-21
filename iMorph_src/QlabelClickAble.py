from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtWidgets import QApplication, QDialog, QLabel, QMessageBox
from PyQt5 import QtGui
from numpy import arange
from PyQt5 import QtCore, QtGui, QtWidgets

# ===================== CLASE QLabelClickable ======================

class QLabelClickable(QLabel):
    clicked = pyqtSignal(str)
    
    def __init__(self, parent=None):
        super(QLabelClickable, self).__init__(parent)
        self.setMouseTracking(True)
        self.points = []

    def updatePoint(self, points):
        self.points = points

    def mousePressEvent(self, event):
        self.ultimo = self.objectName()

    def mouseReleaseEvent(self, event):
        if self.ultimo == "Clic":
            QTimer.singleShot(QApplication.instance().doubleClickInterval(),
                              self.performSingleClickAction)
        else:
            # Realizar acción de doble clic.
            self.clicked.emit(self.ultimo)
    
    def mouseDoubleClickEvent(self, event):
        self.ultimo = self.objectName()

    def mouseMoveEvent(self, event):
        for index in arange(len(self.points)):
            if self.points[index]["clicked"] == 1:
                self.points[index]["point"].setGeometry(QtCore.QRect(event.x(), event.y(), int(12), int(12)))
                # print("============================")
        # print('Mouse coords: ( %d : %d )' % (event.x(), event.y()))

    def performSingleClickAction(self):
        if self.ultimo == "Clic":
            self.clicked.emit(self.ultimo)