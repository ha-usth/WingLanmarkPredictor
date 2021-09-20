# -*- coding: utf-8 -*-
"""
Created on Mon Sep 13 22:44:06 2021

@author: duonghv
"""

# from image_viewer import *
from PyQt5.QtWidgets import QApplication, QDialog, QGraphicsScene, QGraphicsEllipseItem, QGraphicsPixmapItem, QWidget
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt, QPointF
import os
import sys

class MovingObject(QGraphicsEllipseItem):
    def __init__(self, x, y, r):
        super().__init__(0, 0, r, r)
        self.setPos(x, y)
        self.setBrush(Qt.red)
        self.x = x
        self.y = y
        self.setAcceptHoverEvents(True)
    # mouse hover event    
    def hoverEnterEvent(self, event):
        pass
        # app.instance().setOverrideCursor(Qt.OpenHandCursor)
        
    def hoverLeaveEvent(self, event):
        pass
        # app.instance().restoreOverrideCursor()
        
    # mouse click event
    def mousePressEvent(self, event):
        pass
        # print('x: {0}, y: {1}'.format(self.pos().x(), self.pos().y()))
    
    def mouseMoveEvent(self, event):
        orig_cursor_position = event.lastScenePos()
        updated_cursor_position = event.scenePos()
        
        orig_position = self.scenePos()
        
        updated_cursor_x = updated_cursor_position.x() - orig_cursor_position.x() + orig_position.x()
        updated_cursor_y = updated_cursor_position.y() - orig_cursor_position.y() + orig_position.y()
        self.setPos(QPointF(updated_cursor_x, updated_cursor_y))

    def getPoint(self):
        return self.x, self.y
    
    def mouseReleaseEvent(self, event):
        self.x = self.pos().x()
        self.y = self.pos().y()
        # print('x: {0}, y: {1}'.format(self.pos().x(), self.pos().y()))

class Canvas(QWidget):
    def __init__(self):
        super().__init__()

    def loadImage(self):
        image_path = "C:/Users/Do Phong PC/Desktop/photo_2021-09-13_19-24-17.jpg"
        if os.path.isfile(image_path):
            scene = QGraphicsScene(self)
            # scene
            pixmap = QPixmap(image_path)
            item = QGraphicsPixmapItem(pixmap)
            scene.addItem(item)
            
            moveObject = MovingObject(30, 30, 10)
            scene.addItem(moveObject)
            
            self.ui.graphicsView.setScene(scene)