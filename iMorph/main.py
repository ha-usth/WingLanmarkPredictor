from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog
import os
from PyQt5.QtWidgets import QMessageBox
import threading
import imutils
from keypoint_matching import get_true_feature, prediction, prediction_image
from landmark_predictor import predict, extract_samples
from preprocess import align_sample_folder
import cv2
import tkinter as tk
from static import StaticVariable
from PyQt5.QtCore import QTimer
import random
from PyQt5.QtWidgets import (QApplication, QLabel, QWidget)
from QlabelClickAble import QLabelClickable 

root = tk.Tk()

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

print(screen_width, screen_height)

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(int(959 / 1366 * screen_width), int(582 / 768 * screen_height))
        # self.setMouseTracking(True)
        x_start = int(screen_width / 2) - int(959 / 1366 * screen_width / 2)
        y_start = int(screen_height / 2) - int(582 / 768 * screen_height)
        x_start = 0
        y_start = 0
        MainWindow.setGeometry(x_start, y_start, int(820 / 1366 * screen_width), int(582 / 768 * screen_height))
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        
        self.list_train = QtWidgets.QListView(self.centralwidget)
        self.list_train.setGeometry(QtCore.QRect(int(10 / 1366 * screen_width), int(40 / 768 * screen_height), int(161 / 1366 * screen_width), int(231 / 768 * screen_height)))
        self.list_train.setObjectName("list_train")
        self.list_predict = QtWidgets.QListView(self.centralwidget)
        self.list_predict.setGeometry(QtCore.QRect(int(10 / 1366 * screen_width), int(310 / 768 * screen_height), int(161 / 1366 * screen_width), int(251 / 768 * screen_height)))
        self.list_predict.setObjectName("listView")

        self.loadtrain = QtWidgets.QPushButton(self.centralwidget)
        self.loadtrain.setGeometry(QtCore.QRect(int(10 / 1366 * screen_width), int(10 / 768 * screen_height), int(80 / 1366 * screen_width), int(23 / 768 * screen_height)))
        self.loadtrain.setObjectName("loadtrain")
        self.genarate = QtWidgets.QPushButton(self.centralwidget)
        self.genarate.setGeometry(QtCore.QRect(int(91 / 1366 * screen_width), int(10 / 768 * screen_height), int(80 / 1366 * screen_width), int(23 / 768 * screen_height)))
        self.genarate.setObjectName("genarate")
        
        self.loadpredict = QtWidgets.QPushButton(self.centralwidget)
        self.loadpredict.setGeometry(QtCore.QRect(int(10 / 1366 * screen_width), int(280 / 768 * screen_height), int(161 / 1366 * screen_width), int(23 / 768 * screen_height)))
        self.loadpredict.setObjectName("loadpredict")



        self.showimage = QLabelClickable(self.centralwidget) # QtWidgets.QLabel(self.centralwidget)
        self.showimage.setGeometry(QtCore.QRect(int(190 / 1366 * screen_width), int(160 / 768 * screen_height), int(611 / 1366 * screen_width), int(401 / 768 * screen_height)))
        self.showimage.setStyleSheet("border: 3px solid #0B6138;")
        self.showimage.setText("")
        self.showimage.setObjectName("showimage")
        self.showimage.setMouseTracking(True)
        self.train = QtWidgets.QPushButton(self.centralwidget)
        self.train.setGeometry(QtCore.QRect(int(210 / 1366 * screen_width), int(95 / 768 * screen_height), int(120 / 1366 * screen_width), int(31 / 768 * screen_height)))
        self.train.setObjectName("train")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(int(210 / 1366 * screen_width), int(20 / 768 * screen_height), int(71 / 1366 * screen_width), int(21 / 768 * screen_height)))
        self.label_2.setObjectName("label_2")
        
        self.top = QtWidgets.QLabel(self.centralwidget)
        self.top.setGeometry(QtCore.QRect(int(450 / 1366 * screen_width), int(20 / 768 * screen_height), int(130 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.top.setObjectName("top")

        self.topInput = QtWidgets.QLineEdit(self.centralwidget)
        self.topInput.setGeometry(QtCore.QRect(int(470 / 1366 * screen_width), int(20 / 768 * screen_height), int(100 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.topInput.setObjectName("top input")

        self.bottom = QtWidgets.QLabel(self.centralwidget)
        self.bottom.setGeometry(QtCore.QRect(int(590 / 1366 * screen_width), int(20 / 768 * screen_height), int(130 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.bottom.setObjectName("bottom")

        self.bottomInput = QtWidgets.QLineEdit(self.centralwidget)
        self.bottomInput.setGeometry(QtCore.QRect(int(620 / 1366 * screen_width), int(20 / 768 * screen_height), int(100 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.bottomInput.setObjectName("bottom input")

        self.left = QtWidgets.QLabel(self.centralwidget)
        self.left.setGeometry(QtCore.QRect(int(450 / 1366 * screen_width), int(50 / 768 * screen_height), int(130 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.left.setObjectName("left")

        self.leftInput = QtWidgets.QLineEdit(self.centralwidget)
        self.leftInput.setGeometry(QtCore.QRect(int(470 / 1366 * screen_width), int(50 / 768 * screen_height), int(100 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.leftInput.setObjectName("left input")

        self.right = QtWidgets.QLabel(self.centralwidget)
        self.right.setGeometry(QtCore.QRect(int(590 / 1366 * screen_width), int(50 / 768 * screen_height), int(130 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.right.setObjectName("right")

        self.rightInput = QtWidgets.QLineEdit(self.centralwidget)
        self.rightInput.setGeometry(QtCore.QRect(int(620 / 1366 * screen_width), int(50 / 768 * screen_height), int(100 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.rightInput.setObjectName("right input")
        
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(int(210 / 1366 * screen_width), int(60 / 768 * screen_height), int(71 / 1366 * screen_width), int(21 / 768 * screen_height)))
        self.label_3.setObjectName("label_3")

        self.method = QtWidgets.QComboBox(self.centralwidget)
        self.method.setGeometry(QtCore.QRect(int(280 / 1366 * screen_width), int(60 / 768 * screen_height), int(141 / 1366 * screen_width), int(22 / 768 * screen_height)))
        self.method.setObjectName("method")

        self.feature_type_pre = QtWidgets.QComboBox(self.centralwidget)
        self.feature_type_pre.setGeometry(QtCore.QRect(int(280 / 1366 * screen_width), int(20 / 768 * screen_height), int(141 / 1366 * screen_width), int(22 / 768 * screen_height)))
        self.feature_type_pre.setObjectName("feature_type_pre")

        self.windowSize = QtWidgets.QLabel(self.centralwidget)
        self.windowSize.setGeometry(QtCore.QRect(int(470 / 1366 * screen_width), int(80 / 768 * screen_height), int(130 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.windowSize.setObjectName("windowSize")

        self.windowSizeInput = QtWidgets.QLineEdit(self.centralwidget)
        self.windowSizeInput.setGeometry(QtCore.QRect(int(580 / 1366 * screen_width), int(80 / 768 * screen_height), int(140 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.windowSizeInput.setObjectName("window size input")

        self.numOfrandomPoint = QtWidgets.QLabel(self.centralwidget)
        self.numOfrandomPoint.setGeometry(QtCore.QRect(int(470 / 1366 * screen_width), int(100 / 768 * screen_height), int(130 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.numOfrandomPoint.setObjectName("numOfrandomPoint")

        self.numOfrandomPointInput = QtWidgets.QLineEdit(self.centralwidget)
        self.numOfrandomPointInput.setGeometry(QtCore.QRect(int(580 / 1366 * screen_width), int(100 / 768 * screen_height), int(140 / 1366 * screen_width), int(16 / 768 * screen_height)))
        self.numOfrandomPointInput.setObjectName("numOfrandomPoint input")
        
        self.predict = QtWidgets.QPushButton(self.centralwidget)
        self.predict.setGeometry(QtCore.QRect(int(330 / 1366 * screen_width), int(95 / 768 * screen_height), int(120 / 1366 * screen_width), int(31 / 768 * screen_height)))
        self.predict.setObjectName("predict")
        self.status = QtWidgets.QLabel(self.centralwidget)
        self.status.setGeometry(QtCore.QRect(int(190 / 1366 * screen_width), int(140 / 768 * screen_height), int(150 / 1366 * screen_width), int(20 / 768 * screen_height)))
        self.status.setObjectName("status")

        self.show_center_point = QtWidgets.QCheckBox(self.centralwidget)
        self.show_center_point.setGeometry(QtCore.QRect(int(350 / 1366 * screen_width), int(140 / 768 * screen_height), int(150 / 1366 * screen_width), int(20 / 768 * screen_height)))
        self.show_center_point.setObjectName("center_point")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        MainWindow.setMaximumSize(int(959 / 1366 * screen_width), int(582 / 768 * screen_height))

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

        self.folder_train = ""
        self.folder_predict = ""
        self.feature_type_init = None
        self.method_init = None
        self.windown_size_init = None
        self.threshold_init = None
        self.scale_ratio_init = None
        self.num_random =None
        self.center_points = None
        self.true_features = None
        self.isTraining = 0 # 0 chưa train , 1 đang traing , 2 đã training xong
        self.length_train = 0
        self.length_predict = 0
        self.img_select = None
        self.isPreprocess = False

        self.event()
        self.setup()

    def changeShowCenterPoint(self, state):
        if state == QtCore.Qt.Checked:
            if not self.show_rotage:
                self.draw_image_predict(self.file_predict_show, True)
            else:
                self.draw_image_predict(self.file_predict_show_rotage, True)
        else:
            if not self.show_rotage:
                self.draw_image_predict(self.file_predict_show, False)
            else:
                self.draw_image_predict(self.file_predict_show_rotage, False)
            # self.draw_image_predict(self.file_predict_show, False)

    def setup(self):
        self.feature_type_pre.addItem("akaze")
        self.feature_type_pre.addItem("brisk")
        self.feature_type_pre.addItem("surf")

        self.method.addItem("keypoint")
        self.method.addItem("random_point")
        self.method.addItem("dense")

        self.timerTrain = QTimer()
        self.timerTrain.timeout.connect(self.check_train)

        self.timerPredict = QTimer()
        self.timerPredict.timeout.connect(self.check_predict)

        self.predict.setEnabled(False)

        self.file_predict_show = None
        self.file_text_predict = None

        self.file_predict_show_rotage = None

        self.show_rotage = False

        self.line_predict = None

    def check_predict(self):
        self.timerTrain.stop()
        text = "Predicting {0}/{1}".format(StaticVariable.predicting_size, self.length_predict)
        self.status.setText(text)
        self.status.repaint()

    def check_train(self):
        self.timerPredict.stop()
        text = "Training {0}/{1}".format(StaticVariable.training_size, self.length_train)
        self.status.setText(text)
        self.status.repaint()

    def event(self):
        self.loadtrain.clicked.connect(self.choose_folder_train)
        self.genarate.clicked.connect(self.preprocess_data)
        self.list_train.clicked[QtCore.QModelIndex].connect(self.choose_file)
        self.loadpredict.clicked.connect(self.choose_folder_predict)
        self.list_predict.clicked[QtCore.QModelIndex].connect(self.choose_file_predict)
        self.train.clicked.connect(self.training)
        self.predict.clicked.connect(self.predict_model)
        self.method.view().pressed.connect(self.change_method)
        self.show_center_point.stateChanged.connect(self.changeShowCenterPoint)

    def save_image_rotate_image(self):
        name = '.'.join(self.file_predict_show.split('.')[0: -1]) + ".txt"
        contents = open(self.folder_predict + "/tmp/" + name, "r").read()
        img = cv2.imread(self.folder_predict + "/tmp/" + self.file_predict_show)
        try:
            file = open(self.folder_predict + "/" + name, "w")
            file.write(contents)
            cv2.imwrite(self.folder_predict + "/" + self.file_predict_show, img)
            self.showDialog("SUCCESS", "Save file success")
        except :
            self.showDialog("ERROR", "Save file fail")

    def preprocess_data(self):
        if self.img_select is None:
            self.showDialog("ERROR", "Bạn cần chọn ảnh mẫu trong danh sách")
            return
        path = self.folder_train + "/" + self.img_select
        folder_save = self.folder_train + "/gen/"
        if not os.path.isdir(folder_save):
            os.makedirs(folder_save)
        self.genarate.setText("Preprocessing...")
        self.genarate.setEnabled(False)
        threading.Thread(target=self.genimage, args=(self.folder_train + "/", folder_save, path)).start()

    def genimage(self, folder_train, folder_save, path):
        align_sample_folder(folder_train, folder_save, path)
        self.isPreprocess = True
        self.genarate.setEnabled(True)
        self.genarate.setText("Preprocess")

    def change_method(self, index):
        item = self.method.model().itemFromIndex(index)
        if item.text() == "keypoint":
            pass
        elif item.text() == "random_point":
            pass
        elif item.text() == "dense":
            pass

    def get_value_predict(self):
        self.get_value()
        check_error = False
        mess = ""
        if not len(self.folder_predict):
            check_error = True
            mess += "Bạn chưa chọn folder predict . \n"
        if not self.windown_size_init.isdigit():
            mess += "Bạn nhập windown size chưa đúng . \n"
            check_error = True
        else:
            self.windown_size_init = int(self.windown_size_init)
        if self.method_init == "random_point":
            if not self.num_random.isdigit():
                mess += "Bạn nhập num random chưa đúng . \n"
                check_error = True
            else:
                self.num_random = int(self.num_random)
        
        if self.method_init == "keypoint":
            try:
                self.threshold_init = float(self.threshold_init)
            except :
                mess += "Bạn nhập threshold chưa đúng . \n"
                check_error = True
        
        try:
            self.scale_ratio_init = float(self.scale_ratio_init)
        except:
            mess += "Bạn nhập ratio chưa đúng . \n"
            check_error = True
        if self.method_init == "dense":
            if not self.dense_step_init.isdigit():
                mess += "Bạn nhập dense step chưa đúng . \n"
                check_error = True
            else:
                self.dense_step_init = int(self.dense_step_init)

        return check_error, mess

    def predict_model(self):
        StaticVariable.predicting_size = 0
        
        check_error, mess = self.get_value_predict()

        if check_error:
            self.showDialog("ERROR", mess)
            return

        self.predict.setEnabled(False)
        self.train.setEnabled(False)
        self.loadpredict.setEnabled(False)
        self.predict.setText("Predicting...")
        self.predict.repaint()
        t = threading.Thread(target=self.handle_predecting)
        t.start()
        self.timerPredict.start()

    def handle_predecting(self):
        prediction(self.folder_predict, self.true_features, self.feature_type_init, self.method_init, self.center_points, self.windown_size_init, self.num_random, self.threshold_init, self.dense_step_init, self.scale_ratio_init)
        self.predict.setEnabled(True)
        self.predict.setText("Predict landmark")
        self.train.setEnabled(True)
        self.loadpredict.setEnabled(True)

    def showDialog(self, title, mess):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(mess)
        msgBox.setWindowTitle(title)
        msgBox.exec()

    def get_value(self):
        self.feature_type_init = str(self.feature_type_pre.currentText())
        self.method_init = str(self.method.currentText())
        self.scale_ratio_init = self.scale_ratio.text()
        self.dense_step_init = self.dense_step.text()

    def handle_training(self):
        if self.isPreprocess:
            folder = self.folder_train + "/gen"
        else:
            folder = self.folder_train
        self.isTraining = 1
        self.predict.setEnabled(False)
        self.predict.repaint()
        # self.center_points, self.true_features = get_true_feature(folder, self.feature_type_init, self.scale_ratio_init)
        self.center_points, self.true_features = extract_samples(folder)
        self.train.setEnabled(True)
        self.isTraining = 1
        self.predict.setEnabled(True)
        self.train.setText("Extract sample data")
        self.loadtrain.setEnabled(True)

    def training(self):
        StaticVariable.training_size = 0
        self.get_value()
        mess = ""
        if not len(self.folder_train):
            mess += "Bạn chưa load folder train \n"
        try:
            self.scale_ratio_init = float(self.scale_ratio_init)
        except:
            mess += "Bạn cần nhập scale ratio \n"
        if len(mess):
            self.showDialog("ERROR", mess)
            return 
        self.train.setEnabled(False)
        self.train.setText("Training...")
        self.train.repaint()
        self.loadtrain.setEnabled(False)
        t = threading.Thread(target=self.handle_training)
        t.start()
        self.timerTrain.start(1000)

    def draw_image_predict(self, filename, isShowCenter, isDrawPoint=True):
        if filename is None:
            return
        name = ".".join(filename.split('.')[0: -1])
        path_file_text = self.folder_predict + "/" + name + ".txt"
        fileimage = self.folder_predict + "/" + filename
        img = cv2.imread(fileimage)
        if img is not None and os.path.isfile(path_file_text) and isDrawPoint:
            filetext = open(path_file_text, "r")
            contents = filetext.read()
            lines = contents.split("\n")
            index = 0
            for line in lines:
                if len(line) > 0:
                    x, y = int(float(line.split(" ")[0])), int(float(line.split(" ")[1]))
                    cv2.circle(img, (x, y), 7, (0, 0, 255), 4)
                    cv2.putText(img, str(index), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 0), 2, cv2.LINE_AA)
                    index += 1
            if isShowCenter:
                windown_size = int(self.wsize.text())
                scale_ratio = float(self.scale_ratio.text())
                index = 0
                try:
                    for center in self.center_points:
                        x, y = int(center.x / scale_ratio) , int(center.y / scale_ratio)
                        cv2.circle(img, (x, y), 7, (255, 0, 0), 4)
                        cv2.putText(img, str(index), (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX,  1, (255, 0, 0), 2, cv2.LINE_AA)
                        cv2.rectangle(img, (x - int(windown_size), y - int(windown_size)), (x + int(windown_size), y + int(windown_size)), (255, 0, 0), 2)
                        index += 1
                    self.showimage.updatePoint(self.listPoint)
                except:
                    self.showDialog("ERROR", "Bạn chưa tranning dữ liệu")
                    pass
        self.show_image_to_label(img)

        x = random.randint(int(190 / 1366 * screen_width), int(611 / 1366 * screen_width))
        y = random.randint(int(160 / 768 * screen_height), int(401 / 768 * screen_height))
        self.point = QLabelClickable(self.showimage)
        self.point.setGeometry(QtCore.QRect(x, y, int(12), int(12)))
        self.point.setStyleSheet("border: 3px solid red; border-radius: 6px")
        self.point.setText("")
        self.point.setObjectName(str(index))
        self.point.clicked.connect(self.movePoint)
        self.listPoint.append({"point" : self.point, "clicked" : 0})

    def choose_file_predict(self, index):
        item = self.model_predict_file.itemFromIndex(index).text()
        ext = item.split('.')[-1]
        if ext != "txt":
            self.file_predict_show = item
            isShowCenter = False
            if self.show_center_point.isChecked():
                isShowCenter = True
            else:
                isShowCenter = False
            self.draw_image_predict(self.file_predict_show, isShowCenter)
            self.show_rotage = False
        else:
            content_text = ""
            self.file_text_predict = item
            filetext = self.folder_predict + "/" + item
            filecontent = open(filetext, "r").read()
            lines = filecontent.split("\n")
            for line in lines:
                try:
                    x, y = line.split(" ")[0], line.split(" ")[1]
                    x = x if len(x) < 6 else x[0:5]
                    y = y if len(y) < 6 else y[0:5]
                    content_text += "{0} {1}\n".format(x, y)
                except:
                    pass

    def choose_folder_predict(self):
        dialog = QFileDialog()
        foo_dir = dialog.getExistingDirectory(None, 'Select an folder predict')
        if not len(foo_dir):
            return
        self.length_predict = 0
        self.folder_predict = foo_dir
        self.model_predict_file = QtGui.QStandardItemModel()
        self.list_predict.setModel(self.model_predict_file)
        for file in os.listdir(foo_dir):
            ext = file.split(".")[-1]
            name = '.'.join(file.split(".")[0: -1]) + ".txt"
            if ext in ["bmp", "png", "jpg", "jpeg", "tif"]:
                item = QtGui.QStandardItem(file)
                self.length_predict += 1
                self.model_predict_file.appendRow(item)
                if os.path.isfile(self.folder_predict + "/" + name):
                    itemtext = QtGui.QStandardItem(name)
                    self.model_predict_file.appendRow(itemtext)

    def choose_folder_train(self):
        dialog = QFileDialog()
        foo_dir = dialog.getExistingDirectory(None, 'Select an folder trainning')
        if not len(foo_dir):
            return
        self.length_train = 0
        self.folder_train= foo_dir
        self.model_train_file = QtGui.QStandardItemModel()
        self.list_train.setModel(self.model_train_file)
        for file in os.listdir(foo_dir):
            ext = file.split(".")[-1]
            filetext = foo_dir + "/" + '.'.join(file.split(".")[0:-1]) + ".txt"
            if ext in ["bmp", "png", "jpg", "jpeg", "tif"] and os.path.isfile(filetext):
                item = QtGui.QStandardItem(file)
                self.length_train += 1
                self.model_train_file.appendRow(item)

    def choose_file(self, index):
        item = self.model_train_file.itemFromIndex(index).text()
        self.img_select = item
        self.load_image_train(item)

    def load_image_train(self, filename):
        name = '.'.join(filename.split(".")[0: -1])
        image_path = self.folder_train + "/" + filename
        img = cv2.imread(image_path)
        if img is not None:
            filetext = open(self.folder_train + "/" + name + ".txt", "r")
            contents = filetext.read()
            lines = contents.split("\n")
            index = 0
            for line in lines:
                if len(line) > 0:
                    x, y = int(line.split(" ")[0]), int(line.split(" ")[1])
                    cv2.circle(img, (x, y), 7, (0, 0, 255), 4)
                    cv2.putText(img, str(index), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX,  1, (0, 0, 0), 2, cv2.LINE_AA)
                    index += 1
        self.show_image_to_label(img)


    def show_image_to_label(self, image):
        if image is None:
            return
        w_label , h_lable = int(611 / 1366 * screen_width), int(401 / 768 * screen_height)
        rgbImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        convertToQtFormat = QtGui.QImage(rgbImage.data, rgbImage.shape[1], rgbImage.shape[0], QtGui.QImage.Format_RGB888)
        convertToQtFormat = QtGui.QPixmap.fromImage(convertToQtFormat)
        pixmap = QtGui.QPixmap(convertToQtFormat)
        resizeImage = pixmap.scaled(w_label, h_lable, QtCore.Qt.KeepAspectRatio)
        self.showimage.setPixmap(resizeImage)
        
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.loadtrain.setText(_translate("MainWindow", "Load sample data"))
        self.genarate.setText(_translate("MainWindow", "Perprocess data"))
        self.loadpredict.setText(_translate("MainWindow", "Load prediction data"))
        self.train.setText(_translate("MainWindow", "Extract sample data"))
        self.label_2.setText(_translate("MainWindow", "Feature type"))
        self.label_3.setText(_translate("MainWindow", "Cadidate method"))
        self.top.setText(_translate("MainWindow", "Top"))
        self.bottom.setText(_translate("MainWindow", "Bottom"))
        self.left.setText(_translate("MainWindow", "Left"))
        self.right.setText(_translate("MainWindow", "Right"))
        self.windowSize.setText(_translate("MainWindow", "Windows size"))
        self.numOfrandomPoint.setText(_translate("MainWindow", "Number of random points"))
        self.predict.setText(_translate("MainWindow", "Predict folder"))
        self.show_center_point.setText(_translate("MainWindow", "Show center points"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
