# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QWidget, QGridLayout, QPushButton, QSizePolicy, QApplication
from PyQt5.QtWidgets import QFileDialog
import os
from PyQt5.QtWidgets import QMessageBox
from preprocess import align_sample_folder
from landmark_predictor import extract_samples, predict, predict_folder
from feature_extractors import *
import threading
import json
import time

class Ui_iMorph(object):
    def setupUi(self, iMorph):
        iMorph.setObjectName("iMorph")
        iMorph.resize(1098, 790)
        self.groupBox = QtWidgets.QGroupBox(iMorph)
        self.groupBox.setGeometry(QtCore.QRect(10, 10, 271, 361))
        self.groupBox.setStyleSheet("border: 1px solid black;")
        self.groupBox.setObjectName("groupBox")
        self.listTrain = QtWidgets.QListView(self.groupBox)
        self.listTrain.setGeometry(QtCore.QRect(10, 30, 241, 271))
        self.listTrain.setObjectName("listTrain")
        self.loadTrain = QtWidgets.QPushButton(self.groupBox)
        self.loadTrain.setGeometry(QtCore.QRect(10, 320, 75, 23))
        self.loadTrain.setObjectName("loadTrain")
        self.preprocess = QtWidgets.QPushButton(self.groupBox)
        self.preprocess.setGeometry(QtCore.QRect(90, 320, 75, 23))
        self.preprocess.setObjectName("preprocess")
        self.train = QtWidgets.QPushButton(self.groupBox)
        self.train.setGeometry(QtCore.QRect(170, 320, 75, 23))
        self.train.setObjectName("train")
        self.groupBox_2 = QtWidgets.QGroupBox(iMorph)
        self.groupBox_2.setGeometry(QtCore.QRect(10, 390, 271, 381))
        self.groupBox_2.setStyleSheet("border: 1px solid black;")
        self.groupBox_2.setObjectName("groupBox_2")
        self.listPredict = QtWidgets.QListView(self.groupBox_2)
        self.listPredict.setGeometry(QtCore.QRect(10, 30, 241, 301))
        self.listPredict.setObjectName("listPredict")
        self.loadPredict = QtWidgets.QPushButton(self.groupBox_2)
        self.loadPredict.setGeometry(QtCore.QRect(10, 340, 75, 23))
        self.loadPredict.setObjectName("loadPredict")
        self.predict = QtWidgets.QPushButton(self.groupBox_2)
        self.predict.setGeometry(QtCore.QRect(170, 340, 75, 23))
        self.predict.setObjectName("predict")
        self.groupBox_3 = QtWidgets.QGroupBox(iMorph)
        self.groupBox_3.setGeometry(QtCore.QRect(300, 10, 781, 251))
        self.groupBox_3.setStyleSheet("border: 1px solid black;")
        self.groupBox_3.setObjectName("groupBox_3")
        self.groupBox_4 = QtWidgets.QGroupBox(self.groupBox_3)
        self.groupBox_4.setGeometry(QtCore.QRect(30, 30, 721, 91))
        self.groupBox_4.setObjectName("groupBox_4")
        self.label = QtWidgets.QLabel(self.groupBox_4)
        self.label.setGeometry(QtCore.QRect(20, 30, 47, 31))
        self.label.setStyleSheet("border: none;")
        self.label.setObjectName("label")
        self.top = QtWidgets.QLineEdit(self.groupBox_4)
        self.top.setGeometry(QtCore.QRect(70, 30, 71, 31))
        self.top.setObjectName("top")
        self.label_2 = QtWidgets.QLabel(self.groupBox_4)
        self.label_2.setGeometry(QtCore.QRect(200, 30, 47, 31))
        self.label_2.setStyleSheet("border: none;")
        self.label_2.setObjectName("label_2")
        self.bottom = QtWidgets.QLineEdit(self.groupBox_4)
        self.bottom.setGeometry(QtCore.QRect(250, 30, 71, 31))
        self.bottom.setObjectName("bottom")
        self.left = QtWidgets.QLineEdit(self.groupBox_4)
        self.left.setGeometry(QtCore.QRect(430, 30, 71, 31))
        self.left.setObjectName("left")
        self.label_3 = QtWidgets.QLabel(self.groupBox_4)
        self.label_3.setGeometry(QtCore.QRect(380, 30, 47, 31))
        self.label_3.setStyleSheet("border: none;")
        self.label_3.setObjectName("label_3")
        self.right = QtWidgets.QLineEdit(self.groupBox_4)
        self.right.setGeometry(QtCore.QRect(610, 30, 71, 31))
        self.right.setObjectName("right")
        self.label_4 = QtWidgets.QLabel(self.groupBox_4)
        self.label_4.setGeometry(QtCore.QRect(560, 30, 47, 31))
        self.label_4.setStyleSheet("border: none;")
        self.label_4.setObjectName("label_4")
        self.label_5 = QtWidgets.QLabel(self.groupBox_3)
        self.label_5.setGeometry(QtCore.QRect(50, 150, 71, 31))
        self.label_5.setStyleSheet("border: none;")
        self.label_5.setObjectName("label_5")
        self.featureTypeCom = QtWidgets.QComboBox(self.groupBox_3)
        self.featureTypeCom.setGeometry(QtCore.QRect(160, 150, 141, 22))
        self.featureTypeCom.setObjectName("featureTypeCom")
        self.label_6 = QtWidgets.QLabel(self.groupBox_3)
        self.label_6.setGeometry(QtCore.QRect(50, 190, 101, 31))
        self.label_6.setStyleSheet("border: none;")
        self.label_6.setObjectName("label_6")
        self.candidateCom = QtWidgets.QComboBox(self.groupBox_3)
        self.candidateCom.setGeometry(QtCore.QRect(160, 190, 141, 22))
        self.candidateCom.setObjectName("candidateCom")
        self.label_7 = QtWidgets.QLabel(self.groupBox_3)
        self.label_7.setGeometry(QtCore.QRect(380, 150, 91, 31))
        self.label_7.setStyleSheet("border: none;")
        self.label_7.setObjectName("label_7")
        self.windowSize = QtWidgets.QLineEdit(self.groupBox_3)
        self.windowSize.setGeometry(QtCore.QRect(530, 150, 113, 31))
        self.windowSize.setObjectName("windowSize")
        self.label_8 = QtWidgets.QLabel(self.groupBox_3)
        self.label_8.setGeometry(QtCore.QRect(380, 190, 141, 31))
        self.label_8.setStyleSheet("border: none;")
        self.label_8.setObjectName("label_8")
        self.numOfPoint = QtWidgets.QLineEdit(self.groupBox_3)
        self.numOfPoint.setGeometry(QtCore.QRect(530, 190, 113, 31))
        self.numOfPoint.setObjectName("numOfPoint")
        self.viewImage = QtWidgets.QLabel(iMorph)
        self.viewImage.setGeometry(QtCore.QRect(300, 265, 781, 471))
        self.viewImage.setStyleSheet("border: 1px solid black;")
        self.viewImage.setText("")
        self.viewImage.setObjectName("viewImage")
        self.showCenter = QtWidgets.QCheckBox(iMorph)
        self.showCenter.setGeometry(QtCore.QRect(410, 750, 111, 17))
        self.showCenter.setObjectName("showCenter")
        self.save = QtWidgets.QPushButton(iMorph)
        self.save.setGeometry(QtCore.QRect(770, 745, 81, 23))
        self.save.setObjectName("save")

        self.retranslateUi(iMorph)
        QtCore.QMetaObject.connectSlotsByName(iMorph)
        self.event()
        self.setUp()


        self.img_select = None
        self.imgGenarate = False
        self.folder_train = None
        self.center_points = None
        self.true_features = None
        self.folder_predict = None

        self.topValue = None
        self.bottomValue = None
        self.leftValue = None
        self.rightValue = None
        self.windowSizeValue = None
        self.numRandomValue = None

        self.featureExtractor = None
        self.candidate_methodValue = None

        self.configs = None
        self.readFileConfig()
        

    def resize(self, width, height):
        self.groupBox.setGeometry(QtCore.QRect(int(10 / 1098 * width),int(10 / 790 * height),int(271 / 1098 * width) ,int(361 / 790 * height) ))
        self.listTrain.setGeometry(QtCore.QRect((10 / 1098 * width),int(30 / 790 * height),int(241 / 1098 * width) ,int(271 / 790 * height)))
        self.loadTrain.setGeometry(QtCore.QRect((10 / 1098 * width),int(320 / 790 * height) ,int(75 / 1098 * width) ,int(23 / 790 * height) ))
        self.preprocess.setGeometry(QtCore.QRect((90 / 1098 * width),int(320 / 790 * height) , int(75 / 1098 * width), int(23 / 790 * height)))
        self.train.setGeometry(QtCore.QRect((170 / 1098 * width), int(320 / 790 * height), int(75 / 1098 * width), int(23 / 790 * height)))
        self.groupBox_2.setGeometry(QtCore.QRect((10 / 1098 * width), int(390 / 790 * height), int(271 / 1098 * width), int(381 / 790 * height)))
        self.listPredict.setGeometry(QtCore.QRect((10 / 1098 * width), int(30 / 790 * height), int(241 / 1098 * width), int(301 / 790 * height)))
        self.loadPredict.setGeometry(QtCore.QRect((10 / 1098 * width), int(340 / 790 * height), int(75 / 1098 * width), int(23 / 790 * height)))
        self.predict.setGeometry(QtCore.QRect((150 / 1098 * width), int(340 / 790 * height), int(95 / 1098 * width), int(23 / 790 * height)))
        self.groupBox_3.setGeometry(QtCore.QRect((300 / 1098 * width), int(10 / 790 * height), int(781 / 1098 * width), int(251 / 790 * height)))
        self.groupBox_4.setGeometry(QtCore.QRect((30 / 1098 * width), int(30 / 790 * height), int(721 / 1098 * width), int(91 / 790 * height)))
        self.label.setGeometry(QtCore.QRect((20 / 1098 * width), int(30 / 790 * height), int(47 / 1098 * width), int(31 / 790 * height)))
        self.top.setGeometry(QtCore.QRect((70 / 1098 * width), int(30 / 790 * height),  int(71 / 1098 * width), int(31 / 790 * height)))
        self.label_2.setGeometry(QtCore.QRect((200 / 1098 * width), int(30 / 790 * height), int(47 / 1098 * width), int(31 / 790 * height)))
        self.bottom.setGeometry(QtCore.QRect((250 / 1098 * width), int(30 / 790 * height), int(71 / 1098 * width), int(31 / 790 * height)))
        self.left.setGeometry(QtCore.QRect((430 / 1098 * width), int(30 / 790 * height), int(71 / 1098 * width), int(31 / 790 * height)))
        self.label_3.setGeometry(QtCore.QRect((380 / 1098 * width), int(30 / 790 * height), int(47 / 1098 * width), int(31 / 790 * height)))
        self.right.setGeometry(QtCore.QRect((610 / 1098 * width), int(30 / 790 * height), int(71 / 1098 * width), int(31 / 790 * height)))
        self.label_4.setGeometry(QtCore.QRect((560 / 1098 * width), int(30 / 790 * height), int(47 / 1098 * width), int(31 / 790 * height)))
        self.label_5.setGeometry(QtCore.QRect((50 / 1098 * width), int(150 / 790 * height), int(71 / 1098 * width), int(31 / 790 * height)))
        self.featureTypeCom.setGeometry(QtCore.QRect((160 / 1098 * width), int(150 / 790 * height), int(141 / 1098 * width), int(22 / 790 * height)))
        self.label_6.setGeometry(QtCore.QRect((50 / 1098 * width), int(190 / 790 * height), int(101 / 1098 * width), int(31 / 790 * height)))
        self.candidateCom.setGeometry(QtCore.QRect((160 / 1098 * width), int(190 / 790 * height), int(141 / 1098 * width), int(22 / 790 * height)))
        self.label_7.setGeometry(QtCore.QRect((380 / 1098 * width), int(150 / 790 * height), int(91 / 1098 * width), int(31 / 790 * height)))
        self.windowSize.setGeometry(QtCore.QRect((530 / 1098 * width), int(150 / 790 * height), int(113 / 1098 * width), int(31 / 790 * height)))
        self.label_8.setGeometry(QtCore.QRect((380 / 1098 * width), int(190 / 790 * height), int(141 / 1098 * width), int(31 / 790 * height)))
        self.numOfPoint.setGeometry(QtCore.QRect((530 / 1098 * width), int(190 / 790 * height), int(113 / 1098 * width), int(31 / 790 * height)))
        self.viewImage.setGeometry(QtCore.QRect((300 / 1098 * width), int(265 / 790 * height), int(781 / 1098 * width), int(471 / 790 * height)))
        self.showCenter.setGeometry(QtCore.QRect((410 / 1098 * width), int(750 / 790 * height), int(111 / 1098 * width), int(17 / 790 * height)))
        self.save.setGeometry(QtCore.QRect((770 / 1098 * width), int(745 / 790 * height), int(81 / 1098 * width), int(23 / 790 * height)))

    def event(self):
        self.loadTrain.clicked.connect(self.loadFolderTrain)
        self.listTrain.clicked[QtCore.QModelIndex].connect(self.choose_file)
        self.preprocess.clicked.connect(self.preprocessing)
        self.train.clicked.connect(self.training)
        self.loadPredict.clicked.connect(self.loadFolderPredict)
        self.predict.clicked.connect(self.predicting)

    # candidate_method = "keypoint" | "keypoint_on_bin_img" | "random" | "gaussian"
    def setUp(self):
        self.top.setText("0")
        self.left.setText("0")
        self.bottom.setText("0")
        self.right.setText("0")

        self.featureTypeCom.addItem("LBP")
        self.featureTypeCom.addItem("HOG")
        self.featureTypeCom.addItem("CHOG")
        self.featureTypeCom.setCurrentIndex(2)

        self.candidateCom.addItem("keypoint")
        self.candidateCom.addItem("keypoint_on_bin_img")
        self.candidateCom.addItem("random")
        self.candidateCom.addItem("gaussian")

        self.windowSize.setText("60")
        self.numOfPoint.setText("100")

    def getValue(self):
        if len(self.top.text()) > 0:
            try:
                self.topValue = int(self.top.text())
            except:
                self.showDialog("ERROR", "Gi?? tr??? top kh??ng h???p l???")
                return False
        else:
            self.topValue = None

        if len(self.bottom.text()) > 0:
            try:
                self.bottomValue = int(self.bottom.text())
            except:
                self.showDialog("ERROR", "Gi?? tr??? bottom kh??ng h???p l???")
                return False
        else:
            self.bottomValue = None

        if len(self.left.text()) > 0:
            try:
                self.leftValue = int(self.left.text())
            except:
                self.showDialog("ERROR", "Gi?? tr??? left kh??ng h???p l???")
                return False
        else:
            self.leftValue = None

        if len(self.right.text()) > 0:
            try:
                self.rightValue = int(self.right.text())
            except:
                self.showDialog("ERROR", "Gi?? tr??? right kh??ng h???p l???")
                return False
        else:
            self.rightValue = None

        try:
            self.windowSizeValue = int (self.windowSize.text())
        except:
            self.windowSizeValue = 60
            pass
    
        try:
            self.numRandomValue = int(self.numOfPoint.text())
        except:
            self.numRandomValue = 100
            pass
        
        return True

    def readFileConfig(self):
        with open("conf.json", "r") as stream:
            try:
                self.configs = json.loads(stream.read())
                return True
            except:
                self.configs = None
                self.showDialog("ERROR", "L???i x???y ra trong l??c ?????c file config")
                return False


    def loadFolderPredict(self):
        dialog = QFileDialog()
        foo_dir = dialog.getExistingDirectory(None, 'Select an folder predict')
        if not len(foo_dir):
            return
        self.length_predict = 0
        self.folder_predict = foo_dir
        self.model_predict_file = QtGui.QStandardItemModel()
        self.listPredict.setModel(self.model_predict_file)
        for file in os.listdir(foo_dir):
            ext = file.split(".")[-1]
            if ext in ["bmp", "png", "jpg", "jpeg", "tif"]:
                item = QtGui.QStandardItem(file)
                self.length_predict += 1
                self.model_predict_file.appendRow(item)

    def choose_file(self, index):
        item = self.model_train_file.itemFromIndex(index).text()
        self.img_select = item

    def loadFolderTrain(self):
        dialog = QFileDialog()
        foo_dir = dialog.getExistingDirectory(None, 'Select an folder trainning')
        if not len(foo_dir):
            return
        self.length_train = 0
        self.folder_train= foo_dir
        self.imgGenarate = False
        self.model_train_file = QtGui.QStandardItemModel()
        self.listTrain.setModel(self.model_train_file)
        for file in os.listdir(foo_dir):
            ext = file.split(".")[-1]
            filetext = foo_dir + "/" + '.'.join(file.split(".")[0:-1]) + ".txt"
            if ext in ["bmp", "png", "jpg", "jpeg", "tif"] and os.path.isfile(filetext):
                item = QtGui.QStandardItem(file)
                self.length_train += 1
                self.model_train_file.appendRow(item)
    
    def showDialog(self, title, mess):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Information)
        msgBox.setText(mess)
        msgBox.setWindowTitle(title)
        msgBox.exec()
    
    def preprocessing(self):
        if self.img_select is None:
            self.showDialog("ERROR", "B???n c???n ch???n file m???u tr??n danh s??ch")
            return
        path = self.folder_train + "/" + self.img_select
        folder_save = self.folder_train + "/gen/"
        if not os.path.isdir(folder_save):
            os.makedirs(folder_save)
        
        self.preprocess.setText("Pre-processing...")
        self.preprocess.setEnabled(False)
        threading.Thread(target=self.genimage, args=(self.folder_train + "/", folder_save, path)).start()

    def genimage(self, folder_train, folder_save, path):
        align_sample_folder(folder_train, folder_save, path)
        self.isPreprocess = True
        self.preprocess.setEnabled(True)
        self.preprocess.setText("Pre-process")
        self.imgGenarate = True


    def retranslateUi(self, iMorph):
        _translate = QtCore.QCoreApplication.translate
        iMorph.setWindowTitle(_translate("iMorph", "iMorph"))
        self.groupBox.setTitle(_translate("iMorph", "Sample data"))
        self.loadTrain.setText(_translate("iMorph", "Load"))
        self.preprocess.setText(_translate("iMorph", "Pre-process"))
        self.train.setText(_translate("iMorph", "Train"))
        self.groupBox_2.setTitle(_translate("iMorph", "Prediction data"))
        self.loadPredict.setText(_translate("iMorph", "Load"))
        self.predict.setText(_translate("iMorph", "Predict"))
        self.groupBox_3.setTitle(_translate("iMorph", "Paramaters"))
        self.groupBox_4.setTitle(_translate("iMorph", "Border size of key point matching region used for image alignment"))
        self.label.setText(_translate("iMorph", "Top"))
        self.label_2.setText(_translate("iMorph", "Bottom"))
        self.label_3.setText(_translate("iMorph", "Left"))
        self.label_4.setText(_translate("iMorph", "Right"))
        self.label_5.setText(_translate("iMorph", "Feature type"))
        self.label_6.setText(_translate("iMorph", "Candidate method"))
        self.label_7.setText(_translate("iMorph", "Windows size"))
        self.label_8.setText(_translate("iMorph", "Number of random points"))
        self.showCenter.setText(_translate("iMorph", "Show center point"))
        self.save.setText(_translate("iMorph", "Save"))

    def training(self):
        if self.folder_train is None:
            self.showDialog("ERROR", "B???n c???n ch???n folder train")
            return
        if self.imgGenarate:
            path_folder = self.folder_train + "/gen"
        else:
            path_folder = self.folder_train
        if self.configs is None:
            self.showDialog("ERROR", "B???n c???n ch??a th??m file config")
            return
        feature = str(self.featureTypeCom.currentText())
        if not self.readFileConfig():
            return
        if feature == "LBP":
            try:
                numPoints = self.configs["LBP"]["numPoints"]
                radius = self.configs["LBP"]["radius"]
                cell_count = self.configs["LBP"]["cell_count"]
                patchSize = self.configs["LBP"]["patchSize"]
                self.featureExtractor = LocalBinaryPatterns(numPoints, radius, cell_count, patchSize)
            except:
                self.showDialog("ERROR", "C?? l???i x???y ra khi ?????c th??ng tin LBP")
                return
        elif feature == "HOG":
            try:
                winSize = self.configs["HOG"]["winSize"]
                blockSize = self.configs["HOG"]["blockSize"]
                blockStride = self.configs["HOG"]["blockStride"]
                cellSize = self.configs["HOG"]["cellSize"]
                nbins = self.configs["HOG"]["nbins"]
                self.featureExtractor = HOG(winSize = (winSize, winSize), blockSize = (blockSize, blockSize), blockStride=(blockStride, blockStride), cellSize=(cellSize,cellSize), nbins=nbins)
            except Exception as e:
                self.showDialog("ERROR", "C?? l???i x???y ra khi ?????c th??ng tin HOG")
                return
        else:
            try:
                radius = self.configs["CHOG"]["radius"]
                pixelDistance = self.configs["CHOG"]["pixelDistance"]
                blockCount = self.configs["CHOG"]["blockCount"]
                binCount = self.configs["CHOG"]["binCount"]
                self.featureExtractor = CHOG(radius, pixel_distance=pixelDistance, block_count=blockCount, bin_count=binCount)
            except:
                self.showDialog("ERROR", "C?? l???i x???y ra khi ?????c th??ng tin CHOG")
                return

        
        threading.Thread(target=self.runTrain, args=(path_folder, self.featureExtractor)).start()

    def runTrain(self, path_folder, extractor):
        self.train.setText("Training..")
        blur_size = self.configs["blur_size"]
        self.center_points, self.true_features = extract_samples(path_folder, extractor=extractor, valid_img_exts = [".tif",".jpg",".bmp",".png"], blur_size = blur_size)
        self.train.setText("Train")
    
    def predicting(self):
        if self.folder_predict is None:
            self.showDialog("ERROR", "B???n c???n ch???n folder predict")
            return
        if self.img_select is None:
            self.showDialog("ERROR", "B???n c???n ch???n file trong list train")
            return   

        if self.center_points is None or self.true_features is None:
            self.showDialog("ERROR", "B???n ch??a train d??? li???u")
            return
        if self.center_points is None or self.true_features is None:
            self.showDialog("ERROR", "B???n c???n train d??? li???u tr?????c")
            return
        if not self.getValue():
            return
        else:
            smp_img = cv.imread(self.folder_train + "/" + self.img_select)
            blur_size = self.configs["blur_size"]
            smp_img = cv.medianBlur(smp_img,blur_size)   
            if self.topValue is not None and self.bottomValue is not None and self.leftValue is not None and self.rightValue is not None:
                mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
                mask = cv.rectangle(mask, (self.leftValue, self.topValue),(smp_img.shape[1]-self.rightValue, smp_img.shape[0]-self.bottomValue), 255, -1)
                #mask = (self.topValue, self.leftValue, self.bottomValue, self.rightValue)
            else:
                mask = None

        candidate_method = str(self.candidateCom.currentText())        
        
        
        threading.Thread(target=self.runPredict, args=(smp_img, self.folder_predict, self.center_points, self.true_features,  self.windowSizeValue, \
                    candidate_method, self.numRandomValue, self.featureExtractor, mask, [".tif",".jpg",".bmp",".png"], blur_size, (5,5))).start()

    def runPredict(self, smp_img, folder,  center_points, true_features, window_size, candidate_method, num_random, extractor, mask_roi, valid_img_exts, blur_size, open_kernel_size):        
        start_time = time.time()
        self.predict.setText("Predicting..")
        count = 0
        for filename in os.listdir(folder):
            ext = "." + filename.split(".")[-1]
            if ext in valid_img_exts:
                path_file = folder + "/" + filename
                print(path_file)
                predict(smp_img, path_file, center_points, true_features, window_size, candidate_method, num_random, extractor, mask_roi, valid_img_exts, blur_size, open_kernel_size, debug = False )
            count +=1
            self.predict.setText("Predicting {}/{}".format(count, len(os.listdir(folder))))    
        self.predict.setText("Predict")
        print ("Predicting time: %.2f seconds" % (time.time() - start_time))

class Window(QtWidgets.QMainWindow):
    resized = QtCore.pyqtSignal()
    def  __init__(self, parent=None):
        super(Window, self).__init__(parent=parent)
        self.ui = Ui_iMorph()
        self.ui.setupUi(self)
        self.resized.connect(self.someFunction)

    def resizeEvent(self, event):
        self.resized.emit()
        return super(Window, self).resizeEvent(event)

    def someFunction(self):
        self.ui.resize(self.width(), self.height())

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    w = Window()
    w.show()
    sys.exit(app.exec_())

