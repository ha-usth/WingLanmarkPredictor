import cv2
import glob
import random
from sklearn.ensemble import IsolationForest
import joblib
from static import StaticVariable
import math
import time
import numpy
from scipy.spatial import distance
from PIL import Image
class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])
def feature_distance(feature1, feature2):
    a = numpy.array(feature1)
    b = numpy.array(feature2)
    dst = distance.euclidean(a, b)
    return(dst)
def get_final_point (detected_points):
    X=[]
    for point in detected_points:
        X.append([point.x,point.y])
    clf = IsolationForest()
    clf.fit(X)
    ret = clf.predict(X)
    x_average = 0
    y_average = 0
    num_point = 0
    for i in range (0,len(detected_points)):
        if ret[i]==1:
            x_average = x_average + detected_points[i].x
            y_average = y_average + detected_points[i].y
            num_point = num_point + 1
    if (num_point!=0):
        x_average = x_average/num_point
        y_average = y_average/num_point
    return (Point(x_average,y_average))

def computeAkazeNonScale(image, point, diameter_size = 10):
    extractor = cv2.AKAZE_create()
    keypoints = [cv2.KeyPoint(point.x, point.y, _size = diameter_size, _class_id=0)]    
    kps, fts = extractor.compute(image, keypoints)            
    list = fts.tolist()[0]
    return list
def computeSURF(listImgs, listPoints, diameter_size = 5):
    featureVector = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1]
        extractor = cv2.xfeatures2d.SURF_create(extended=1)
        keypoints = [cv2.KeyPoint(x, y, _size = diameter_size, _class_id=0)]
        kps, fts = extractor.compute(listImgs[i], keypoints)                        
        featureVector = featureVector + fts.tolist()[0]
    return featureVector

def computeSurfNonScale(image, point):
    extractor = cv2.xfeatures2d.SURF_create(extended=1)
    keypoints = [cv2.KeyPoint(point.x, point.y, 5, _class_id=0)]    
    kps, fts = extractor.compute(image, keypoints)              
    list = fts.tolist()[0]
    return list
def computeBriskNonScale(image, point):
    extractor = cv2.BRISK_create()
    keypoints = [cv2.KeyPoint(point.x, point.y, _size = 0, _class_id=0)]
    kps, fts = extractor.compute(image, keypoints)              
    list = fts.tolist()[0]
    return list

#------------Load the training data-----------------------------------
def get_true_feature(training_directory,feature_type,scale_ratio):
    print(training_directory, feature_type,scale_ratio)
    true_features = [] # all the feature vectors of all landmarks for all images
    list_file = training_directory + '/*.txt'
    len_prefix = len(training_directory)
    center_points = []
    num_file = 0
    for name in glob.glob(list_file):
        num_file = num_file + 1
        file_name = name[len_prefix+1:len(name)-4]
        img_file_bmp = training_directory + '/' + file_name + '.bmp'  # need to check for all the popular image format bmp, png, jpg, jpeg
        img_file_png = training_directory + '/' + file_name + '.png'
        img_file_jpg = training_directory + '/' + file_name + '.jpg'
        img_file_jpeg = training_directory + '/' + file_name + '.jpeg'
        img_file_tif = training_directory + '/' + file_name + '.tif'
        img = cv2.imread(img_file_bmp)
        if img is None:
            img = cv2.imread(img_file_png)
            if img is None:
                img =cv2.imread(img_file_jpg)
                if img is None:
                    img = cv2.imread(img_file_jpeg)
                    if img is None:
                        img = cv2.imread(img_file_tif)
        img = cv2.resize(img,None,fx=scale_ratio,fy=scale_ratio)
        true_feature = [] # feature vectors of all landmarks for one image
        StaticVariable.training_size += 1
        # print (file_name)
        try:
            with open(name) as f:
                lines = f.readlines()
                index = 0 
                for line in lines:
                    xy=line.split()
                    xy[0]=float(xy[0])*scale_ratio
                    xy[1]=float(xy[1])*scale_ratio
                    point_tmp=Point(float(xy[0]),float(xy[1])) 
                    if len (center_points) <= index:
                        center_points.append(Point(float(xy[0]),float(xy[1])))
                    else:
                        center_points[index].x = center_points[index].x + float(xy[0])
                        center_points[index].y = center_points[index].y + float(xy[1])
                    index = index + 1                            
                    if (feature_type == 'akaze'):
                        feature_tmp = computeAkazeNonScale(img,point_tmp)
                    elif (feature_type == 'brisk'):
                        feature_tmp = computeBriskNonScale(img,point_tmp)
                    elif (feature_type == 'surf'):
                        feature_tmp = computeSurfNonScale(img,point_tmp)
                    true_feature.append(feature_tmp)
            true_features.append(true_feature)
        except Exception as e:
            print(e)
            continue
    for point in center_points:
        point.x = int (point.x/num_file)
        point.y = int (point.y/num_file)
    # print (len(true_features))
    return (center_points,true_features)


def prediction_image(img, text_file, true_features,feature_type,method,center_points,window_size,num_random,kp_threshold,dense_step,scale_ratio):
    f = open(text_file, "w")
    img = cv2.resize(img,None,fx=scale_ratio,fy=scale_ratio)
    img_height = img.shape[0]
    img_width = img.shape[1]
    index = 0
    for point in center_points:
        print('index=',index)
        minx = max (0,point.x - window_size)
        maxx = min (point.x + window_size,img_width-1)
        miny = max (0, point.y -window_size)
        maxy = min (point.y + window_size,img_height-1)
        print('minx, max= ',minx,maxx,miny,maxy)
        candidate_points = []
        candidate_features = []
        detected_points = []
        if (method == 'keypoint'):
            img_roi=img[int(miny):int(maxy),int(minx):int(maxx)] 
            gray = cv2.cvtColor(img_roi,cv2.COLOR_BGR2GRAY)
            gray = cv2.blur(gray,(5,5))
            gray = numpy.float32(gray)
            dst = cv2.cornerHarris(gray,2,3,0.1)
            th = dst.max() * kp_threshold
            for i in range(len(dst)):
                for j in range(len(dst[i])):
                    val = dst[i][j]
                    if val > th:
                        point_tmp1 = Point (0,0)
                        point_tmp1.x= j + minx
                        point_tmp1.y = i + miny
                        try:
                            if (feature_type == 'akaze'):
                                feature_tmp = computeAkazeNonScale(img,point_tmp1)
                            elif (feature_type == 'brisk'):
                                feature_tmp = computeBriskNonScale(img,point_tmp1)
                            elif (feature_type == 'surf'):
                                feature_tmp = computeSurfNonScale(img,point_tmp1)
                            candidate_features.append(feature_tmp)
                            candidate_points.append(point_tmp1)
                        except:
                            continue

        if (method == 'random_point'):
            for i in range (0,num_random):
                
                x=random.randrange (int(minx),int(maxx),1)
                y=random.randrange (int(miny),int(maxy),1)
                point_tmp1=Point(x,y)
                try:
                    if (feature_type == 'akaze'):
                        feature_tmp = computeAkazeNonScale(img,point_tmp1)
                    elif (feature_type == 'brisk'):
                        feature_tmp = computeBriskNonScale(img,point_tmp1)
                    elif (feature_type == 'surf'):
                        feature_tmp = computeSurfNonScale(img,point_tmp1)
                    candidate_features.append(feature_tmp)
                    candidate_points.append(point_tmp1)
                except:
                    continue

        if (method == 'dense'):
            for i in range (minx,maxx,dense_step):
                for j in range (miny,maxy,dense_step):
                    point_tmp1=Point(i,j)
                    try:
                        if (feature_type == 'akaze'):
                            feature_tmp = computeAkazeNonScale(img,point_tmp1)
                        elif (feature_type == 'brisk'):
                            feature_tmp = computeBriskNonScale(img,point_tmp1)
                        elif (feature_type == 'surf'):
                            feature_tmp = computeSurfNonScale(img,point_tmp1)
                        # print('-----------',candidate_point.x,candidate_point.y,feature_tmp)
                        candidate_features.append(feature_tmp)
                        candidate_points.append(point_tmp1)
                    except:
                        continue

        for i in range (0,len(true_features)):# loop through all the training images
            true_feature_tmp = true_features[i][index]
            min_distance = 999999
            for j in range (0,len(candidate_features)):
                dis_tmp= feature_distance(candidate_features[j],true_feature_tmp)
                if (dis_tmp<min_distance):
                    min_distance=dis_tmp
                    min_index=j
            print('mindex= ',min_index)
            detected_points.append(candidate_points[min_index])
        result_point = get_final_point (detected_points)
        img = cv2.circle(img, (int(result_point.x),int(result_point.y)), 2, (255,0,0), 2)
        img = cv2.rectangle(img, (minx,miny), (maxx,maxy), (0,0,255), 2)
        result_point.x = int(result_point.x/scale_ratio)
        result_point.y = int(result_point.y/scale_ratio)
        string_point = str(result_point.x) + " "+ str(result_point.y) + "\n"
        f.write(string_point)
        index = index + 1
    f.close()

def prediction (prediction_directory,true_features,feature_type,method,center_points,window_size,num_random,kp_threshold,dense_step,scale_ratio):    
    list_file_bmp = prediction_directory + '/*.bmp'
    list_file_png = prediction_directory + '/*.png'
    list_file_jpg = prediction_directory + '/*.jpg'
    list_file_jpeg = prediction_directory + '/*.jpeg'
    list_file_tif = prediction_directory + '/*.tif'
        
    list_files = [list_file_bmp,list_file_png,list_file_jpg,list_file_jpeg,list_file_tif]
    window_size = int(window_size*scale_ratio)
    for list_file in list_files:
        for name in glob.glob(list_file):
            StaticVariable.predicting_size += 1
            len_prefix = len(prediction_directory)
            name_tmp = name[len_prefix+1:len(name)-4]
            print (name_tmp)
            text_file = prediction_directory + '/' + name_tmp + '.txt'
            img=cv2.imread(name)
            prediction_image(img, text_file, true_features,feature_type,method,center_points,window_size,num_random,kp_threshold,dense_step,scale_ratio)
        
        