from __future__ import print_function

import glob
import os
import time
from static import StaticVariable
import cv2 as cv
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import IsolationForest #pip install sklearn
# from skimage.feature import local_binary_pattern  # # pip install scikit-image
import random
from pathlib import Path
import math 

from point import Point
import preprocess
from feature_extractors import *

def extract_samples(sample_paths, extractor = CHOG(), valid_img_exts = [".tif",".jpg",".bmp",".png"], blur_size = 3):#extract features of samples images    
    true_features = [] # all the feature vectors of all landmarks for all images
    list_file = sample_paths + '/*.txt'
    len_prefix = len(sample_paths)
    center_points = []
    num_file = 0    
    
    for name in glob.glob(list_file):#for each txt file
        num_file = num_file + 1
        file_name = name[len_prefix+1:len(name)-4]
        StaticVariable.training_size += 1
        img_sample_ori = None #
        for img_file in glob.glob(sample_paths + '/' + file_name + '.*'):
            ext = os.path.splitext(img_file)[1]
            if ext.lower() in valid_img_exts:
                img_sample_ori = cv.imread(img_file, cv.IMREAD_GRAYSCALE)
                img_sample_ori = cv.medianBlur(img_sample_ori, blur_size)              
                break
            else:
                continue   
        if img_sample_ori is None:
            continue           

        extractor.set_image(img_sample_ori)       
        try:
            f = open(name)
            lines = f.readlines()
            index = 0 
            features_a_sample = []
            for line in lines:
                xy=line.split()
                xy[0]=float(xy[0])
                xy[1]=float(xy[1])
                p=Point(int(xy[0]),int(xy[1])) 
                if len (center_points) <= index:
                    center_points.append(p)
                else:
                    center_points[index].x = center_points[index].x + p.x
                    center_points[index].y = center_points[index].y + p.y         

                #extract feature        
                desc = extractor.compute_a_point(p)                
                features_a_sample.append(desc)
                index +=1
            true_features.append(features_a_sample)
        except cv.error:
            continue
    for point in center_points:
        point.x = int (point.x/num_file)
        point.y = int (point.y/num_file)
    return center_points, true_features       

# candidate_method = "keypoint" | "keypoint_on_bin_img" | "harris" | "random" | "gaussian"
def predict(smp_img, pre_img_ori_path, center_points, true_features, window_size = 60, candidate_method = "keypoint_on_bin_img", num_random = 300, \
            extractor = CHOG(), mask_roi = None,\
            valid_img_exts = [".tif",".jpg",".bmp",".png"], blur_size = 3, open_kernel_size = (5,5), lm_indexes = None, skip_outlier = False, debug = False):  
    if lm_indexes is None:
        lm_indexes= np.asarray(list(range(0, len(center_points))))
    #--------------1. PREPROCESS predicted images: align the pre_img so that the wing in predicted imag is almost fit to the sample one
    pre_img = cv.imread(pre_img_ori_path, cv.IMREAD_GRAYSCALE) 
    pre_img = cv.medianBlur(pre_img, blur_size)        

    pre_img_alig = pre_img.copy()
    affine = np.float32([[1,0,0],[0,1,0]])
    if(smp_img is not None):
        pre_img_alig, affine = preprocess.align(smp_img, pre_img, mask_roi, mask_roi, debug = debug)    
    affine_inv = cv.invertAffineTransform(affine)   
    
    #images just for visualizing when running debug
    result_img = cv.imread(pre_img_ori_path) 
    result_img_alig = cv.warpAffine(result_img, affine,  (result_img.shape[1],result_img.shape[0]))                 

    #--------------2. FIND all candidates
    all_candidates = []  
    if(candidate_method == "keypoint" or candidate_method == "keypoint_on_bin_img"):
        mask = np.zeros(pre_img_alig.shape[:2], dtype="uint8") 
        for lm_index in range(len(center_points)):
            if lm_index not in lm_indexes:
                continue        
            # for i in range (-window_size,window_size):
            #     for j in range (-window_size,window_size):
            #         p = center_points[lm_index] + Point(j,i)
            #         p.force_in_range(pre_img_alig.shape[:2][1]-1, pre_img_alig.shape[:2][0]-1)                    
            #         mask[p.y][p.x] = 255
            cv.circle(mask, center = (center_points[lm_index].x, center_points[lm_index].y), radius = window_size, color = (255,255,255), thickness=-1)

        img_to_find_keypoint = pre_img_alig
        if(candidate_method == "keypoint_on_bin_img"):
            img_to_find_keypoint = preprocess.remove_particles(pre_img_alig, open_kernel_size = open_kernel_size)                 
        keypoint_detector = cv.AKAZE_create()         
        kps = keypoint_detector.detect(img_to_find_keypoint, mask)    
        for kp in kps:                     
            p = Point(int(kp.pt[0]),int(kp.pt[1]))     
            if (candidate_method == "keypoint_on_bin_img" and img_to_find_keypoint[p.y][p.x] >0):#skip point is not on the skeleton
                continue   
            all_candidates.append(p)              



    elif(candidate_method == "harris" ):
        mask = np.zeros(pre_img_alig.shape[:2], dtype="uint8") 
        for lm_index in range(len(center_points)):      
            for i in range (-window_size,window_size):
                for j in range (-window_size,window_size):
                    p = center_points[lm_index] + Point(j,i)
                    p.force_in_range(pre_img_alig.shape[:2][1]-1, pre_img_alig.shape[:2][0]-1)
                    mask[p.y][p.x] = 255

        dst = cv.cornerHarris(pre_img_alig,2,3,0.1)
        th = dst.max() * 0.01
        for i in range(len(dst)):
            for j in range(len(dst[i])):
                val = dst[i][j]
                if val > th and mask[i][j] > 0:
                        # logger.debug("%s %s: %s", i, j, val)
                    # print('Point= ',j,i)
                    point_tmp1 = Point (j,i)                    
                    all_candidates.append(point_tmp1)  
    elif (candidate_method == "random"):
        for lm_index in range(len(center_points)):
            if lm_index not in lm_indexes:
                continue        
            center_roi = center_points[lm_index]
            for i in range (0,num_random):                        
                x=random.randrange (int(center_roi.x - window_size),int(center_roi.x + window_size),1)
                y=random.randrange (int(center_roi.y - window_size),int(center_roi.y + window_size),1)
                if(  (center_roi.x-x)**2 + (center_roi.y-y)**2 <= window_size**2 ):
                    all_candidates.append(Point(x,y))  
    elif (candidate_method == "gaussian"):
        for lm_index in range(len(center_points)):
            if lm_index not in lm_indexes:
                continue    
            center_roi = center_points[lm_index]
            random_pairs = np.random.normal(0, 20, (num_random,2))
            print(random_pairs)
            for pair in random_pairs:
                if(abs(pair[0]) < window_size and  abs(pair[1]) < window_size):
                    all_candidates.append(Point(center_roi.x + int(pair[0]), center_roi.y + int(pair[1])))  
    if(debug):#Visualize the candidates
        for id in lm_indexes:
            c = center_points[id]
            cv.circle(result_img_alig, (c.x, c.y), radius= window_size, color = (0,0,0), thickness=2)
        for p in all_candidates:
            cv.drawMarker(result_img_alig, (p.x, p.y), (0, 222,0), markerType= cv.MARKER_TILTED_CROSS, markerSize=15, thickness = 2)     
        cv.imshow("Candidates", result_img_alig)
        cv.waitKey()
    #--------------3. EXTRACT features for candidates    
    extractor.set_image(pre_img_alig)        
    candidates = [[] * len(center_points)]  #1st dim: lm_index, 2nd dim: coordinate of candidate
    features = [[] * len(center_points)]  #1st dim: lm_index, 2nd dim: features of candidates         
    for lm_index in range(len(center_points)):
        candidates.append([])
        features.append([])
    for p in all_candidates:    
        desc = extractor.compute_a_point(p)
        for lm_index in range(len(center_points)):#foreach landmark center, if the point is in range -> put in candidate array
            if (center_points[lm_index].x - window_size <= p.x and p.x <= center_points[lm_index].x + window_size and \
                center_points[lm_index].y - window_size <= p.y and p.y <= center_points[lm_index].y + window_size): 
                candidates[lm_index].append(p)
                features[lm_index].append(desc)     

    #--------------4. MATCH each landmark of each sample images with a candidate, then compute the center
    detected_lms_align = []
    for lm_index in range(len(center_points)):#foreach landmark index         
        if( lm_index not in lm_indexes):
            continue
        if(len(candidates[lm_index]) ==0):
            detected_lms_align.append(np.array([[center_points[lm_index].x, center_points[lm_index].y]])) 
            continue
        selecteds = []  # n selected correspond to n sample images
        distances = {}

        for sample_index in range(len(true_features)):#foreach sample features images
            true_feature = true_features[sample_index][lm_index]                       
            best_distance = 0#np.Infinity 
            most_like_p = center_points[lm_index]
            for i in range(len(candidates[lm_index])):#with this sample_index, search for the most likely point from all candidates of this lm_index                    
                fea = features[lm_index][i]                                     
                dist = cv.compareHist(true_feature, fea, cv.HISTCMP_INTERSECT)
                #dist = np.linalg.norm(true_feature - fea)
                distances[i] = [candidates[lm_index][i], dist]
                # if(i == 78):
                #plot_hisvec(fea, f"{candidates[lm_index][i]}", f"{candidates[lm_index][i]}.png")     
                if(dist > best_distance):
                    most_like_p = candidates[lm_index][i]
                    best_distance = dist   
                    #plot_hisvec(fea, f"{most_like_p}")         
            selecteds.append(most_like_p)
        
        if skip_outlier and len (selecteds)>2:
            X=[]
            for point in selecteds:
                X.append([point.x,point.y])
            clf = IsolationForest(n_estimators=100,max_samples='auto',contamination=float(0.2),random_state=np.random.RandomState(42))
            clf.fit(X)
            classified = clf.predict(X)
            x_average = 0
            y_average = 0
            num_point = 0
            for i in range (0,len(selecteds)):
                if classified[i]==1:
                    x_average = x_average + selecteds[i].x
                    y_average = y_average + selecteds[i].y
                    num_point = num_point + 1
                if debug:    
                    if classified[i]==1:
                        cv.drawMarker(result_img_alig, (int(selecteds[i].x),int(selecteds[i].y)),  color=(0, 255, 0), markerType = cv.MARKER_TRIANGLE_UP, markerSize = 5, thickness = 2)
                    else:
                        cv.drawMarker(result_img_alig, (int(selecteds[i].x),int(selecteds[i].y)),  color=(0, 0, 255), markerType = cv.MARKER_TRIANGLE_DOWN, markerSize = 5, thickness = 2)          
        else:
            x_average = 0
            y_average = 0
            num_point = 0
            for i in range (0,len(selecteds)):            
                x_average = x_average + selecteds[i].x
                y_average = y_average + selecteds[i].y
                num_point = num_point + 1
                if debug:    
                    cv.drawMarker(result_img_alig, (int(selecteds[i].x),int(selecteds[i].y)),  color=(0, 255, 0), markerType = cv.MARKER_DIAMOND, markerSize = 5, thickness = 2)  
        if (num_point!=0):
            x_average = x_average/num_point
            y_average = y_average/num_point 
        else:
            x_average = center_points[lm_indexes].x
            y_average =  center_points[lm_indexes].y                          
        detect_lm = np.array([[x_average, y_average]])                                
        detected_lms_align.append(detect_lm) 
        if debug: 
            result_img_alig = cv.drawMarker(result_img_alig, (int(x_average),int(y_average)),  color=(0, 0, 255), markerType = cv.MARKER_CROSS, markerSize = 15, thickness = 1)            
    if debug: 
        cv.imshow("result_img_alig",result_img_alig)
        cv.waitKey(0)

    #--------------5. Final process: transform the predicted landmark to the space of original images, then save to txt file
    if len(detected_lms_align) > 0: #there are detected landmarks 
        detected_lms_align = np.asarray(detected_lms_align)
        detected_lms_align = detected_lms_align.reshape((-1,1,2))
        detect_lm_tfback = cv.transform(detected_lms_align,affine_inv)

        text_file_out = pre_img_ori_path.rpartition('.')[0] + '.txt'
        f = open(text_file_out, "w")     
        for lm in detect_lm_tfback:
            x = int(lm[0][0])
            y = int(lm[0][1])
            string_point = str(x) + " "+ str(y) + "\n"
            f.write(string_point)
            if debug: 
                cv.drawMarker(result_img, (int(lm[0][0]),int(lm[0][1])),  color=(0, 255, 0), markerType = cv.MARKER_CROSS, markerSize = 15, thickness = 1) 
        f.close()

    if debug: 
        cv.imshow("result", result_img)    
        cv.waitKey()
    #--------------END OF FUNCTION---------------------------------------------------------------------

def predict_folder(smp_img, folder, center_points, true_features, window_size = 60, candidate_method = "keypoint_on_bin_img", num_random = 100, \
            extractor = CHOG(), mask_roi = None,\
            valid_img_exts = [".tif",".jpg",".bmp",".png"], blur_size = 3, open_kernel_size = (5,5), debug = False):
    # print(smp_img_path)
    # smp_img = cv.imread(smp_img_path)
    # smp_img = cv.medianBlur(smp_img,blur_size)
    # print(smp_img.shape)
    # print("window_size : ", window_size)
    for filename in os.listdir(folder):
        ext = "." + filename.split(".")[-1]
        if ext in valid_img_exts:
            path_file = folder + "/" + filename
            print(path_file)
            predict(smp_img, path_file, center_points, true_features, window_size, candidate_method, 
                    num_random, extractor, mask_roi, valid_img_exts, blur_size, open_kernel_size, debug )


