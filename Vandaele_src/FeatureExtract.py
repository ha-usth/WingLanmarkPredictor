# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 12:23:26 2018

To run xfeatures2d.SURF_create:
    pip3 install opencv-python==3.4.2.17
    pip3 install opencv-contrib-python==3.4.2.17

"""

import cv2
import numpy as np
from skimage.transform import integral_image
from point import Point
#from skimage.feature import haar_like_feature

# function for rescale original images in different scales
def RescaleImage(imageFile, NoOfScale):
    #read input file
    img = cv2.imread(imageFile,cv2.IMREAD_GRAYSCALE)
    height, width = img.shape[:2]
    #print("size ", height," ", width)
    
    #rescale image by NoOfScale 
    listImgs = []
    
    for i in range(NoOfScale):
        img1 = cv2.resize(img, None, fx = 1/(2**i), fy = 1/(2**i))
        listImgs.append(img1)
        
    # return the list of images
    return listImgs

# xpos, ypos: horizontal and vertical cordinates of a point
def RescalePoint(xpos,ypos,NoOfScale):
    listPoints = []
    for i in range(NoOfScale):
        x = round(xpos/(2**i))
        y = round(ypos/(2**i))
        listPoints.append([x,y])
    return listPoints

# compute the RAW features
def computeRAW(listImgs, listPoints, W):
    RawFeature = []
    for scale in range(len(listImgs)):
        for j in range(-W,W+1):
            for k in range(-W,W+1):
                x = listPoints[scale][0]
                y = listPoints[scale][1]               
                #print("j ",j, " y+j,x+k", y+j,x+k, "image size ",listImgs[scale].shape, "\n",)
                #print(listImgs[scale].shape)
                RawFeature.append(listImgs[scale][y+j,x+k])
    return RawFeature

# def compute_raw_multiscale(imgsAtScales, p, no_scale, W):
#     feaVec = []
#     for scale in range(0, no_scale):
#         x = p.x/
#         for i in range(-W,W+1):
#             for j in range(-W,W+1):
#                 feaVec.append(imgsAtScales[scale][pScaled.y+j,pScaled.x+i])
#     return feaVec

# compute the SUB features
def computeSignedSUB(listImgs, listPoints, W):
    SUBFeature = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1]
        for j in range(-W,W+1):
            for k in range(-W,W+1):                            
                SUBFeature.append(listImgs[i][y+j,x+k].astype(np.int16)-listImgs[i][y,x].astype(np.int16))
    return SUBFeature

# compute the SUB features
def computeUnsignedSUB(listImgs, listPoints, W):
    SUBFeature = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1] 
        for j in range(-W,W+1):
            for k in range(-W,W+1):                           
                SUBFeature.append(np.abs(listImgs[i][y+j,x+k].astype(np.int16)-listImgs[i][y,x].astype(np.int16)))
    return SUBFeature

# compute the SURF features
def computeSURF(listImgs, listPoints, diameter_size = 5):
    featureVector = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1]
        extractor = cv2.xfeatures2d.SURF_create(extended=1)
        keypoints = [cv2.KeyPoint(x, y, _size = diameter_size, _class_id=0)]
        kps, fts = extractor.compute(listImgs[i], keypoints)                        
        #list = fts.tolist()[0]
        featureVector = featureVector + fts.tolist()[0]
    return featureVector

def computeSurfNonScale(image, point):
    extractor = cv2.xfeatures2d.SURF_create(extended=1)
    keypoints = [cv2.KeyPoint(point.x, point.y, 5, _class_id=0)]    
    kps, fts = extractor.compute(image, keypoints)              
    list = fts.tolist()[0]
    return list
# compute the GAUSSIAN SUB descriptors
def computeGaussianSUB(listImgs, listPoints, gausSigma, Ng):
    GAUSSSUBFeature = []
    for i in range(len(listImgs)):
        xOff = np.random.normal(0,gausSigma,Ng)
        xOff = np.int16(xOff)
        yOff = np.random.normal(0,gausSigma,Ng)
        yOff = np.int16(yOff)
        x = listPoints[i][0]
        y = listPoints[i][1]
        for p in range(Ng):            
            GAUSSSUBFeature.append(listImgs[i][y+yOff[p],x+xOff[p]].astype(np.int16)-listImgs[i][y,x].astype(np.int16))
    return GAUSSSUBFeature

# compute the GAUSSIAN SUB descriptors
def computeUnsignedGaussianSUB(listImgs, listPoints, gausSigma, Ng):
    GAUSSSUBFeature = []
    for i in range(len(listImgs)):
        xOff = np.random.normal(0,gausSigma,Ng)
        xOff = np.int16(xOff)
        yOff = np.random.normal(0,gausSigma,Ng)
        yOff = np.int16(yOff)
        x = listPoints[i][0]
        y = listPoints[i][1]
        for p in range(Ng):            
            GAUSSSUBFeature.append(np.abs(listImgs[i][y+yOff[p],x+xOff[p]].astype(np.int16)-listImgs[i][y,x].astype(np.int16)))            
    return GAUSSSUBFeature

# compute AKAZE
# diameter_size size of keypoint region
def computeAkaze(listImgs, listPoints, diameter_size = 5):
    featureVector = []
    for i in range(len(listImgs)):
        x = listPoints[i][0]
        y = listPoints[i][1]
        extractor = cv2.AKAZE_create()
        keypoints = [cv2.KeyPoint(x, y, _size = diameter_size, _class_id=0)]
        kps, fts = extractor.compute(listImgs[i], keypoints)                        
        list = fts.tolist()[0]
        featureVector = featureVector + list
    return featureVector

def computeAkazeNonScale(image, point, diameter_size = 10):
    extractor = cv2.AKAZE_create()
    keypoints = [cv2.KeyPoint(point.x, point.y, _size = diameter_size, _class_id=0)]    
    kps, fts = extractor.compute(image, keypoints)            
    list = fts.tolist()[0]
    return list



# def computeHogNonScale(image, point):
#     extractor = cv2.xfeatures2d.SURF_create(extended=1)
#     keypoints = [cv2.KeyPoint(point.x, point.y, 5, _class_id=0)]
#     kps, fts = extractor.compute(image, keypoints)              
#     list = fts.tolist()[0]
#     return list

def computeBriskNonScale(image, point):
    extractor = cv2.BRISK_create()
    keypoints = [cv2.KeyPoint(point.x, point.y, _size = 0, _class_id=0)]
    kps, fts = extractor.compute(image, keypoints)              
    list = fts.tolist()[0]
    return list


# compute the HAAR-LIKE descriptors
# Nh HAAR-LIKE features of random size and position are extracted inside each of the D windows, leading to Nh x D features
# the 4-type is used
def computeHAARLIKE(listImgs, listPoints, W, Nh, maxSizeHAAR):
    HAARFeature = []
    #print("len ",len(listImgs),"\n Points")
    # print(listPoints)
    # print("\n")
    for i in range(len(listImgs)):
        # compute the integral image
        ii = integral_image(listImgs[i])
        #print("Scale ",i, "size iiiiiiiiiiiiiiiiii", ii.shape," point ", listPoints[i][0],"\n")
        for j in range(Nh):
            # generate the position of the random
            x = np.random.randint(-W,W+1) + listPoints[i][0]
            y = np.random.randint(-W,W+1) + listPoints[i][1]
            size = np.random.randint(0,maxSizeHAAR)
            #print("Size ", size, "x ", x, "y ", y)
            # calculate the 4-type 
            
            HAARFeature.append(2*ii[y,x+size]+2*ii[y-size,x] + 2*ii[y+size,x]+2*ii[y,x-size] - ii[y-size,x+size] - 4*ii[y,x] - ii[y+size,x-size] - ii[y+size,x+size]-ii[y-size,x-size])                
    return HAARFeature
    

def computeHog(img_gray, point, cell_size=8, block_size=16, bins=9):
    winSize = (64,64)
    blockSize = (block_size, block_size)
    blockStride = (8,8)
    cellSize = (cell_size, cell_size)
    nbins = bins
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    #compute(img[, winStride[, padding[, locations]]]) -> descriptors
    winStride = (8,8)
    padding = (8,8)
    locations = ((point.x,point.y),)
    hist = hog.compute(img_gray,winStride,padding,locations).flatten()
    return hist


# Dau vao diem: x: chieu ngang, y: chieu doc
#inputF = "Z:\My Drive\Research\iMorphSharedByHai\Datasets\LabeledData\LeftWings\mam_F_R_oly_2X_81.tif"
# inputF = "Z:\My Drive\Research\iMorphSharedByHai\Datasets\Droso_nature_paper/001.bmp"
# D = 5 # number of scale
# W = 8 # window size is 2W+1

# img = cv2.imread(inputF)
# # #int_img = integral_image(img)
# cv2.imshow("title",img)
# cv2.waitKey(0)
# listImgs = RescaleImage(inputF,D)
# for img in listImgs:
#     h,w = img.shape[:2]
#     print("size ", h, " : ", +w)

# listPoints = RescalePoint(504,343,D)
# listPoints = RescalePoint(689,504,D)
# for i in range(len(listPoints)):
#     print("Point count at scale ",i," ",listPoints[i])
#raw = computeRAW(listImgs, listPoints, W)
#uSub = computeUnsignedSUB(listImgs, listPoints, W)
#print("size of uSub vector",len(uSub))
#sSub = computeSignedSUB(listImgs, listPoints, W)
#surf = computeSURF(listImgs, listPoints, W)

#gausSigma, Ng = 1, 100
# print(np.random.normal(0,gausSigma,Ng))
#gSub = computeGaussianSUB(listImgs, listPoints, gausSigma, Ng)
#print(gSub)
#haar = computeHAARLIKE(listImgs,listPoints, W, Nh = 8, maxSizeHAAR =5)

# hog = computeHog(listImgs[0], Point(504, 343))
# print(hog)
#print(raw)
#print(uSub)

#print(sSub)
#print(surf)
# print(len(raw))
#print(len(haar))

# akaze = computeAkazeNonScale(img, Point(100,100))
# print(akaze)