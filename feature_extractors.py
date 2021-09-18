#Extract Centric HOG
from __future__ import print_function

from numpy.core.shape_base import block
# from tables.tests.common import test_filename
import cv2 as cv
import numpy as np
import argparse
from math import cos, sqrt
# import matplotlib.pyplot as plt
from numpy.core.fromnumeric import shape
from skimage.feature import local_binary_pattern # # pip install scikit-image

import numpy as np
# import matplotlib.pyplot as plt
from scipy.stats import norm
# from scipy.spatial import distance
import time
import preprocess
import glob
import os

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius, cell_count = 8, patchSize = 50):
        self.numPoints = numPoints
        self.radius = radius
        self.cell_count = cell_count
        self.patchSize = patchSize
 
    def describe(self, image, eps=1e-7):#compute LBP histogram for the input image
        lbp = local_binary_pattern(image, self.numPoints, self.radius, method="uniform").astype("uint8")        
        cell_size = int(lbp.shape[0]/self.cell_count)
        overall_hist = []
        for i in range(self.cell_count):
            for j in range(self.cell_count):
                cell = lbp[cell_size*i:cell_size*(i+1), cell_size*j:cell_size*(j+1)]
                hist = cv.calcHist(cell, [0], None, [self.numPoints], [0,self.numPoints],).flatten()
                overall_hist = np.concatenate([overall_hist, hist])                
        overall_hist = overall_hist.astype("float")
        overall_hist /= (overall_hist.sum()+eps) 
        return overall_hist
    def compute_a_point(self, image, p):
        crop = image[p.y-self.patchSize:p.y+self.patchSize, p.x-self.patchSize:p.x+self.patchSize ]
        desc = self.describe(crop)
        return desc.ravel().astype('float32')
        
        # overall_hist_2 = []
        # for i in range(int(cell_count/2)):
        #     for j in range(int(cell_count/2)):
        #         cell = lbp[cell_size*i:cell_size*(i+1), cell_size*j:cell_size*(j+1)]
        #         hist = cv.calcHist(cell, [0], None, [self.numPoints], [0,self.numPoints],).flatten()
        #         overall_hist_2 = np.concatenate([overall_hist_2, hist])                
        # overall_hist_2 = overall_hist_2.astype("float")
        # overall_hist_2 /= (overall_hist_2.sum()+eps) 

        # overall_hist = np.concatenate([overall_hist, overall_hist_2])  
        #hist = np.histogram(lbp.ravel(), bins=range(0,self.numPoints+3), range=(0,self.numPoints+2))[0]
        
        #print(hist)    
        # hist = cv.calcHist([lbp], [0], None, [self.numPoints+2], [0,self.numPoints+2],).flatten()
        # hist = hist.astype("float")
        # hist /= (hist.sum()+eps) 
        # overall_hist = np.concatenate([overall_hist, hist])
class Hog:
    def __init__(self, winSize,blockSize,blockStride,cellSize,nbins = 9):
        self.winSize = winSize
        self.blockSize = blockSize
        self.blockStride = blockStride
        self.cellSize = cellSize
        self.nbins = nbins        
        self.winStride = (4,4)
        self.padding = (2,2)
        self.hog = cv.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins)
        
    def compute_a_point(self, img, p):        
        locations = ((p.x,p.y),)
        hist = self.hog.compute(img, self.winStride, self.padding,locations)
        return hist.ravel().astype('float32')  
                      
class CHOG:
    def __init__(self, radius = 30, norm_scale_dis = 15, blur_size = 9, block_count = 8, bin_count = 18, pixel_distance = 1):        
        self.radius = radius        
        self.norm_scale_dis = norm_scale_dis
        self.block_count = block_count
        self.bin_count = bin_count

        self.full_magnitude = None
        self.full_orientation = None
        self.blur_size = blur_size
        self.pixel_distance = pixel_distance
    def set_image(self, grey_img):
        xkernel = np.array([[-1, 0, 1]])
        ykernel = np.array([[-1], [0], [1]])
        #grey_img_blur = cv.medianBlur(grey_img, self.blur_size)
        grey_img_blur = grey_img.copy()
        dx = cv.filter2D(grey_img_blur, cv.CV_32F, xkernel)        
        dy = cv.filter2D(grey_img_blur, cv.CV_32F, ykernel)
        self.full_magnitude = np.sqrt(np.square(dx) + np.square(dy))
        self.full_orientation = np.arctan(np.divide(dy, dx+0.0000001)) # radian

    def compute_a_point(self, p):   
        block_width = (2*np.pi)/self.block_count   #how many radian each bin is
        
        vector_of_blocks = [[]] * self.block_count 

        magnitudes = np.asarray([[self.full_magnitude[p.y + i, p.x + j]  for j in range(-self.radius, self.radius, self.pixel_distance)] for i in range(-self.radius, self.radius, self.pixel_distance)]) 
        orientations = np.asarray([[self.full_orientation[p.y + i, p.x + j]  for j in range(-self.radius, self.radius, self.pixel_distance)] for i in range(-self.radius, self.radius, self.pixel_distance)]) 

        distances = np.asarray([[sqrt(i**2 + j**2) for j in range(-self.radius, self.radius, self.pixel_distance)] for i in range(-self.radius, self.radius, self.pixel_distance)])
        weight = norm.pdf(distances, 0, scale = self.norm_scale_dis)
        weight_magnitudes = magnitudes * weight   

        point_angles = np.asarray([[np.arctan2(i,j) for j in range(-self.radius, self.radius, self.pixel_distance)] for i in range(-self.radius, self.radius, self.pixel_distance)])

        point_angles = np.where(point_angles < 0, point_angles + 2*np.pi, point_angles)
        point_angles = np.where(point_angles > 2*np.pi, point_angles - 2*np.pi, point_angles)        
        
        orientation_modifieds = orientations + point_angles # this is data to compute histogram    
        #orientation_modifieds = orientations
        orientation_modifieds = np.where(orientation_modifieds < 0, orientation_modifieds +  2*np.pi, orientation_modifieds)
        orientation_modifieds = np.where(orientation_modifieds >  2*np.pi, orientation_modifieds -  2*np.pi, orientation_modifieds)

        if(self.bin_count == 1):
            hist, _ = np.histogram(orientation_modifieds, bins= self.bin_count*2, range = [0, 2*np.pi], weights=weight_magnitudes)
            hist /= (hist.sum()+1e-7) 
            return hist.ravel().astype('float32')
        elif(self.bin_count > 1):
            base_block_indexes = np.floor(point_angles/block_width).astype(int)
            remainder_angles = point_angles - base_block_indexes*block_width
            base_block_ratios = (block_width-remainder_angles)/block_width
            sibl_block_indexes = (base_block_indexes + 1)%self.block_count

            for i in range(len(magnitudes)):
                for j in range(len(magnitudes[i])): 
                    if magnitudes[i][j] < 1e-8:
                        continue
                    vector_of_blocks[base_block_indexes[i][j]].append([orientation_modifieds[i][j], weight_magnitudes[i][j]*base_block_ratios[i][j]])                        
                    vector_of_blocks[sibl_block_indexes[i][j]].append([orientation_modifieds[i][j], weight_magnitudes[i][j]*(1-base_block_ratios[i][j])])

            overall_hist = []
            for block_index in range(len(vector_of_blocks)):#compute weighted histogram for each block, then concatenate them
                ar = np.asarray(vector_of_blocks[block_index])
                if(len(ar) == 0):
                    hist  = [0] * self.bin_count
                else:
                    phase = ar[:,0]
                    magnitude = ar[:,1]
                    hist, _ = np.histogram(phase, bins= self.bin_count, range = [0, 2*np.pi], weights=magnitude)
                    hist /= (hist.sum()+1e-7) 
                    #hist = cv.normalize(hist)
                overall_hist.append(hist) # = np.concatenate([overall_hist, hist])            
            overall_hist = np.hstack(overall_hist)
            overall_hist = overall_hist.astype("float")
            overall_hist /= (overall_hist.sum()+1e-7)         
            return overall_hist.ravel().astype('float32')


