from __future__ import print_function
from numpy.core.shape_base import block
import cv2 as cv
import numpy as np
from math import cos, sin, sqrt
from numpy.core.fromnumeric import shape
from skimage.feature import local_binary_pattern # # pip install scikit-image
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from skimage.transform import integral_image

def plot_hisvec(vector, title = "", path = ""):
    x = list(range(1, len(vector)+1))
    x = np.asarray(x)
    plt.title(title)
    plt.bar(x, vector)
    if path == "":
        plt.show()
    else:
        plt.savefig(path)

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius, cell_count = 8, patchSize = 50):
        self.numPoints = numPoints
        self.radius = radius
        self.cell_count = cell_count
        self.patchSize = patchSize
    def set_image(self, img):
        self.img = img
    def __describe(self, image, eps=1e-7):#compute LBP histogram for the input image
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
    def compute_a_point(self, p):
        crop = self.img[p.y-self.patchSize:p.y+self.patchSize, p.x-self.patchSize:p.x+self.patchSize ]
        desc = self.__describe(crop)
        return desc.ravel().astype('float32')

class HOG:
    def __init__(self, winSize, blockSize, blockStride, cellSize, nbins = 9):     
        self.winStride = (4,4)
        self.padding = (2,2)
        self.hog = cv.HOGDescriptor((winSize,winSize),(blockSize, blockSize),(blockStride, blockStride),(cellSize, cellSize), nbins)
    def set_image(self, img):
        self.img = img
    def compute_a_point(self, p):        
        locations = ((p.x,p.y),)
        hist = self.hog.compute(self.img, self.winStride, self.padding,locations)
        hist = hist/(hist.sum()+1e-9)
        return hist.ravel().astype('float32')  
                      
class CHOG:
    def __init__(self, radius = 30, norm_scale_dis = 10, block_count = 8, bin_count = 18, pixel_distance = 1):        
        self.radius = radius        
        self.norm_scale_dis = norm_scale_dis
        self.block_count = block_count
        self.bin_count = bin_count

        self.full_magnitude = None
        self.full_orientation = None
        self.pixel_distance = pixel_distance

        r2 = self.radius**2
        size = int(2*self.radius/self.pixel_distance + 1)
        distances = [ [np.Infinity]*size for i in range(size)]
        for i in range(-self.radius, self.radius  + 1, self.pixel_distance):
            for j in range(-self.radius, self.radius  + 1, self.pixel_distance):
                sq = float(i**2+ j**2)
                if(sq<=r2):                    
                    distances[int((i+self.radius)/self.pixel_distance)][int((j+self.radius)/self.pixel_distance)] = sqrt(sq)
                    #print("{},{} is set to {}".format(i+self.radius, j+self.radius, sqrt(sq)))
        distances = np.asarray(distances)
        self.weight = norm.pdf(distances, 0, scale = self.norm_scale_dis)

        self.point_angles = np.asarray([[np.arctan2(j,i) for j in range(-self.radius, self.radius+1, self.pixel_distance)] for i in range(-self.radius, self.radius+1, self.pixel_distance)])
        self.point_angles = np.where(self.point_angles < 0, self.point_angles + 2*np.pi, self.point_angles)
        self.point_angles = np.where(self.point_angles > 2*np.pi, self.point_angles - 2*np.pi, self.point_angles) 
    def set_image(self, grey_img, debug = False, gradient_vc_dist = 3):
        xkernel = np.array([[-1, 0, 1]])
        ykernel = np.array([[-1], [0], [1]])
        #grey_img_blur = cv.medianBlur(grey_img, self.blur_size)
        grey_img_blur = grey_img.copy()
        dx = cv.filter2D(grey_img_blur, cv.CV_32F, xkernel)        
        dy = cv.filter2D(grey_img_blur, cv.CV_32F, ykernel)
        self.full_magnitude = np.sqrt(np.square(dx) + np.square(dy) + +1e-9)
        self.full_magnitude = cv.copyMakeBorder(self.full_magnitude, 0, 0, self.radius, self.radius, borderType= cv.BORDER_CONSTANT, value=0) #make a padding
        self.full_orientation = np.arctan2(dy, dx+0.0000001) # radian        
        self.full_orientation = cv.copyMakeBorder(self.full_orientation, 0, 0, self.radius, self.radius, borderType= cv.BORDER_CONSTANT, value=0) #make a padding
  
        if(debug):
            clImg = cv.cvtColor(grey_img, cv.COLOR_GRAY2BGR)
            for i in range (0, grey_img.shape[0], gradient_vc_dist):
                for j in range (0, grey_img.shape[1], gradient_vc_dist):
                    if(self.full_magnitude[i][j] > 0):
                        x = j+ self.full_magnitude[i][j] *  np.cos(self.full_orientation[i][j])
                        y = i+ self.full_magnitude[i][j] *  np.sin(self.full_orientation[i][j])
                        cv.arrowedLine(clImg, (j,i), (int(x), int(y)), (255,0,0), 1)
            cv.imshow("Gradient vector", clImg)
            cv.waitKey()
    def compute_a_point(self, p):   
        block_width = (2*np.pi)/self.block_count   #how many radian each bin is        
        vector_of_blocks = []
        for i in range(self.block_count):
            vector_of_blocks.append([])        
                
        magnitudes = np.asarray([[self.full_magnitude[p.y + i, p.x + j]  for j in range(-self.radius, self.radius+1, self.pixel_distance)] for i in range(-self.radius, self.radius+1, self.pixel_distance)]) 
        orientations = np.asarray([[self.full_orientation[p.y + i, p.x + j]  for j in range(-self.radius, self.radius+1, self.pixel_distance)] for i in range(-self.radius, self.radius+1, self.pixel_distance)]) 
        
        #weight_magnitudes = magnitudes * self.weight           
        weight_magnitudes = magnitudes
     
        #point_angles = point_angles*180/np.pi

        #orientation_modifieds = orientations - self.point_angles# this is data to compute histogram    
        orientation_modifieds = orientations
        orientation_modifieds = np.where(orientation_modifieds < 0, orientation_modifieds +  2*np.pi, orientation_modifieds)
        orientation_modifieds = np.where(orientation_modifieds >  2*np.pi, orientation_modifieds -  2*np.pi, orientation_modifieds)

        if(self.block_count == 1):
            #hist, _ = np.histogram(orientation_modifieds, bins= self.bin_count, range = [0, 2*np.pi], weights=weight_magnitudes)
            hist, bin = np.histogram(orientation_modifieds, bins= self.bin_count, range = [0, 2*np.pi], weights=weight_magnitudes, density = True)
            #hist /= (hist.sum()+1e-7) 
            return hist.ravel().astype('float32')
        elif(self.block_count > 1):
            base_block_indexes = np.floor(self.point_angles/block_width).astype(int)
            remainder_angles = self.point_angles - base_block_indexes*block_width
            base_block_ratios = (block_width-remainder_angles)/block_width
            sibl_block_indexes = (base_block_indexes + 1)%self.block_count

            for i in range(len(magnitudes)):
                for j in range(len(magnitudes[i])): 
                    if magnitudes[i][j] > 1e-8:                        
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
                    hist, _ = np.histogram(phase, bins= self.bin_count, range = [0, 2*np.pi], weights=magnitude, density = True)
                overall_hist.append(hist) # = np.concatenate([overall_hist, hist])            
            overall_hist = np.hstack(overall_hist)
            overall_hist = overall_hist.astype("float")
            overall_hist /= (overall_hist.sum()+1e-9)         
            return overall_hist.ravel().astype('float32')

class Haarlike:
    def __init__(self, W, Nh, D, maxSizeHAAR):
        self.W = W
        self.Nh = Nh
        self.maxSizeHAAR = maxSizeHAAR
        self.D = D #Number of scale
    def set_image(self, img):
        self.list_iis = []    
        for i in range(self.D):
            img1 = cv.resize(img, None, fx = 1/(2**i), fy = 1/(2**i))
            ii = integral_image(img1)
            self.list_iis.append(ii)         
        
    def __make_in_range(self, number, min, max):
        if(number <min):
            number = min
        elif(number>max):
            number = max
        return number
    def compute_a_point(self, p):  
        listPoints = []
        for i in range(self.D):
            x = round(p.x/(2**i))
            y = round(p.y/(2**i))
            listPoints.append([x,y])        

        HAARFeature = []
        for i in range(len(self.list_iis)):
            ii = self.list_iis[i]
            for j in range(self.Nh):
                # generate the position of the random
                x = np.random.randint(-self.W,self.W+1) + listPoints[i][0]
                y = np.random.randint(-self.W,self.W+1) + listPoints[i][1]                
                size = np.random.randint(0,self.maxSizeHAAR)      

                x= self.__make_in_range(x, size, ii.shape[1]-1 -size)
                y= self.__make_in_range(y, size, ii.shape[0]-1 -size)          
                HAARFeature.append(2*ii[y,x+size]+2*ii[y-size,x] + 2*ii[y+size,x]+2*ii[y,x-size] - ii[y-size,x+size] - 4*ii[y,x] - ii[y+size,x-size] - ii[y+size,x+size]-ii[y-size,x-size])                
        return np.array(HAARFeature)

class Surf:
    def __init__(self, D, diameter_size):
        self.D = D #number of scale
        self.diameter_size = diameter_size    
        
    def set_image(self, img):
        self.imgs = []   
        for i in range(self.D):
            img1 = cv.resize(img, None, fx = 1/(2**i), fy = 1/(2**i))            
            self.imgs.append(img1)         

    
    def compute_a_point(self, p):  
        listPoints = []
        self.extractor = cv.xfeatures2d.SURF_create(extended=1)  
        feature = []
        for i in range(self.D):
            x = round(p.x/(2**i))
            y = round(p.y/(2**i))
            keypoints = [cv.KeyPoint(x, y, _size = self.diameter_size, _class_id=0)]    
            kpt, fts = self.extractor.compute(self.imgs[0], keypoints)              
            feature.append(fts)
        feature = np.hstack(feature)
        feature = feature.astype("float")        
        return feature.ravel().astype('float32')