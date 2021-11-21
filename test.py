from os import makedirs
import cv2 as cv
import numpy as np
import glob
import time

from feature_extractors import *
from landmark_predictor import extract_samples, extract_samples_scale, predict, predict_scale
from preprocess import *
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

################################################# ds 2k, ext = LocalBinaryPatterns is the bé####################################
# OPEN_KERNEL_SIZE = (5,5)
# WINDOW_SIZE = 50
# LBP_PATCH_SIZE = 50  #for lbp

BLUR_SIZE = 3 #good for ds 2k 
# num_random = 100

# #ext = LocalBinaryPatterns(8,3, cell_count = 8, patchSize= LBP_PATCH_SIZE)
ext = CHOG(radius = 20, pixel_distance= 1, block_count = 1, bin_count = 9)

center_points, true_features, sim = extract_samples ('../KeypointMatching/training2k_align', extractor = ext, blur_size = BLUR_SIZE, compute_similarity= True )
print ("similarity {training2k}", sim)

# smp_img = cv.imread('../KeypointMatching/training2k/egfr_F_R_oly_2X_3.tif', cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
# smp_img = cv.medianBlur(smp_img,BLUR_SIZE)

# mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
# ROI_BORDER_SIZE = 250
# mask = cv.rectangle(mask, (ROI_BORDER_SIZE,ROI_BORDER_SIZE),(smp_img.shape[1]-ROI_BORDER_SIZE, smp_img.shape[0]-ROI_BORDER_SIZE), 255, -1)

# predict (smp_img, '../KeypointMatching//predict2k\egfr_F_R_oly_2X_7.tif', center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint_on_bin_img", extractor= ext, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= True)
# for name in glob.glob('./predict2k/*.tif'):#for each tif
#     print("Processing ", name)
#     predict(smp_img, name, center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint_on_bin_img", extractor= ext, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= False)

################################################# ds nature####################################
extr = CHOG(radius = 20, pixel_distance= 1, block_count = 8, bin_count = 18)
#extr = LocalBinaryPatterns(16, 3, cell_count = 8, patchSize= 30)
# extr = Haarlike(W = 8, Nh = 8, D = 5, maxSizeHAAR= 5)
# extr = Surf(D = 1, diameter_size= 5)
# # #extr = StackedLocalBinaryPatterns(8, 3)
#extr = HOG(winSize = 64, blockSize =16,blockStride = 4, cellSize = 8)

# BLUR_SIZE = 5
# OPEN_KERNEL_SIZE = (5,5)
# WINDOW_SIZE = 80

# start_time = time.time()
center_points, true_features, sim = extract_samples ('../KeypointMatching/training_nature_align_15', extractor = extr, blur_size = BLUR_SIZE, compute_similarity= True)
print ("similarity {nature}", sim)
# print ("Learning run time: %.2f seconds" % (time.time() - start_time))

# #smp_img = cv.imread('../KeypointMatching/training_nature_align_11/011.bmp', cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
# smp_img = cv.imread('../KeypointMatching/training_nature_fine/011.bmp', cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
# smp_img = cv.medianBlur(smp_img, BLUR_SIZE)
# mask = np.zeros(smp_img.shape[:2], dtype="uint8") 

# #ROI_BORDER_SIZE = 10
# #mask = cv.rectangle(mask, (ROI_BORDER_SIZE,ROI_BORDER_SIZE),(smp_img.shape[1]-ROI_BORDER_SIZE, smp_img.shape[0]-ROI_BORDER_SIZE), 255, -1)
# mask = cv.rectangle(mask, (0,90),(smp_img.shape[1], smp_img.shape[0]-90), 255, -1)

# start_time = time.time()
# predict (smp_img, '../KeypointMatching/predict_nature/008.bmp', center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint", extractor= extr, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= False)
# print ("Predct run time: %.2f seconds" % (time.time() - start_time))

# # start_time = time.time()
# # index = 1
# # for name in glob.glob('../KeypointMatching/predict_nature/*.bmp'):#for each tif
# #     start_time_an_img = time.time()
# #     print("Processing {} ".format(index), name)
# #     # if index > 10:
# #     #     break
# #     index +=1
    
# #     predict (smp_img, name, center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint", extractor= extr, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, skip_outlier= False, debug= False)
# #     print ("Predicting time: %.2f seconds" % (time.time() - start_time_an_img))
# # print ("Predicting time: %.2f seconds" % (time.time() - start_time))
####################################################################################



#extr = LocalBinaryPatterns(16, 3, cell_count = 8, patchSize= 30)
# extr = HOG(winSize = 64, blockSize =16,blockStride = 4, cellSize = 8)
extr = CHOG(radius = 40, pixel_distance= 1, block_count = 4, bin_count = 9)
# extr = Haarlike(W = 8, Nh = 8, D = 5, maxSizeHAAR= 5)
#extr = Surf(D = 1, diameter_size= 5)
#extr = StackedLocalBinaryPatterns(8, 3)

# BLUR_SIZE = 5
# OPEN_KERNEL_SIZE = (5,5)
# WINDOW_SIZE = 100

# start_time = time.time()
center_points, true_features, sim = extract_samples ('../KeypointMatching/training_Bactrocera', extractor = extr, blur_size = BLUR_SIZE, compute_similarity= True)
print ("similarity Bactrocera:", sim)
# print ("ex run time: %.2f seconds" % (time.time() - start_time))
#plot_hisvec(true_features[0][14])

#smp_img = cv.imread('../KeypointMatching/training_sandfly/F_ST03_06.22.11_LT_01.tif', cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
#smp_img = cv.medianBlur(smp_img, BLUR_SIZE)
# mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
# ROI_BORDER_SIZE = 10
# mask = cv.rectangle(mask, (10, 200),(smp_img.shape[1]-10, smp_img.shape[0]-200), 255, -1)


#lm_indexes = None# [2,4,9]
# predict_scale (None, '../KeypointMatching/predict_sandfly/M_ST12_06.11_ST_DV_02.tif', center_points, true_features, window_size= WINDOW_SIZE, \
#              candidate_method = "keypoint_on_bin_img", extractor= extr, mask_roi= None, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, lm_indexes = lm_indexes, debug= True)


# align_sample_folder(input_folder = 'G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\predict_sandfly\\', 
#                 output_folder = 'G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\predict_sandfly\\resized\\', 
#                 sample_file_name = '../KeypointMatching/training_sandfly/F_ST03_06.22.11_LT_01.tif', mask = mask)


#################### Bactrocera ##############################################################
# start_time = time.time()
extr = CHOG(radius = 20, pixel_distance= 1, block_count = 8, bin_count = 18)
center_points, true_features, sim = extract_samples ('../KeypointMatching/training_Bactrocera', extractor = extr, blur_size = BLUR_SIZE, compute_similarity= True )
print ("similarity Bactrocera:", sim)
# print ("ex run time: %.2f seconds" % (time.time() - start_time))

# predict_scale (None, '../KeypointMatching/predict_Bactrocera/red_KN(MC)C-31_RF_F-14.jpg', center_points, true_features, window_size= WINDOW_SIZE, \
#              candidate_method = "keypoint_on_bin_img", extractor= extr, mask_roi= None, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, lm_indexes = lm_indexes, debug= True)

# ########### Sandfly#################
start_time = time.time()
extr = CHOG(radius = 20, pixel_distance= 1, block_count = 8, bin_count = 18)
extr = LocalBinaryPatterns(16, 3, cell_count = 8, patchSize= 30)
center_points, true_features, sim = extract_samples ('../KeypointMatching/training_sandfly_2', extractor = extr, blur_size = BLUR_SIZE, compute_similarity= True )
print ("similarity Sandfly:", sim)
print ("ex run time: %.2f seconds" % (time.time() - start_time))
