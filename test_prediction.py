import cv2 as cv
import numpy as np
import glob
import time

from feature_extractors import CHOG, Hog, LocalBinaryPatterns
from landmark_predictor import extract_samples, predict

################################################# ds 2k, ext = LocalBinaryPatterns is the bé####################################
OPEN_KERNEL_SIZE = (5,5)
WINDOW_SIZE = 50
LBP_PATCH_SIZE = 50  #for lbp

BLUR_SIZE = 3 #good for ds 2k 
num_random = 100


#ext = LocalBinaryPatterns(8,3, cell_count = 8, patchSize= LBP_PATCH_SIZE)
ext = CHOG(radius = 20, pixel_distance= 1, block_count = 1, bin_count = 9)

center_points, true_features = extract_samples ('training2k_align', extractor = ext, blur_size = BLUR_SIZE )

smp_img = cv.imread('./training2k/egfr_F_R_oly_2X_3.tif', cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
smp_img = cv.medianBlur(smp_img,BLUR_SIZE)

mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
ROI_BORDER_SIZE = 250
mask = cv.rectangle(mask, (ROI_BORDER_SIZE,ROI_BORDER_SIZE),(smp_img.shape[1]-ROI_BORDER_SIZE, smp_img.shape[0]-ROI_BORDER_SIZE), 255, -1)

#predict (smp_img, './predict2k\egfr_F_R_oly_2X_7.tif', center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint_on_bin_img", extractor= ext, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= True)
for name in glob.glob('./predict2k/*.tif'):#for each tif
    print("Processing ", name)
    predict(smp_img, name, center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint_on_bin_img", extractor= ext, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= False)

################################################# ds nature####################################
# chog = CHOG(radius = 30, pixel_distance= 1, block_count = 1, bin_count = 9)
# BLUR_SIZE = 9
# OPEN_KERNEL_SIZE = (5,5)
# WINDOW_SIZE = 60

# start_time = time.time()
# center_points, true_features = extract_samples ('training_nature_fine_align', extractor = chog, blur_size = BLUR_SIZE )
# print ("ex run time: %.2f seconds" % (time.time() - start_time))


# smp_img = cv.imread('./training_nature_fine_align/011.bmp', cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
# smp_img = cv.medianBlur(smp_img, BLUR_SIZE)
# mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
# ROI_BORDER_SIZE = 150
# mask = cv.rectangle(mask, (ROI_BORDER_SIZE,ROI_BORDER_SIZE),(smp_img.shape[1]-ROI_BORDER_SIZE, smp_img.shape[0]-ROI_BORDER_SIZE), 255, -1)

# #predict (smp_img, './predict_nature/006.bmp', center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint", extractor= chog, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= True)
# for name in glob.glob('./predict_nature/*.bmp'):#for each tif
#     print("Processing ", name)
#     predict (smp_img, name, center_points, true_features, window_size= WINDOW_SIZE, candidate_method = "keypoint", extractor= chog, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= False)

#####################################################################################

