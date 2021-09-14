import cv2 as cv
import numpy as np
import argparse
from matplotlib import pyplot as plt
import glob
import os

def remove_particles(src, min_component_size = 150, open_kernel_size = (5,5), debug = False):
    #img = cv.medianBlur(src,3)
    th = cv.adaptiveThreshold(src,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11,2)   
    inv = 255-th

    # cv.imshow("thres", inv)
    # cv.waitKey()
    #find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(inv, connectivity=8)
    #connectedComponentswithStats yields every seperated component with information on each of them, such as size    
    sizes = stats[1:, -1]; nb_components = nb_components - 1

    # minimum size of particles we want to keep (number of pixels)
    #here, it's a fixed value, but you can set it as you want, eg the mean of the sizes or whatever    
    img2 = np.zeros((output.shape), dtype = "uint8")            
    for i in range(0, nb_components): #for every component in the image, you keep it only if it's above min_size
        if sizes[i] >= min_component_size:
            img2[output == i + 1] = 255

    kernel = np.ones((1, 1), 'uint8')
    dilate_img = cv.dilate(img2, kernel, iterations=1)
    outcome = 255-dilate_img

    kernel = np.ones(open_kernel_size,np.uint8)
    outcome = cv.morphologyEx(outcome, cv.MORPH_OPEN, kernel)    
    if debug: 
        cv.imshow("remove_particles result", outcome)
        cv.waitKey()
    return outcome


#return the aligned image and affine matrix
def align(sample_img, applied_img, sample_mask = None, aplied_mask = None, descriptor = cv.AKAZE_create(), MIN_MATCH_COUNT = 4, knn_match_ratio_thr = 0.8, debug = False):    
    kpts1, desc1 = descriptor.detectAndCompute(sample_img, sample_mask)
    kpts2, desc2 = descriptor.detectAndCompute(applied_img, aplied_mask)

    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_BRUTEFORCE_HAMMING)
    matches = matcher.knnMatch(desc1,desc2,k=2)

    # Apply ratio test    
    singleMatches = []
    for m,n in matches:        
        if m.distance < knn_match_ratio_thr*n.distance:                    
            singleMatches.append(m)
    bestmatches = sorted(singleMatches, key = lambda x:x.distance)

    if debug:
        img3 = cv.drawMatches(sample_img,kpts1,applied_img,kpts2,bestmatches[0:100],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        cv.imshow("Matching illustration", img3)
        cv.waitKey()       

    src_pts = np.float32([ kpts1[m.queryIdx].pt for m in bestmatches  ]).reshape(-1,1,2)
    dst_pts = np.float32([ kpts2[m.trainIdx].pt for m in bestmatches  ]).reshape(-1,1,2)                
    affine, mask = cv.estimateAffinePartial2D(dst_pts, src_pts)

    image_affined = cv.warpAffine(applied_img, affine,  (applied_img.shape[1],applied_img.shape[0]))   
    if debug:
        cv.imshow("aling", image_affined)
        cv.waitKey()
    return image_affined, affine    


def align_sample_folder(input_folder, output_folder, sample_file_name):
    smp_img = cv.imread(sample_file_name, cv.IMREAD_GRAYSCALE)
    img_type = os.path.splitext(sample_file_name)[1]
    mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
    ROI_BORDER_SIZE = 250
    mask = cv.rectangle(mask, (ROI_BORDER_SIZE,ROI_BORDER_SIZE),(smp_img.shape[1]-ROI_BORDER_SIZE, smp_img.shape[0]-ROI_BORDER_SIZE), 255, -1)
    for file_name in glob.glob(input_folder + '*' + img_type): 
        print("Aligning: ", file_name)
        only_file_name =  os.path.splitext(os.path.basename(file_name))[0]
        #if(file_name != sample_file_name):
        pre_img = cv.imread(file_name)
        pre_img_alig, affine = align(smp_img, pre_img, mask, mask, debug = False)        
        cv.imwrite(output_folder + only_file_name + img_type, pre_img_alig)

        txt_file = os.path.splitext(file_name)[0]  + ".txt"
        f = open(txt_file,"r")
        lines = f.readlines()
        index = 0 
        features_a_sample = []
        lms = []
        for line in lines:
            xy=line.split()
            xy[0]= int(float(xy[0]))
            xy[1]= int(float(xy[1]))
            lm = np.array([[xy[0], xy[1]]])            
            lms.append(lm)
        f.close()

        lms = np.asarray(lms)
        lms = lms.reshape((-1,1,2))
        lms_align = cv.transform(lms,affine)
        #print(lms_align)

        text_file_out = output_folder + only_file_name + ".txt"
        f = open(text_file_out, "w")     
        for lm in lms_align:
            x = int(lm[0][0])
            y = int(lm[0][1])
            cv.drawMarker(pre_img_alig, (int(lm[0][0]),int(lm[0][1])),  color=(0, 255, 0), markerType = cv.MARKER_CROSS, markerSize = 15, thickness = 1) 
            string_point = str(x) + " "+ str(y) + "\n"
            f.write(string_point)
        f.close()    

# save_path = "G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\training2k_binarized\\"
# process_path = "G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\training2k"

#save_path = "G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\training_nature_binarized\\"
#process_path = "G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\training_nature"

# save_path = "G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Sandfly_data\\France_bin\\"
# process_path = "G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Sandfly_data\France"

# for file_name in glob.glob(process_path + "/*.bmp"):    
#     ori = cv.imread(file_name,0)    
#     img = remove_particles(ori)    


predict_path = "G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\predict2k"    
save_path_aligned = "G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\predict2k_aligned\\"

# img1 = cv.imread('./training_nature_fine/011.bmp',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('./predict_nature/008.bmp',cv.IMREAD_GRAYSCALE) # trainImage
# mask = np.zeros(img1.shape[:2], dtype="uint8") 
# mask = cv.rectangle(mask, (0, 0), (img1.shape[1],img1.shape[0] - 200),255, -1)

img1 = cv.imread('G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\training2k\\egfr_F_R_oly_2X_3.tif',cv.IMREAD_GRAYSCALE)          # queryImage
# img2 = cv.imread('G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\predict2k\\star_M_R_oly_2X_86.tif',cv.IMREAD_GRAYSCALE) # trainImage
mask = np.zeros(img1.shape[:2], dtype="uint8") 
mask = cv.rectangle(mask, (0, 0), (img1.shape[1],img1.shape[0] - 200),255, -1)

#img, affine = align(img1, img2, mask, mask, descriptor = cv.ORB_create(), MIN_MATCH_COUNT = 4, knn_match_ratio = 0.8, debug = False)


# for file_name in glob.glob(predict_path + "/*.tif"):   
#     print("filename", file_name)     
#     img2 = cv.imread(file_name, cv.IMREAD_GRAYSCALE)
#     img, affine = align(img1, img2, mask, mask, descriptor = cv.AKAZE_create(), MIN_MATCH_COUNT = 4, knn_match_ratio_thr = 0.8, debug = False)
#     head, tail = ntpath.split(file_name)    
#     cv.imwrite(save_path_aligned + tail, img)

#     ori = cv.imread(file_name, cv.IMREAD_GRAYSCALE)    
#     img = ori[0: ori.shape[1]-BOTTOM_CUT, 0:ori.shape[0]]
#     matching(roi_sample, img)

#     img = remove_particles(ori)    

#     kernel = np.ones((5,5),np.uint8)
#     #img = cv.morphologyEx(img, cv.MORPH_OPEN, kernel)

    
#     img, affine = align(sample_img, img, roi_sample, roi_applied)    

#     head, tail = ntpath.split(file_name)    
#     cv.imwrite(save_path_binarized + tail, img)

#     image_affined = cv.warpAffine(ori, affine,  (ori.shape[1],ori.shape[0]))   
#     cv.imwrite(save_path_aligned + tail, image_affined)
        
#align_sample_folder(align_folder = 'training_nature_fine_align\\', sample_file_name = 'training_nature_fine\\024.bmp')
#align_sample_folder(input_folder = 'training2k\\', output_folder = 'training2k_align\\', sample_file_name = 'training2k\\egfr_F_R_oly_2X_3.tif')

