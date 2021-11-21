import cv2 as cv
import numpy as np
import argparse
import glob
import os

def remove_particles(src, min_component_size = 150, open_kernel_size = (5,5), debug = False):
    if len(src.shape) == 3:
        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
        th = cv.adaptiveThreshold(gray,255,cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 11,2)   
    else:
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
def align(sample_img, applied_img, sample_mask = None, aplied_mask = None, descriptor = cv.AKAZE_create(), MIN_MATCH_COUNT = 4, knn_match_ratio_thr = 0.8, use_bin_img = False, debug = False):    
    if(use_bin_img):
        sample_img_bin = remove_particles(sample_img)
        applied_img_bin = remove_particles(applied_img)
        kpts1, desc1 = descriptor.detectAndCompute(sample_img_bin, sample_mask)
        kpts2, desc2 = descriptor.detectAndCompute(applied_img_bin, aplied_mask)
    else:
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
        img3 = cv.drawMatches(sample_img,kpts1,applied_img,kpts2,bestmatches[0:30],None,flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS) 
        scale_percent = 50 # percent of original size
        width = int(img3.shape[1] * scale_percent / 100)
        height = int(img3.shape[0] * scale_percent / 100)
        img3 = cv.resize(img3, (width, height), interpolation = cv.INTER_AREA)
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


def align_sample_folder(input_folder, output_folder, sample_file_name, border_top = 0, border_bottom = 0, border_left = 0, border_right = 0,  use_bin_img= False):
    smp_img = cv.imread(sample_file_name, cv.IMREAD_GRAYSCALE)   
    mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
    mask = cv.rectangle(mask, (border_left, border_top), (smp_img.shape[1] - border_right,smp_img.shape[0] - border_bottom),255, -1)

    img_type = os.path.splitext(sample_file_name)[1] 
    for file_name in glob.glob(input_folder + '*' + img_type):
        try:
            print("Aligning: ", file_name)
            only_file_name =  os.path.splitext(os.path.basename(file_name))[0]
            #if(file_name != sample_file_name):
            pre_img = cv.imread(file_name)
            pre_img_alig, affine = align(smp_img, pre_img, mask, mask, knn_match_ratio_thr= 0.85, use_bin_img= use_bin_img, debug = False)        
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
        except:
            print("ERROR file ", file_name)

#resize all images in a folder according to a sample file, save to output_folder, also update accompanied txt annotation files


def resize(input_folder, output_folder, sample_file, img_type = ".tif"):
    img_sample = cv.imread(sample_file)
    sample_height = img_sample.shape[0]    
    sample_width = img_sample.shape[1]  

    for name in glob.glob(input_folder+"*.txt"):
        print("processing file: ", name)

        img_file_name = name.replace(".txt",img_type)        
        img = cv.imread(img_file_name)
        img_resize = cv.resize(img, (sample_width, sample_height))
        base_name = os.path.basename(img_file_name)
        cv.imwrite(output_folder + base_name, img_resize)

        img_height = img.shape[0]    
        img_width = img.shape[1]    
        
        fx = img_width/sample_width
        fy = img_height/sample_height
        out_txt_file = open(output_folder + base_name.replace(img_type,".txt"), "w")
        with open(name) as f:                    
            lines = f.readlines()       
            for line in lines:
                words = line.split()            
                y = int(float(words[1]))/fy         
                x = int(float(words[0]))/fx
                out_txt_file.writelines(f"{x} {y}\n")            
        out_txt_file.close()

# mask = np.zeros(img1.shape[:2], dtype="uint8") 
# mask = cv.rectangle(mask, (0, 0), (img1.shape[1],img1.shape[0] - 200),255, -1)
# align_sample_folder(input_folder = 'G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Droso_nature_paper\\', 
#                 output_folder = 'G:\\My Drive\\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\training_nature_fine_align\\', 
#                 sample_file_name = 'G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Droso_nature_paper\\024.bmp', use_bin_img= False)


# img1 = cv.imread('G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Droso_nature_paper\\024.bmp',cv.IMREAD_GRAYSCALE) # trainImage
# img2 = cv.imread('G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Droso_nature_paper\\001.bmp',cv.IMREAD_GRAYSCALE)          # queryImage

# img, affine = align(img1, img2, mask, mask, descriptor = cv.ORB_create(), MIN_MATCH_COUNT = 4, debug = True)