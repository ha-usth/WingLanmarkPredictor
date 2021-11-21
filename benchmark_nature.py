import cv2 as cv
import numpy as np
import glob
import time
import glob
from pathlib import Path

from feature_extractors import *
from landmark_predictor import extract_samples, extract_samples_scale, predict, predict_scale


def benchmark (extr = LocalBinaryPatterns(16, 3, cell_count = 8, patchSize= 30), sample_folder = "", sample_align_img = "", groundtruth_folder ="", predict_path = "", img_type = "*.bmp", \
                BLUR_SIZE = 5, OPEN_KERNEL_SIZE = (5,5), WINDOW_SIZE = 130, candidate_method = "keypoint", scale = 0):
    if extr is not None: 
        start_time = time.time()
        center_points, true_features = extract_samples (sample_folder, extractor = extr, blur_size = BLUR_SIZE )
        print ("ex run time: %.2f seconds" % (time.time() - start_time))

        smp_img = cv.imread(sample_align_img, cv.IMREAD_GRAYSCALE)  # sample image used for aligning predicted images
        smp_img = cv.medianBlur(smp_img, BLUR_SIZE)
        mask = np.zeros(smp_img.shape[:2], dtype="uint8") 
        mask = cv.rectangle(mask, (0,90),(smp_img.shape[1], smp_img.shape[0]-90), 255, -1)        

        start_time = time.time()
        index = 1
        for name in glob.glob(predict_path + img_type):#for each tif
            start_time_an_img = time.time()
            print("Processing {} ".format(index), name)
            index +=1
            # if(index >= 20):
            #     break
            
            predict (smp_img, name, center_points, true_features, window_size= WINDOW_SIZE, candidate_method = candidate_method, extractor= extr, mask_roi= mask, blur_size= BLUR_SIZE, open_kernel_size = OPEN_KERNEL_SIZE, debug= False)            
            print ("Predicting time: %.2f seconds" % (time.time() - start_time_an_img))
        print ("Predicting time: %.2f seconds" % (time.time() - start_time))

    #Summarize data
    results = []
    sums = [0]*len(center_points)
    for name in glob.glob(predict_path + '*.txt'):#for each txt result file
        print("Computing error ", name)
        stem = Path(name).stem
        tps_file = open(groundtruth_folder + stem + '.txt')
        tps_lines = tps_file.readlines()

        result_aline = []
        f = open(name)
        lines = f.readlines()
        index = int(0)
        for line in lines:     
            xy=line.split()
            x_predict=int(xy[0])
            y_predict=int(xy[1])     

            line_truth = tps_lines[index].split()
            x_truth = int(float(line_truth[0]))
            y_truth = int(float(line_truth[1]))   

            line = f"  ,{stem},{x_truth},{y_truth},{x_predict},{y_predict}\n" 
            error = sqrt((x_truth-x_predict)**2 + (y_truth-y_predict)**2)
            sums[index] += error
            result_aline.append(error)            
            index +=1
        results.append(result_aline)

    
    
    line = "\n" + str(extr) + " " + sample_folder
    for i in range(len(sums)):
        sums[i] = sums[i]/len(results)        
        line = line + "," + str(round(sums[i],2))
    print(line)
    f = open("benchmark.csv","a")    
    f.writelines(line)
    f.close()                    

sample_align_img = "../KeypointMatching/training_nature_align_11/011.bmp"
predict_path = '../KeypointMatching/predict_nature/'
groundtruth_folder =  'G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\Droso_nature_paper\\'

sample_folders = []
sample_folders.append('../KeypointMatching/training_nature_align_1')
sample_folders.append('../KeypointMatching/training_nature_align_1-018')
sample_folders.append('../KeypointMatching/training_nature_align_1-103')
sample_folders.append('../KeypointMatching/training_nature_align_1-136')

sample_folders.append('../KeypointMatching/training_nature_align_3')
sample_folders.append('../KeypointMatching/training_nature_align_3-1')
sample_folders.append('../KeypointMatching/training_nature_align_3-2')
sample_folders.append('../KeypointMatching/training_nature_align_3-3')
sample_folders.append('../KeypointMatching/training_nature_align_3-4')

sample_folders.append('../KeypointMatching/training_nature_align_6')
sample_folders.append('../KeypointMatching/training_nature_align_6-2')
sample_folders.append('../KeypointMatching/training_nature_align_6-3')
sample_folders.append('../KeypointMatching/training_nature_align_6-4')

sample_folders.append('../KeypointMatching/training_nature_align_8')
sample_folders.append('../KeypointMatching/training_nature_align_8-2')
sample_folders.append('../KeypointMatching/training_nature_align_8-3')
sample_folders.append('../KeypointMatching/training_nature_align_8-4')

sample_folders.append('../KeypointMatching/training_nature_align_11')
sample_folders.append('../KeypointMatching/training_nature_align_11-2')
sample_folders.append('../KeypointMatching/training_nature_align_11-3')
sample_folders.append('../KeypointMatching/training_nature_align_11-4')

sample_folders.append('../KeypointMatching/training_nature_align_15')
sample_folders.append('../KeypointMatching/training_nature_align_15-2')
sample_folders.append('../KeypointMatching/training_nature_align_15-3')
sample_folders.append('../KeypointMatching/training_nature_align_15-4')

sample_folders.append('../KeypointMatching/training_nature_align_20')



# extr = CHOG(radius = 20, pixel_distance= 1, block_count = 8, bin_count = 9)
# extr = LocalBinaryPatterns(16, 3, cell_count = 8, patchSize= 30)
# extr = Haarlike(W = 8, Nh = 8, D = 5, maxSizeHAAR= 5)
# extr = Surf(D = 1, diameter_size= 5)
# extr = StackedLocalBinaryPatterns(8, 3)
# extr = HOG(winSize = 64, blockSize =16,blockStride = 4, cellSize = 8)
# for sample_folder in sample_folders:    
#     benchmark (extr, sample_folder, sample_align_img, groundtruth_folder, predict_path, img_type = "*.bmp",  WINDOW_SIZE = 50, candidate_method = "keypoint", scale=0)

extr = CHOG(radius = 25, pixel_distance= 1, block_count = 18, bin_count = 9)
for sample_folder in sample_folders:    
    benchmark (extr, sample_folder, sample_align_img, groundtruth_folder, predict_path, img_type = "*.bmp",  WINDOW_SIZE = 30, candidate_method = "keypoint", scale=0)

# extr = LocalBinaryPatterns(16, 3, cell_count = 8, patchSize= 30)
# for sample_folder in sample_folders:    
#     benchmark (extr, sample_folder, sample_align_img, groundtruth_folder, predict_path, img_type = "*.bmp",  WINDOW_SIZE = 50, candidate_method = "keypoint", scale=0)
# extr = Haarlike(W = 8, Nh = 8, D = 5, maxSizeHAAR= 5)

# for sample_folder in sample_folders:
#     benchmark (extr, sample_folder, sample_align_img, groundtruth_folder, predict_path, img_type = "*.bmp")