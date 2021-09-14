import cv2 as cv
import glob
from pathlib import Path

groudtruth_folder = 'G:\\My Drive\\Research\\iMorphSharedByHai\\Datasets\\LabeledData\\RightWings\\'
prediction_folder = 'G:\\My Drive\Research\\iMorph\\CodeHandcraftFeatures\\KeypointMatching\\predict2k'

out_fs = []
for i in range(1,16):
    f = open("lm_{index}.csv".format(index=i),"w")    
    f.writelines("No, image_name, x_coord_act, y_coord_act, x_coord_pre, y_coord_pre, distance\n")
    out_fs.append(f)

for name in glob.glob(prediction_folder + './*.txt'):#for each txt result file
    print("Processing ", name)
    stem = Path(name).stem
    tps_file = open(groudtruth_folder + stem + '.txt')
    tps_lines = tps_file.readlines()

    f = open(name)
    lines = f.readlines()
    index = 0
    for line in lines:        
        f_to_write = out_fs[index]        
        xy=line.split()
        x_predict=int(xy[0])
        y_predict=int(xy[1])     

        line_truth = tps_lines[index].split()
        x_truth = int(float(line_truth[0]))
        y_truth = int(float(line_truth[1]))   

        line = f"  ,{stem},{x_truth},{y_truth},{x_predict},{y_predict}\n"
        f_to_write.writelines(line)     

        index +=1
        #f_to_write("{No},{img_name},{x_act},{y_act},{x_pre},{y_pre}".format(No="", img_name = stem, x_act= x_truth, y_act = y_truth, x_pre = x_predict, y_pre = y_predict))     
for f in out_fs:
    f.close()
    


        