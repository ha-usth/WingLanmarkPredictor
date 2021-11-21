import cv2
import glob
import random
from FeatureExtract_Ha import *
import joblib
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
import sklearn
import math
import time
print(cv2.__version__)
def distance(point1,point2):
    distance_tmp=math.sqrt((point1.x-point2.x)*(point1.x-point2.x)+(point1.y-point2.y)*(point1.y-point2.y))
    return(distance_tmp)
LM=9
width=1360
height=1024

class Point:
    def __init__(self,x_init,y_init):
        self.x = x_init
        self.y = y_init

    def shift(self, x, y):
        self.x += x
        self.y += y

    def __repr__(self):
        return "".join(["Point(", str(self.x), ",", str(self.y), ")"])

landmark=[]
start_time = time.time()
x1=[]
x2=[]
for name in glob.glob("/storage/tonlh/Imorph/data/LeftWings/*.tps"):
    # print(name)
    with open(name) as f:
        lines = f.readlines()
        xy=lines[LM].split()
        point_tmp=Point(float(xy[0]),1024-float(xy[1]))
        x1.append(point_tmp.x)
        x2.append(point_tmp.y)
        #landmark.append(point_tmp)
#--------Find the bounding box of the landmark----------------------------
LM_cov=np.cov(x1,x2)
mean_x1=np.mean(x1)
mean_x2=np.mean(x2)
LM_mean=[mean_x1,mean_x2]
# print ('cov,mean = ',LM_cov,LM_mean)
candidate_points= np.random.multivariate_normal(LM_mean, LM_cov, 5000)
#-------------------------------------------------------------------------
D = 5 # number of scale
W = 8 # window size is 2W+1
Nh = 128
threshold=int(20*math.sqrt((1360*1024)/(2576*1392)))
print("threshold= ",threshold)
index=0
num_correct=0
num_prediction=0
num_test=2
num_random_point=5000
# model = joblib.load("/storage/tonlh/Imorph/Model/ERT_raw.sav")
# model = joblib.load("/storage/tonlh/Imorph/Model/ann_gsub_9.sav")
# model = joblib.load("/storage/tonlh/Imorph/Model/ERT_haar.sav")
# model = joblib.load("/storage/tonlh/Imorph/Model/ann_sSub_9.sav")
model = joblib.load("/storage/tonlh/Imorph/Model/2k/ds2k_ert_aka_9.sav")
dis_sum=0
# for name in glob.glob("/storage/tonlh/Imorph/data/LeftWings/*.tif"):
filepath = 'test_list.txt'
file1=open(filepath,'r')
Lines_tmp = file1.readlines()
for line in Lines_tmp:
    line = line.replace("\r", "").replace("\n", "") 
    name='/storage/tonlh/Imorph/data/LeftWings/'+line+'.tif'
    name1='/storage/tonlh/Imorph/data/LeftWings/'+line+'.tps'
    name2='/storage/tonlh/Imorph/data/LeftWings/'+line+'.TPS'
    index=index+1
    #if(index>num_test):
        #break
    #index=index+1
    # if(index>num_test):
    #     break
    name_tps=name[0:(len(name)-4)]
    name1=name_tps+'.TPS'
    name2=name_tps+'.tps'
    try:
        f=open(name1)
    except:
        f=open(name2)
    # print(name1)
    lines = f.readlines()
    xy=lines[LM].split()
    point_true=Point(float(xy[0]),1024-float(xy[1]))
    img_tmp=cv2.imread(name)
    listImgs = RescaleImage(name,D)
    input=[]
    points=[]
    for point in candidate_points:
        x = int (point [0])
        y = int (point [1])
    #for i in range (0,num_random_point):
    #    x=random.randrange (int(minx),int(maxx),1)
    #    y=random.randrange (int(miny),int(maxy),1)
        points.append(Point(x,y))
        listPoints = RescalePoint(x,y,D)
        #w = computeHAARLIKE(listImgs, listPoints,  W, Nh = 128, maxSizeHAAR =5)
        #w=computeRAW(listImgs, listPoints,  W)
        #w=computeUnsignedSUB(listImgs, listPoints,  W)
        gausSigma, Ng = 1, 100
        #w = computeGaussianSUB(listImgs, listPoints, gausSigma, Ng)
        # w = computeSignedSUB(listImmgs, listPoints, W)
        #w= computeBriskNonScale(img_tmp, Point(x,y))
        #w = computeSurfNonScale(img_tmp, Point(x,y))
        w = computeAkazeNonScale(img_tmp, Point(x,y))
        input.append(w)
    output=model.predict(input)
    x_tmp=0
    y_tmp=0
    num_positive=0
    for i in range(0,len(output)):
        if output[i]==1:
            num_positive=num_positive+1
            # print(points[i])
            x_tmp=x_tmp+points[i].x
            y_tmp=y_tmp+points[i].y
    if num_positive!=0:
        num_prediction=num_prediction+1
        point_predict=Point(x_tmp/num_positive,y_tmp/num_positive)
        dis=distance(point_predict,point_true)        
        # print("point_predict= ",point_predict)
        # print("point_true= ",point_true)
        # print("dis= ",dis) 
        dis_sum=dis_sum+dis
        if (dis<threshold):
            num_correct=num_correct+1
print("ert aka 9")
print("--- %s seconds ---" % (time.time() - start_time))
print("num_correct= ",num_correct)
print("num_prediction= ",num_prediction)
print("num_image= ",index-1)
print("accuracy= ",num_correct/(index-1))
print("dis_averate= ",dis_sum/(index-1))

    # print("point_predict= ",point_predict)
    # print("point_true= ",point_true)
    

# imagefile='/home/tonlh/Desktop/Code/Imorph/Data/LeftWings/egfr_F_R_oly_2X_1.tif'
# listImgs = RescaleImage(imagefile,5)
# input=[]
# points=[]
# for i in range (0,10):
#     x=random.randrange (minx,maxx,1)
#     y=random.randrange (miny,maxy,1)
#     points.append(Point(x,y))
#     listPoints = RescalePoint(x,y,D)
#     print(x,y)
#     w = computeHAARLIKE(listImgs, listPoints,  W, Nh = 8, maxSizeHAAR =5)
#     input.append(w)
# output=model.predict(input)
# print(output)
# for i in range(0,len(output)):
#     if output[i]==1:
#         print(points[i])
