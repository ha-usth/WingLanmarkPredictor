from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle
from sklearn.svm import SVC
import glob
import os
import numpy as np
import math
import random
import pickle

#lm_index = 9
#train_list="Z:\My Drive\Research\iMorphSharedByHai\Datasets\LabeledData\LeftWingsFeatures/train_list.txt"
#featureFolder ="Z:\My Drive\Research\iMorphSharedByHai\Datasets\LabeledData\LeftWingsFeaturesIndex9/
#pos_subfix_file= "_nev_sSub.txt"
#nev_subfix_file = "_pos_sSub.txt"
#outModelFile = 'ERT_gsub.sav'
def train(lm_index, train_list, featureFolder, pos_subfix_file, nev_subfix_file, outModelFile):
    X=[]
    y=[]
    img_train_list = open(train_list,'r').read().splitlines()   

    index=1
    for imgName in img_train_list:  
        feaFile = featureFolder + imgName + "_" +str(lm_index)+ pos_subfix_file
        #print (index, "PosFile: ",imgName)
        with open(feaFile) as f:
            lines = f.readlines()
            for line in lines:
                numbers = list(map(float, line.split()))
                X.append(numbers)
                y.append(1)
        index=index+1

    index=1
    for imgName in img_train_list:  
        feaFile = featureFolder + imgName + "_" +str(lm_index)+ nev_subfix_file
        #print (index, "NevFile: ",imgName)        
        with open(feaFile) as f:
            lines = f.readlines()
            for line in lines:
                numbers = list(map(float, line.split()))
                X.append(numbers)
                y.append(0)
        index=index+1 

    print (len(X))
    print (len(y))
    X, y = shuffle(X, y)
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=0)
    clf = MLPClassifier(solver='adam',activation='logistic',
    hidden_layer_sizes=(256,2),
    early_stopping=True,
    alpha=0.1,
    random_state=1,max_iter=200)
    #clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=0)
    #scores = cross_val_score(clf, X, y, cv=5)
    #print ("Score ",outModelFile, ": ",scores)
    #clf = SVC(gamma='auto')
    # print("Preict:",y_pre,"\n")
    # print("Test  :",y_test,"\n")
    # for X_ele in X_test:
    #     print(X_ele,"\n")
    #     y_pre = clf.predict(X_ele)
    #     print(" ",y_pre,"\n")
    clf.fit(X,y)
    y_pred=clf.predict(X)
    print('score=: ',accuracy_score(y, y_pred))
    pickle.dump(clf, open(outModelFile, 'wb'))
    print("finish fitting")


def test(lm_index, test_list, featureFolder, pos_subfix_file, nev_subfix_file, modelFile):
    X=[]
    y=[]
    img_list = open(test_list,'r').read().splitlines()   

    index=1
    for imgName in img_list:  
        feaFile = featureFolder + imgName + "_" +str(lm_index)+ pos_subfix_file
        print (index, "PosFile: ",imgName)
        with open(feaFile) as f:
            lines = f.readlines()
            for line in lines:
                numbers = list(map(float, line.split()))
                X.append(numbers)
                y.append(1)
        index=index+1

    index=1
    for imgName in img_list:  
        feaFile = featureFolder + imgName + "_" +str(lm_index)+ nev_subfix_file
        print (index, "NevFile: ",imgName)        
        with open(feaFile) as f:
            lines = f.readlines()
            for line in lines:
                numbers = list(map(float, line.split()))
                X.append(numbers)
                y.append(0)
        index=index+1

    print (len(X))
    print (len(y))
   
    
    model = pickle.load(open(modelFile, 'rb'))
    result = model.score(X, y)
    print(modelFile, result)
################################################################################ START TRAINING  ###############################################
train_list="train_list.txt"
test_list="test_list.txt"

lm_index = 3
featureFolder = "/storage/tonlh/Imorph/data/LeftWingsFeaturesIndex3/"
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_raw.txt", nev_subfix_file = "_nev_raw.txt", outModelFile = 'ERT_raw_9.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_sSub.txt", nev_subfix_file = "_nev_sSub.txt", outModelFile = 'ERT_sSub_9.sav')
#train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_uSub.txt", nev_subfix_file = "_nev_uSub.txt", outModelFile = 'ERT_uSub_9.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_haar.txt", nev_subfix_file = "_nev_haar.txt", outModelFile = 'ERT_haar_9.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_gSub.txt", nev_subfix_file = "_nev_gSub.txt", outModelFile = 'ERT_gSub_9.sav')                       

# test(lm_index, test_list, featureFolder, pos_subfix_file= "_pos_raw.txt", nev_subfix_file = "_nev_raw.txt", modelFile = 'ERT_raw_9.sav')            
# test(lm_index, test_list, featureFolder, pos_subfix_file= "_pos_sSub.txt", nev_subfix_file = "_nev_sSub.txt", modelFile = 'ERT_sSub_9.sav')    
#test(lm_index, test_list, featureFolder, pos_subfix_file= "_nev_uSub.txt", nev_subfix_file = "_nev_uSub.txt", modelFile = 'ERT_uSub_9.sav')    
# test(lm_index, test_list, featureFolder, pos_subfix_file= "_pos_haar.txt", nev_subfix_file = "_nev_haar.txt", modelFile = 'ERT_haar_9.sav')    
# test(lm_index, test_list, featureFolder, pos_subfix_file= "_pos_gSub.txt", nev_subfix_file = "_nev_gSub.txt", modelFile = 'ERT_gSub_9.sav')
             
# lm_index = 10
# featureFolder = "Z:\My Drive\Research\iMorphSharedByHai\Datasets\LabeledData\LeftWingsFeaturesIndex10/"

train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_raw.txt",
 nev_subfix_file = "_nev_raw.txt", outModelFile = 'ann_raw_3.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_sSub.txt", nev_subfix_file = "_nev_sSub.txt", outModelFile = 'ERT_sSub_10.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_uSub.txt", nev_subfix_file = "_nev_uSub.txt", outModelFile = 'ERT_uSub_10.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_haar.txt", nev_subfix_file = "_nev_haar.txt", outModelFile = 'ERT_haar_10.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_gSub.txt", nev_subfix_file = "_nev_gSub.txt", outModelFile = 'ERT_gSub_10.sav')           

# lm_index = 3
# featureFolder = "Z:\My Drive\Research\iMorphSharedByHai\Datasets\LabeledData\LeftWingsFeaturesIndex3/"

# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_raw.txt", nev_subfix_file = "_nev_raw.txt", outModelFile = 'ERT_raw_3.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_sSub.txt", nev_subfix_file = "_nev_sSub.txt", outModelFile = 'ERT_sSub_3.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_uSub.txt", nev_subfix_file = "_nev_uSub.txt", outModelFile = 'ERT_uSub_3.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_haar.txt", nev_subfix_file = "_nev_haar.txt", outModelFile = 'ERT_haar_3.sav')
# train(lm_index, train_list, featureFolder, pos_subfix_file= "_pos_gSub.txt", nev_subfix_file = "_nev_gSub.txt", outModelFile = 'ERT_gSub_3.sav')     
