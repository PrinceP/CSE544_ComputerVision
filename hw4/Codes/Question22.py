import cv2
import numpy as np
import os
import glob


import pandas as pd
import random
import matplotlib.pyplot as plt

'''
Code picked up from the 
http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
'''

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = (xB - xA + 1) * (yB - yA + 1)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
 
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    return iou


'''
Calculate the Tau for a given video and ten values 
bins = 20
bin size = 1/20

'''
def calculate_t(truth, prediction):
    iou_list = []
    for original,predicted in zip(truth,prediction):
        iou = bb_intersection_over_union(original,predicted)
        iou_list.append(iou)
    bins = np.linspace(0,1,20)
    freq, bins = np.histogram(iou_list,bins)
    print(freq)  



feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

"""
Read the ground truth.

"""

file = open('3/groundtruth_rect.txt', 'r')
# print(file.readline())
labels = file.readlines()
label = []
for l in labels:
    l = l.split(",")
    x = int(l[0])
    y = int(l[1])
    width = int(l[2])
    height = int(l[3])
    label.append([x,y,x+width,y+height])

"""
Read the images
"""

images = []
for filename in glob.iglob('3/**/*.jpg' , recursive = True):
    img = cv2.imread(filename)
    images.append(img)


"""
Take the images in sequence

"""

old_frame = images[0]
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
# p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
p0 = cv2.cornerHarris(old_gray,blockSize = 1, ksize = 3, k = 0.02)

mask = np.zeros_like(old_frame)
X1 = label[0][0]
Y1 = label[0][1]
X2 = label[0][2]
Y2 = label[0][3]
pt1 = (X1,Y1)
pt2 = (X2,Y2)

(a,b) = np.where(p0)

p0 = np.stack((a, b), axis=-1)
print(type(p0))

p0 = p0.reshape(-1,1,2)
p0 =  np.array(p0)
p0 = p0.astype(np.float32)
prediction = []
    

for img in images:
    frame_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    
    # Compute the previous and new points 
    good_new = p1[st==1]
    good_old = p0[st==1]
    
    # Application of affine transformation 
    gn = np.float32(good_new)[0:3]
    go = np.float32(good_old)[0:3]
    
    M = cv2.getAffineTransform(go,gn)
    
    pt1 = np.concatenate((pt1,[1]))
    pt2 = np.concatenate((pt2,[1]))
    new_pt1 = M.dot(pt1)
    new_pt2 = M.dot(pt2)
    

    img = cv2.rectangle(img,(np.int32(new_pt1[0]),np.int32(new_pt1[1])),(np.int32(new_pt2[0]),np.int32(new_pt2[1])),(0,255,0),50)
    # cv2.imshow('frame',img)
    # cv2.waitKey(0)
    k=0
    if k == 27:
        break
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)
    pt1 = new_pt1
    pt2 = new_pt2
    prediction.append(np.concatenate((new_pt1,new_pt2)))



cv2.destroyAllWindows()

calculate_t(label,prediction)