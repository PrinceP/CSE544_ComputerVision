import numpy as np
import cv2
import glob

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:6,0:8].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('Set1/*')
print(images)
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,(480,360))
    img = cv2.resize(img,(480,360))
    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (6,8), None)
    # If found, add object points, image points (after refining them)
    print(ret)  
    if ret == True:
    	corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
    	objpoints.append(objp)
    	imgpoints.append(corners)
    	cv2.drawChessboardCorners(img, (6,8), corners2, ret)
    	cv2.imshow('img', img)
    	cv2.waitKey(500)
cv2.destroyAllWindows()

print ("objpoints len: " + str(len(objpoints)))
print ("imgpoints len: " + str(len(imgpoints)))

objpoints = [objpoints]
imgpoints = [imgpoints]

try:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
    datathings = (ret, mtx, dist, rvecs, tvecs)
    outf = open("calibration_return_values_rows_and_cols.pickle", "wb" )
    pickle.dump(datathings, outf)
    fieldnames = ["ret", "mtx", "dist", "rvecs", "tvecs"]
    for fieldname, data in zip(fieldnames, datathings):
        print (fieldname + ": ")
        print (data)
#     print "ret, mtx, dist, rvecs, tvecs:"
    print (ret, mtx, dist, rvecs, tvecs)
except Exception:
    print ("Failed getting cv2.calibrateCamera"+str(Exception)+"  ")
    pass

#cv2.destroyAllWindows()

