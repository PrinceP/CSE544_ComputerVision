import cv2
import numpy as np

img = cv2.imread("floor.jpg")

points = []

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img, (x, y), 1, (0, 0, 00), thickness=10)
        print(x,y)
        points.append([x,y,1])

cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    k = cv2.waitKey(1) & 0xFF
    if k==ord('m'):
    	break
cv2.destroyAllWindows()

print(points)

points = np.array(points)

img_affine = np.zeros((img.shape[0],img.shape[1]),np.uint8)
l1 = np.cross(points[0],points[1])
l2 = np.cross(points[2],points[3])
m1 = np.cross(points[1],points[2])
m2 = np.cross(points[0],points[3])

v1 = np.cross(l1,l2)
v1 = np.divide(v1,v1[2])
v2 = np.cross(m1,m2)
v2 = np.divide(v2,v2[2])

vanishL = np.cross(v1,v2)
vanishL = np.float32(vanishL)
print vanishL
vanishL = np.divide(vanishL,vanishL[2])
print vanishL

H1 = np.array([[1,0,0],[0,1,0],[vanishL[0],vanishL[1],vanishL[2]]])
print H1

for i in range(height):
    for j in range(width):
        ptxp = np.array([j,i,1],np.float32)

        ptx = np.dot(H1,ptxp)
        curpi = clip_value(0,height-1,np.int(ptx[1]/ptx[2]))
        curpj = clip_value(0,width-1,np.int(ptx[0]/ptx[2]))
        
        img_affine[curpi,curpj] = img[i,j]

print img_affine
print img
plt.imshow(img_affine,cmap='gray')







