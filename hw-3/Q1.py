import cv2
import numpy as np
from matplotlib import pyplot as plt
import scipy.io as sio


loadfile = sio.loadmat("q1data.mat")
startpoints = loadfile['startpoints']
endpoints   = loadfile['endpoints']

point_x = startpoints[0]
point_y = startpoints[1]

s_points = []
e_points = []
s_points = np.vstack((point_x,point_y,np.ones(42)))


point_x = endpoints[0]
point_y = endpoints[1]
e_points = np.vstack((point_x,point_y,np.ones(42)))



Matrix1 = np.array([[np.sqrt(3),-1,1],
					[1,np.sqrt(3),1],
					[0,0,2]])
Matrix2 = np.array([[1,-1,1],
					[1,1,0],
					[0,0,1]])
Matrix3 = np.array([[1,1,0],
					[0,2,0],
					[0,0,1]])
Matrix4 = np.array([[np.sqrt(3),-1,1],
					[1,np.sqrt(3),1],
					[0.25,0.5,2]])

s_transform = np.dot(Matrix1,s_points)
s_transform = s_transform/s_transform[2:]

e_transform = np.dot(Matrix1,e_points)
e_transform = e_transform/e_transform[2:]


X = np.vstack((s_points[0,:],e_points[0,:]))
Y = np.vstack((s_points[1,:],e_points[1,:]))
plt.plot(X,Y)
plt.show()





X = np.vstack((s_transform[0,:],e_transform[0,:]))
Y = np.vstack((s_transform[1,:],e_transform[1,:]))
plt.plot(X,Y+
	)
plt.show()

