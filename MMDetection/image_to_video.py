import cv2
import numpy as np
import glob
 
img_array = []

for filename in sorted(glob.glob('./visualization/*.jpg')):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)
    print(filename)
 
 
out = cv2.VideoWriter('test.mp4',cv2.VideoWriter_fourcc(*'DIVX'), 5, size)
 
for i in range(len(img_array)):
    out.write(img_array[i])
    print(i)
out.release()