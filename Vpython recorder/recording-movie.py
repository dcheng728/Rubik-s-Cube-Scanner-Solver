import os
import cv2

img_root = '/Users/davidsoncheng/Desktop/codes/images/B/'#Directory that the video will be saved
fps = 10    #Frames per second

fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('/Users/davidsoncheng/Desktop/codes/videos/B.avi',fourcc,fps,(600,600))

for i in range(18):
    frame = cv2.imread(img_root+'img-'+str(i)+'.png')
    videoWriter.write(frame)
videoWriter.release()

