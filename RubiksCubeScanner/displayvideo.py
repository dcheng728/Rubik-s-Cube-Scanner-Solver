# -*- coding: utf-8 -*-
# ################ A simple graphical interface which communicates with the server #####################################

from tkinter import *
#from vpython import *

import cv2

from PIL import Image
from PIL import ImageTk
import time
import numpy as np
import math

import start_server
import socket
import face
import cubie
from threading import Thread

cubestring = 'DUUBULDBFRBFRRULLLBRDFFFBLURDBFDFDRFRULBLUFDURRBLBDUDL'

"""

# ################################## some global variables and constants ###############################################
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '8080'
width = 50  # width of a facelet in pixels
facelet_id = [[[0 for col in range(3)] for row in range(3)] for face in range(6)]
scanner_id = [[0 for col in range(3)] for row in range(3)]
scanner_block_id = [0 for num in range(200000)]
scanner_pos = [0,0,0,0,0,0,0,0,0]
colorpick_id = [0 for i in range(6)]
curcol = None
global cameraloop
global cap
cameraloop = [1]
t = ("U", "R", "F", "D", "L", "B")
cols = ("yellow", "green", "red", "white", "blue", "orange")
scanner_color = ["white","white","white","white","white","white","white","white","white"]
thecolor = ["white","white","white","white","white","white","white","white","white"]
################################################## Opencv Functions ######################################################

"""

def cubevideo(path,action):
    path2 = path + action + '.mp4'
    cap = cv2.VideoCapture(path2)

    while(cap.isOpened()):
        ret, frame = cap.read()

        cv2.imshow('frame',frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
