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

import displayvideo
import start_server
import socket
import face
import cubie
from threading import Thread
import threading

cubestring = 'DUUBULDBFRBFRRULLLBRDFFFBLURDBFDFDRFRULBLUFDURRBLBDUDL'

background_thread = Thread(target=start_server.start, args=(8080, 20, 2))
background_thread.start()



# ################################## some global variables and constants ###############################################
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '8080'
width = 50  # width of a facelet in pixels
facelet_id = [[[0 for col in range(3)] for row in range(3)] for face in range(6)] #id for each face on rubik's cube in a list
scanner_id = [[0 for col in range(3)] for row in range(3)] #id for each face scanner
scanner_block_id = [0 for num in range(200000)]
scanner_pos = [0,0,0,0,0,0,0,0,0]
colorpick_id = [0 for i in range(6)]
curcol = None
global cameraloop #Camera loop is a value that displays the status of the camera, 1 means off, 2 means on
global cubephoto2 
global cubephoto 
global solution 
global solution_string #The string of steps that the solve function outputs
global cap #The variable for camera in opencv
sleeptime = 0
sleeptime1 = 0
solution_string = ['U1','R2','L2','R3','R1','B2','D2','F2','F2'] #Default values for solution_string
cameraloop = [1]
t = ("U", "R", "F", "D", "L", "B") #U is upper, R is right, F is front, they refer to the rotation of corresponding side of a rubik's cube
cols = ("yellow", "green", "red", "white", "blue", "orange")
scanner_color = ["white","white","white","white","white","white","white","white","white"] #Default values
thecolor = ["white","white","white","white","white","white","white","white","white"] #Default values
################################################## Opencv Functions ######################################################


def ev(img,x,y,layer): #Evaluates the average value inside a rectangle of one color channel
    
    #(x,y) is the coordinate of the center of the rectangle
    #w and l are the width and lendth of the rectangle
    
    w = 10
    h = 10
    a = list(range(x-w, x+w))
    b = list(range(y-h, y+h))
    t = 0
    tot = 0
    for c in a:
        for d in b:
            tot = tot + img[d,c,layer]
            t = t +1
            
    return int(tot/t)

def draw_aim(image,x1,y1,x2,y2,w,b,g,r):
    
    #Draws a box aim on an image to tell the user where to put the rubik's cube
    
    cv2.line(image,(x1,y1),(x1,y1+w),(b,g,r),2)
    cv2.line(image,(x1,y1),(x1+w,y1),(b,g,r),2)
    
    cv2.line(image,(x2,y1),(x2,y1+w),(b,g,r),2)
    cv2.line(image,(x2,y1),(x2-w,y1),(b,g,r),2)
    
    cv2.line(image,(x1,y2),(x1,y2-w),(b,g,r),2)
    cv2.line(image,(x1,y2),(x1+w,y2),(b,g,r),2)
    
    cv2.line(image,(x2,y2),(x2,y2-w),(b,g,r),2)
    cv2.line(image,(x2,y2),(x2-w,y2),(b,g,r),2)
    

def cv22tkinter(cv2img):
    
    #Converts a np.array into one that can be displayed in a tkinter label
    
    img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    
    return img

def cv2binary2tkinter(cv2img):
    
    #Converts a np.array into one that can be displayed in a tkinter label
    
    img = Image.fromarray(cv2img)
    img = ImageTk.PhotoImage(img)
    
    return img

def process(img):

    #Process an bgr image to binary 
    
    #kernel = np.ones((3,3),np.uint8) this is an alternative way to create kernel
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(2,2))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Corresponding grayscale image to the input
    binary = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,11,5) 
    binary_blurred = cv2.medianBlur(binary,5)
    binary_dilated = cv2.dilate(binary_blurred,kernel,iterations = 5)
    binary_inv = 255 - binary_dilated
    
    return binary_inv

def color(bgrtuple):
    
    
    #Takes a tuple input that has (b,g,r) and return the color of that pixel
    
    
    bgrtuple = list(bgrtuple)
    
    b = bgrtuple[0]
    g = bgrtuple[1]
    r = bgrtuple[2]
    #if (r >100 and  r*1.3> g > r*0.9 and r*0.9>b>r*0.7):
    if (170 < r < 200 and 200<g<230 and 200<b<230):
        return "yellow"
    if (r>180 and g<r*0.8 and b< r*0.8):
        return "orange"
    if (r-g>30 and r-b>30):
        return "red"
    if (g-b>30 and g-r>30):
    #if (g>120 and r <120 and b <120):
        return "green"
    if (b-r >30 and b - g >30):
        return "blue"
    if (g*1.2>r>g*0.8 and g*1.2>b>g*0.8):
        return "white"
    else:
        return "grey"

def get_average_color(image,image_coordinates):

    #Average the RGB values of pixels in a rectangle
    
    result_string = [[0 for col in range(3)] for face in range(9)]
    running = 0
    while (running < 9):
        x = image_coordinates[running][0]
        y = image_coordinates[running][1]
        result_string[running][0]= ev(image,x,y,0)
        result_string[running][1]= ev(image,x,y,1)
        result_string[running][2]= ev(image,x,y,2)
        running = running +1
    return result_string

def get_color_string(bgrlist):

    #Convert a list of RGB tuple values to list of color strings
    
    result_string = ["hi" for face in range(9)]
    
    running = 0
    while (running < 9):
        bgrstring = bgrlist[running]
        result_string[running] = color(tuple(bgrstring))
        running = running+1
    return result_string

def create_9box(cords_list,image_height,image_width):

    #This code is for future improvement, it will create a box at each coordinate in the cords_list
    
    """
    Takes a list of (x,y) coordinates and create a box at each coordinate.
    This can be used to check if the 9 coordinates scanned are alligned in the way
    that each block of a rubik's cube is
    """
    image = np.zeros((image_height,image_width)) 
    for num in range(9):
        x = cords_list[num-1][0]
        y = cords_list[num-1][1]
        if (x != 0 and y != 0):
            cv2.rectangle(image,(x,y),(x+50,y+50),(255,255,255),-1)
    return image

def create_referrence_color(image,color_string):

    #Shows the user what color the computer is seeing
    
    #"yellow", "green", "red", "white", "blue", "orange"
    string = [(255,255,255) for rect in range(9)]
    for n in range(len(color_string)):
        if (color_string[n] == "yellow"):
            string[n] = (0,255,200)
        elif (color_string[n] == "green"):
            string[n] = (0,255,0)
        elif (color_string[n] == "red"):
            string[n] = (0,0,255)
        elif (color_string[n] == "white"):
            string[n] = (255,255,255)
        elif (color_string[n] == "blue"):
            string[n] = (255,0,0)
        elif (color_string[n] == "orange"):
            string[n] = (10,100,255)
        else:
            string[n] = (100,100,100)
    #print (string)
    for row in range(3):
        y = 10 + row * 30
        for col in range(3):
            x = 450 + col *30
            cv2.rectangle(image,(x,y),(x+20,y+20),string[row*3+col],-1)

def pixel_distance(A, B):

    
    #Pythagrian therom to find the distance between two pixels
    
    A = (1,1)
    B = (1,1)
    (col_A, row_A) = A
    (col_B, row_B) = B

    return (math.sqrt(math.pow(col_B - col_A, 2) + math.pow(row_B - row_A, 2))+0)

def get_string(c_list):
    
    #Takes a list of (x,y) coordinates and arrange them in order from top left corner to bottom right
    
    
    x = [0 for num in range(9)]
    y = [0 for num in range(9)]
    #cords = [[0 for col in range(2)] for row in range(9)]
    for a in range(len(c_list)):
        x[a] = c_list[a][0]
        y[a] = c_list[a][1]
    xmin = min(x)
    xmax = max(x)
    
    ymin = min(y)
    ymax = max(y)

    xavg = int((xmin+xmax)/2)
    yavg = int((ymin+ymax)/2)
    
    #cv2.rectangle(img,(xmin,ymin),(xmax,ymax),(100,100,100), 5)
    string = [[xmin,ymin],[xavg,ymin],[xmax,ymin],[xmin,yavg],[xavg,yavg],[xmax,yavg],[xmin,ymax],[xavg,ymax],[xmax,ymax]]
    #string = [[xmin,ymin],[xmax,ymax]]

    return string

def sort_corners(corner1, corner2, corner3, corner4):

    #Sorts corners in a rectangle, this function is taken from the internet
    
    """
    Sort the corners such that
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right
    Return an (A, B, C, D) tuple
    """
    results = []
    corners = (corner1, corner2, corner3, corner4)

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for (x, y) in corners:
        if min_x is None or x < min_x:
            min_x = x

        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y

        if max_y is None or y > max_y:
            max_y = y

    # top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        distance = pixel_distance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    results.append(top_left)

    # top right
    top_right = None
    top_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, min_y), (x, y))
        if top_right_distance is None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance
    results.append(top_right)

    # bottom left
    bottom_left = None
    bottom_left_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((min_x, max_y), (x, y))

        if bottom_left_distance is None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance
    results.append(bottom_left)

    # bottom right
    bottom_right = None
    bottom_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, max_y), (x, y))

        if bottom_right_distance is None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance
    results.append(bottom_right)

    return results

def approx_is_square(approx, SIDE_VS_SIDE_THRESHOLD=0.70, ANGLE_THRESHOLD=20, ROTATE_THRESHOLD=30):

    #Decides whether the four_sided object entered is a square

    """
    Rules
    - there must be four corners
    - all four lines must be roughly the same length
    - all four corners must be roughly 90 degrees
    - AB and CD must be horizontal lines
    - AC and BC must be vertical lines
    SIDE_VS_SIDE_THRESHOLD
        If this is 1 then all 4 sides must be the exact same length.  If it is
        less than one that all sides must be within the percentage length of
        the longest side.
        A ---- B
        |      |
        |      |
        C ---- D
    """

    assert SIDE_VS_SIDE_THRESHOLD >= 0 and SIDE_VS_SIDE_THRESHOLD <= 1, "SIDE_VS_SIDE_THRESHOLD must be between 0 and 1"
    assert ANGLE_THRESHOLD >= 0 and ANGLE_THRESHOLD <= 90, "ANGLE_THRESHOLD must be between 0 and 90"

    # There must be four corners
    if len(approx) != 4:
        return False
    
    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)
    distances = (AB, AC, DB, DC)
    max_distance = max(distances)
    cutoff = int(max_distance * SIDE_VS_SIDE_THRESHOLD)

    # If any side is much smaller than the longest side, return False
    for distance in distances:
        if distance < cutoff:
            return False

    return True
        
# ################################################ Diverse functions ###################################################


def show_text(txt):
    
    #Displays messages.
    
    print(txt)
    display.insert(INSERT, txt)
    root.update_idletasks()


def create_facelet_rects(a):
    
    #Initializes the facelet grid on the canvas.
    
    offset = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (3, 1))
    for f in range(6):
        for row in range(3):
            y = 10 + offset[f][1] * 3 * a + row * a
            for col in range(3):
                x = 10 + offset[f][0] * 3 * a + col * a
                facelet_id[f][row][col] = canvas.create_rectangle(x, y, x + a, y + a, fill="grey")
                if row == 1 and col == 1:
                    canvas.create_text(x + width // 2, y + width // 2, font=("", 14), text=t[f], state=DISABLED)
    for f in range(6):
        canvas.itemconfig(facelet_id[f][1][1], fill=cols[f])

def create_string_rects(a,color_string):
    
    #Initializes the facelet grid on the canvas. "yellow", "green", "red", "white", "blue", "orange"

    temp = ["","","","","","","","",""]
    
    temp[0] = color_string[2]
    temp[1] = color_string[1]
    temp[2] = color_string[0]
    temp[3] = color_string[5]
    temp[4] = color_string[4]
    temp[5] = color_string[3]
    temp[6] = color_string[8]
    temp[7] = color_string[7]
    temp[8] = color_string[6]

    print (temp)
     
    color_string = temp


    
    offset = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (3, 1))
    center_color = color_string[4]
    if (center_color == "yellow"):
        f = 0
    if (center_color == "green"):
        f = 1
    if (center_color == "red"):
        f = 2
    if (center_color == "white"):
        f = 3
    if (center_color == "blue"):
        f = 4
    if (center_color == "orange"):
        f = 5
    if (center_color == "grey"):
        return
    num = 0
    for row in range(3):
        y = 10 + offset[f][1] * 3 * a + row * a
        for col in range(3):
            x = 10 + offset[f][0] * 3 * a + col * a
            facelet_id[f][row][col] = canvas.create_rectangle(x, y, x + a, y + a, fill=color_string[num])
            num = num +1
            if row == 1 and col == 1:
                canvas.create_text(x + width // 2, y + width // 2, font=("", 14), text=t[f], state=DISABLED)
    for f in range(6):
        canvas.itemconfig(facelet_id[f][1][1], fill=cols[f])

def create_scanner_rects(a,color):

    #Creates the scanned colors to tell the user what colors the computer is seeing
    
    aa = a - 30
    offset = (0,2)
    num = 0
    for row in range(3):
        y = 25 + offset[1]*3*a + row*a
        for col in range(3):
            x = 25 + offset[0]*3*a + col*a
            scanner_id[row][col] = canvas.create_rectangle(x, y, x + aa, y + aa, fill=color[num])
            num = num +1
        if (row == 1 and col == 1):
                canvas.create_text(x + aa // 2, y + aa // 2, font=("",10),text="Scan",state=DISABLED)
          

def create_colorpick_rects(a):
    
    #Initializes the "paintbox" on the canvas
    
    global curcol
    global cols
    for i in range(6):
        x = (i % 3)*(a+5) + 7*a
        y = (i // 3)*(a+5) + 7*a
        colorpick_id[i] = canvas.create_rectangle(x, y, x + a, y + a, fill=cols[i])
        canvas.itemconfig(colorpick_id[0], width=4)
        curcol = cols[0]


def get_definition_string():
    
    #Generates the cube definition string from the facelet colors
    
    color_to_facelet = {}
    for i in range(6):
        color_to_facelet.update({canvas.itemcget(facelet_id[i][1][1], "fill"): t[i]})
    s = ''
    for f in range(6):
        for row in range(3):
            for col in range(3):
                s += color_to_facelet[canvas.itemcget(facelet_id[f][row][col], "fill")]
    return s


########################################################################################################################

# ############################### Solve the displayed cube with a local or remote server ###############################


def solve():
    """Connects to the server and returns the solving maneuver."""
    display.delete(1.0, END)  # clear output window
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        show_text('Failed to create socket')
        return
    # host = 'f9f0b2jt6zmzyo6b.myfritz.net'  # my RaspberryPi, if online

    host = "localhost"
    port = 8080
        
    #host = txt_host.get(1.0, END).rstrip()  # default is localhost
    #port = int(txt_port.get(1.0, END))  # default is port 8080

    try:
        remote_ip = socket.gethostbyname(host)
    except socket.gaierror:
        show_text('Hostname could not be resolved.')
        return
    try:
        s.connect((remote_ip, port))
    except:
        show_text('Cannot connect to server!')
        return
    show_text('Connected with ' + remote_ip + '\n')
    try:
        defstr = get_definition_string()+'\n'
    except:
        show_text('Invalid facelet configuration.\nWrong or missing colors.')
        return
    show_text(defstr)
    try:
        s.sendall((defstr+'\n').encode())
    except:
        show_text('Cannot send cube configuration to server.')
        return
    global solution
    solution = (s.recv(2048).decode())
    show_text(string_decode(solution))
    #show_text(s.recv(2048).decode())
########################################################################################################################

# ################################# Functions to change the facelet colors #############################################


def clean():
    """Restores the cube to a clean cube."""
    for f in range(6):
        for row in range(3):
            for col in range(3):
                canvas.itemconfig(facelet_id[f][row][col], fill=canvas.itemcget(facelet_id[f][1][1], "fill"))


def empty():
    """Removes the facelet colors except the center facelets colors."""
    for f in range(6):
        for row in range(3):
            for col in range(3):
                if row != 1 or col != 1:
                    canvas.itemconfig(facelet_id[f][row][col], fill="grey")



def random():
    """Generates a random cube and sets the corresponding facelet colors."""
    cc = cubie.CubieCube()
    cc.randomize()
    fc = cc.to_facelet_cube()
    idx = 0
    for f in range(6):
        for row in range(3):
            for col in range(3):
                canvas.itemconfig(facelet_id[f][row][col], fill=cols[fc.f[idx]] )
                idx += 1

def capture():
    
    #takes a string and input the colors on the GUI
    
    create_string_rects(width,thecolor)

def tfcam():

    #Turn off the camera
    
    #cubevideo('/Users/davidsoncheng/Desktop/codes/videos/','U')
    global cap
    try:
        cameraloop[0] = 2
        cap.release()

    except NameError:
        return
    else:
        cameraloop[0] = 2
        cap.release()


    
def stcam():

    #Turn on the camera
    
    global cap
    try:
        if (cap.isOpened()):
            return
        else:
            cap = cv2.VideoCapture(0)
            cameraloop[0] = 1
            cameraprocess(cap)
    except NameError:
        cap = cv2.VideoCapture(0)
        cameraloop[0] = 1
        cameraprocess(cap)
    


def cubevideo(path,action):
    #global canvas2
    global sleeptime1
    path2 = path + action + '.mp4'
    cap = cv2.VideoCapture(path2)

    while(cap.isOpened()):
        try:
            ret, frame = cap.read()
            imageoutput = cv22tkinter(frame)
            
            time.sleep(0.02)
            canvas2.create_image(0, 0, image=imageoutput, anchor=NW)
            root.update_idletasks()
            root.update()
        except:
            #print ("error")
            return
    try:
        cubeoriginal = cv2.imread('/Users/davidsoncheng/Desktop/codes/videos/img-0.png')
        cubeoriginal2 = cv22tkinter(cubephoto)
        canvas2.create_image(0, 0, image=cubeoriginal2, anchor=NW)
    except:
        print ("error")
        return
    try:
        cap.release()
    except:
        #print ("error")
        return


def movestring():

    #play the rubik's cube's movement videos
    
    global solution
    global solution_string
    global sleeptime
    solution_string = string_decode(solution)
    #solution_string = solution_string.split()
    print (solution_string)
    for i in range(len(solution_string)):
        try:
            action = str(solution_string[i])
            if (action[0] == "-"):
                move = action[:2]
            else:
                move = action[0]
            number = int(action[-1])
            for i in range(number):
                cubevideo('/Users/davidsoncheng/Desktop/codes/videos/',move)
                time.sleep(1.5)
        except:
            return
    """
    for i in range(len(solution_string)):
        cubevideo_thread = Thread(target=cubevideo, args=('/Users/davidsoncheng/Desktop/codes/videos/',str(solution_string[i]))
        cubevideo_thread.run()
    """

    
def string_decode(string1):

    #Decode the string returned by solve function
    
    noend = string1[:-6]
    splitted = noend.split()
    for i in range(len(splitted)):
        if (splitted[i] == "U3"):
            splitted[i] = "-U1"
        if (splitted[i] == "D3"):
            splitted[i] = "-D1"
        if (splitted[i] == "L3"):
            splitted[i] = "-L1"
        if (splitted[i] == "R3"):
            splitted[i] = "-R1"
        if (splitted[i] == "F3"):
            splitted[i] = "-F1"
        if (splitted[i] == "B3"):
            splitted[i] = "-B1"
        else:
            splitted[i] = splitted[i]
    print (splitted)
            
    return splitted
"""
def update_slider(slider):
    global sleeptime
    global sleeptime1
    global ws
    while True:
        sleeptime = slider.get()
        sleeptime1 = slider.get()/200000
        print (sleeptime)
"""

########################################################################################################################

# ################################### Edit the facelet colors ##########################################################


def click(event):
    """Defines how to react on left mouse clicks"""
    global curcol
    idlist = canvas.find_withtag("current")
    if len(idlist) > 0:
        if idlist[0] in colorpick_id:
            curcol = canvas.itemcget("current", "fill")
            for i in range(6):
                canvas.itemconfig(colorpick_id[i], width=1)
            canvas.itemconfig("current", width=5)
        else:
            canvas.itemconfig("current", fill=curcol)
            
def processKeyboardEvent(event):
    create_string_rects(width,thecolor)
########################################################################################################################

class VideoGet:
    """
    Class that continuously gets frames from a VideoCapture object
    with a dedicated thread.
    """

    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        
    def start(self):    
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            if not self.grabbed:
                self.stop()
            else:
                (self.grabbed, self.frame) = self.stream.read()

    def stop(self):
        self.stopped = True





def cameraprocess(camera):

    #The most key component of the project, will turn on the camera and receive the image
    #Then process the image and find the rubik's cube faces using opencv functions and label them
    #Finally, return the image and list that contains the colors that were detected.
    
    while True:
        if (cameraloop[0] == 2):
            break
        ret,image = camera.read()
        
        image = cv2.flip(image,1)
        
        x,y = image.shape[0:2]
    
        image = cv2.resize(image, (int(y/2),int(x/2)))

        b,g,r = cv2.split(image)
    
        recnum = 0

        cords = [[0 for col in range(2)] for row in range(9)]

        dilation = process(image)
        
        #contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contours, hierarchy = cv2.findContours(dilation,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt,0.12*cv2.arcLength(cnt,True),True)
            x = approx.ravel()[0]
            y = approx.ravel()[1]

            if (len(approx) == 4 and 245<x<395 and 105<y<255):
            
            #Approx has 4 (x,y) coordinates, where the first is the top left,and
            #the third is the bottom right. Findind the mid point of these two coordinates
            #will give me the center of the rectangle
                recnum = recnum + 1
            
                x1=approx[0,0,0]
                y1=approx[0,0,1]
                
                x2=approx[(approx.shape[0]-2),0,0] #X coordinate of the bottom right corner
                y2=approx[(approx.shape[0]-2),0,1] 
            

                xavg = int((x1+x2)/2)
                yavg = int((y1+y2)/2)


                if (recnum > 9):
                    break
            
                cords = list(cords)
                cords[recnum-1] = [xavg,yavg]
            
                if (approx_is_square(approx) == True):
                
                    cv2.circle(image,(xavg,yavg),15,(255,255,255),5)
                #cv2.putText(image,str(b[yavg,xavg])+str(g[yavg,xavg])+str(r[yavg,xavg]),(100,recnum*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
                #cv2.drawContours(image, [approx],0,(0,0,255),2)
                if (recnum == 9 and approx_is_square(approx) == True):
                    global thecolor
                    global color_string
                    string = get_string(cords)
                    color_string = get_average_color(image,string)
                    thecolor = get_color_string(color_string)

                    create_referrence_color(image,thecolor)
                #create_scanner_rects(width,thecolor)
                #create_string_rects(width,thecolor)

        
        #pic = create_9box(cords,image.shape[0],image.shape[1])


        
        draw_aim(image,245,105,395,255,50,30,30,30)


        y,x = image.shape[0:2]
        cv2.putText(image,str(x), (int(x*0.9),20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv2.putText(image,str(y), (int(x*0.9),30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        cv2.putText(image,str(recnum), (int(x*0.9),40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))

        img = cv22tkinter(image)
        
        #img2 = cv2binary2tkinter(dilation)

        panel = Label(image = img)
        panel.place(x = 700, y = 50)
        #panel2 = Label(image = img2)
        #panel2.place(x = 700, y = 470)
        root.update_idletasks()
        root.update()
        panel.destroy()
        
#  ###################################### Generate and display the TK_widgets ##########################################
root = Tk()
root.wm_title("Solver Client")
root.geometry("1440x900")





canvas = Canvas(root, width=12 * width + 20, height=12 * width+20)
canvas.place(x = 0, y = 0)

canvas2 = Canvas(root, width=360, height=360)
canvas2.place(x=150,y=500)
cubephoto = cv2.imread('/Users/davidsoncheng/Desktop/codes/videos/img-0.png')

cubephoto2 = cv22tkinter(cubephoto)
canvas2.create_image(0, 0, image=cubephoto2, anchor=NW)

bstcam = Button(text="Turn on cam", height = 2,width = 10, relief=RAISED, command=stcam)
bstcam_window = canvas.create_window(10.5 * width, 6.5 * width, anchor=NW, window=bstcam)

btfcam = Button(text="Turn off cam", height = 2,width = 10, relief=RAISED, command=tfcam)
btfcam_window = canvas.create_window(10.5 * width, 7.25 * width, anchor=NW, window=btfcam)

bclean = Button(text="Clean", height=2, width=10, relief=RAISED, command=clean)
bclean_window = canvas.create_window(10.5 * width, 8 * width, anchor=NW, window=bclean)

bempty = Button(text="Empty", height=2, width=10, relief=RAISED, command=empty)
bempty_window = canvas.create_window(10.5 * width, 8.75 * width, anchor=NW, window=bempty)

brandom = Button(text="Random", height=2, width=10, relief=RAISED, command=random)
brandom_window = canvas.create_window(10.5 * width, 9.5 * width, anchor=NW, window=brandom)


bsolve = Button(text="Solve", height=4, width=10, relief=RAISED, command=solve)
bsolve_window = canvas.create_window(10.5 * width, 10.25 * width, anchor=NW, window=bsolve)

Ubutton = Button(text="Animate", height = 2,width = 10, relief=RAISED, command=movestring)
Ubutton_window = canvas.create_window(25, 500, anchor=NW, window=Ubutton)

w = Scale(root, from_=0, to=1000, orient=HORIZONTAL)
w.place(x = 25,y = 550)




display = Text(height=7, width=39)
text_window = canvas.create_window(10 + 6.5 * width, 10 + .5 * width, anchor=NW, window=display)


"""
hp = Label(text='    Hostname and Port')
hp_window = canvas.create_window(10 + 0 * width, 10 + 0.6 * width, anchor=NW, window=hp)


txt_host = Text(height=1, width=20)
txt_host_window = canvas.create_window(10 + 0 * width, 10 + 1 * width, anchor=NW, window=txt_host)
txt_host.insert(INSERT, DEFAULT_HOST)

txt_port = Text(height=1, width=20)
txt_port_window = canvas.create_window(10 + 0 * width, 10 + 1.5 * width, anchor=NW, window=txt_port)
txt_port.insert(INSERT, DEFAULT_PORT)
"""

"""
txt_color = Text(height=1, width=60)
txt_color_window = canvas.create_window(10 + 6.5 * width, 10 + 1.5 * width, anchor=NW, window=txt_color)
txt_color.insert(INSERT, "hi")
"""

canvas.bind("<Button-1>", click)
root.bind("<space>",processKeyboardEvent)
create_facelet_rects(width)
#create_scanner_rects(width,scanner_color)
create_colorpick_rects(width)

   

########################################################################################################################





root.mainloop()
########################################################################################################################


