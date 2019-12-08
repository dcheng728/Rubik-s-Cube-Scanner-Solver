#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 15:36:20 2018

@author: davidsoncheng
"""

# ################ A simple graphical interface which communicates with the server #####################################

from tkinter import *
import cv2
from PIL import Image
from PIL import ImageTk
import time
import socket
import cubie
import face

import start_server

from threading import Thread

cubestring = 'DUUBULDBFRBFRRULLLBRDFFFBLURDBFDFDRFRULBLUFDURRBLBDUDL'

background_thread = Thread(target=start_server.start, args=(8080, 20, 2))
background_thread.start()
#from threading import Thread

###############################################################################
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '8080'
width = 30  # width of a facelet in pixels
facelet_id = [[[0 for col in range(3)] for row in range(3)] for face in range(6)]
colorpick_id = [0 for i in range(6)]
curcol = None
t = ("U", "R", "F", "D", "L", "B")
cols = ("yellow", "green", "red", "white", "blue", "orange")
################################################################################


root = Tk()

root.geometry("1440x900")

topFrame = Frame(root)
topFrame.pack()

bottomFrame = Frame(root)
bottomFrame.pack(side=BOTTOM)

"""
button1 = Button(topFrame,text="button1",fg="red")
button1.pack()
"""



def cv22tkinter(cv2img):
    #cv2.imshow("original",cv2img)
    img = cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    return img


def create_facelet_rects(a):
    """Initializes the facelet grid on the canvas."""
    offset = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (3, 1))
    print (type(offset))
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
        
def create_colorpick_rects(a):
    """Initializes the "paintbox" on the canvas"""
    global curcol
    global cols
    for i in range(6):
        x = (i % 3)*(a+5) + 7*a
        y = (i // 3)*(a+5) + 7*a
        colorpick_id[i] = canvas.create_rectangle(x, y, x + a, y + a, fill=cols[i])
        canvas.itemconfig(colorpick_id[0], width=4)
        curcol = cols[0]

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
def solve():
    """Connects to the server and returns the solving maneuver."""
    display.delete(1.0, END)  # clear output window
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        show_text('Failed to create socket')
        return
    # host = 'f9f0b2jt6zmzyo6b.myfritz.net'  # my RaspberryPi, if online
    host = txt_host.get(1.0, END).rstrip()  # default is localhost
    port = int(txt_port.get(1.0, END))  # default is port 8080

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
    show_text(s.recv(2048).decode())

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


canvas = Canvas(root, width=12 * width + 20, height=9 * width + 20)
canvas.pack()

bclean = Button(text="Clean", height=1, width=10, relief=RAISED, command=clean)
bclean_window = canvas.create_window(10 + 10.5 * width, 10 + 7.5 * width, anchor=NW, window=bclean)
bclean.pack()


bsolve = Button(text="Solve", height=2, width=10, relief=RAISED, command=solve)
bsolve_window = canvas.create_window(10 + 10.5 * width, 10 + 6.5 * width, anchor=NW, window=bsolve)
bsolve.pack()

bempty = Button(text="Empty", height=1, width=10, relief=RAISED, command=empty)
bempty_window = canvas.create_window(10 + 10.5 * width, 10 + 8 * width, anchor=NW, window=bempty)
bempty.pack()

display = Text(height=7, width=39)
text_window = canvas.create_window(10 + 6.5 * width, 10 + .5 * width, anchor=NW, window=display)

hp = Label(text='    Hostname and Port')
hp_window = canvas.create_window(10 + 0 * width, 10 + 0.6 * width, anchor=NW, window=hp)
txt_host = Text(height=1, width=20)
txt_host_window = canvas.create_window(10 + 0 * width, 10 + 1 * width, anchor=NW, window=txt_host)
txt_host.insert(INSERT, DEFAULT_HOST)
txt_port = Text(height=1, width=20)
txt_port_window = canvas.create_window(10 + 0 * width, 10 + 1.5 * width, anchor=NW, window=txt_port)
txt_port.insert(INSERT, DEFAULT_PORT)

canvas.bind("<Button-1>", click)
create_facelet_rects(width)
create_colorpick_rects(width)


cap = cv2.VideoCapture(0)

x = 0
y = 0

while True:
    ret,image = cap.read()
    image = cv2.flip( image, 1 )
    
    cv2.circle(image,(x,y),20,(0,0,255),-1)
    img = cv22tkinter(image)
    
    panel = Label(image = img)
    panel.pack(side = "left")
    #panel.pack(side = "bottom")
    root.update_idletasks()
    root.update()
    panel.destroy()
    x = x +1
    y = y +1
    #time.sleep(0.01)
    print (x)

#img = cv22tkinter(image)






root.mainloop()


"""

while True:
    
    ret,img = cap.read()
    #img = cv2.flip( img, 1 )
    
    
    #img = cv2.imread("/Users/davidsoncheng/Desktop/artificial-intelligence-icon-1.png")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = ImageTk.PhotoImage(img)
    
    panel = Label(image = img)

    panel.pack(side = "left")
    
    root.mainloop()

    #cv2.imshow('img',img)
    k = cv2.waitKey(1000) & 0xff
    if k == 27:
        break

print ("done")

cap.release()
cap.destroyAllWindows()
"""