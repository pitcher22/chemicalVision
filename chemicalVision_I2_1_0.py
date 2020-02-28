"""
SETUP
#all imports, OS specific settings, read/load settings file
#goals: all values and flags set in a separate file (*.set), Python 2.7 and 3.X and OS independent
#tactics/issues: store the gmail info not as plaintext (pickle?)
lookup tables here as well 588to602
get rid of prompt and read settings from a file 606to621
#leave out QRcode stuff for now
"""
from __future__ import division
from __future__ import print_function
import sys
import os
import time
import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import smtplib
import imaplib
import email
import email.utils
from time import strftime
import ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


if sys.version[0:1]=='3':
    versionPython=3
else:
    versionPython=2

if sys.platform=='win32':
    versionOS='W'
elif sys.platform=='linux':
    versionOS='L'
elif sys.platform=='darwin':
    versionOS='M'
    
if versionPython==2:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
    from tkFileDialog import asksaveasfilename
    input=raw_input
else:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter.filedialog import asksaveasfilename
 
if versionOS=='W':
    upArrow=38
    dnArrow=40
    ltArrow=37
    rtArrow=39
    filePath=os.getcwd()+'\\EmailedVideo'
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = cv2.VideoWriter_fourcc(*'X264')
elif versionOS=='L':
    ltArrow=81
    upArrow=82
    rtArrow=83
    dnArrow=84
    filePath=os.getcwd()+'/EmailedVideo'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

font = cv2.FONT_HERSHEY_SIMPLEX
    
ORG_EMAIL   = "@gmail.com"
FROM_EMAIL  = "chem.sensor.up" + ORG_EMAIL
FROM_PWD    = "RubberDuck1"
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993
 
emailStartTime=time.mktime((2019,2,6,0,0,0,0,0,-1))
endTime=time.mktime((2020,2,14,18,30,0,0,0,-1))
dayLightSavings=False

if dayLightSavings:
    greenwichTimeAdjustment = 7
else:
    greenwichTimeAdjustment = 8
    
maskDiagnostic=False    
referenceFlag=True
settingsFlag=False    
RecordFlag=True
overlayFlag=True
displayHelp=True
cmPerPixel=2.54/300
cellDepth=2
cellWidth=5.5
dropVolume=0.034




"""
Low Level Functions
#include here wrapper functions for OpenCV
#goals: start with most basic functions and groups by basic task, make sure each function is its most updated version and has a docstring
#tactics/issues: move existing functions here, fix OpenCVDisplayedScatter to have adjustable color and size of points, lines, labels
"""


"""
High Level Functions
#goals:process one frame, make on display frame
specifics: fix the roated image lower right
"""

"""
Main Code
monitor email
scan email for attachment and download files
Process video
    process one frame
        which elements are calculatd?
        which elements are displayed?
create output
    create video
    create Excel
post to share/email output

"""