#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 17:54:53 2020

@author: kevin
"""
from __future__ import division
from __future__ import print_function
import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#requires: pip install opencv-python
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
import data_analysis_helpers as da
#from image_processing_source_file import *
import image_processing_source_file as ip
#import datetime

"""
SETUP
#all imports, OS specific settings, read/load settings file
#goals: all values and flags set in a separate file (*.set) Need to work on lines 121 to 228
Python 2.7 and 3.X 

lookup tables here as well 588to602
get rid of prompt and read settings from a file 606to621
#leave out QRcode stuff for now, check
"""

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
    ltArrow=2424832
    upArrow=2490368
    rtArrow=2555904
    dnArrow=2621440
    filePathEmail=os.getcwd()+'\\EmailedVideo'
    filePathSettings=os.getcwd()+'\\Settings'
    osSep='\\'
    fourcc = cv2.VideoWriter_fourcc(*'MP42')
    #fourcc = cv2.VideoWriter_fourcc(*'XVID')
    #fourcc = cv2.VideoWriter_fourcc(*'H264')
    #fourcc = cv2.VideoWriter_fourcc(*'X264')
elif versionOS=='L':
    ltArrow=65361
    upArrow=65362
    rtArrow=65363
    dnArrow=65364
    filePathEmail=os.getcwd()+'/EmailedVideo'
    filePathSettings=os.getcwd()+'/Settings'
    osSep='/'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
elif versionOS=='M':
    ltArrow=81
    upArrow=82
    rtArrow=83
    dnArrow=84
    filePathEmail=os.getcwd()+'/EmailedVideo'
    filePathSettings=os.getcwd()+'/Settings'
    osSep='/'
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')

font = cv2.FONT_HERSHEY_SIMPLEX
  
SMTP_SERVER = "imap.gmail.com"
SMTP_PORT   = 993

emailStartTime=time.mktime((2019,2,6,0,0,0,0,0,-1))
endTime=time.mktime((2020,2,14,18,30,0,0,0,-1))
dayLightSavings=False

if dayLightSavings:
    greenwichTimeAdjustment = 7
else:
    greenwichTimeAdjustment = 8
    
referenceFlag=True
settingsFlag=False    
RecordFlag=True
overlayFlag=True
displayHelp=True
cmPerPixel=2.54/300

ActiveState="Process"

linLUTfloat=np.zeros((256),dtype='float32')
linLUTint=np.zeros((256),dtype='uint8')
linLUTabs=np.zeros((256),dtype='float32')
for chan in range(256):
    val=chan/255.0
    if (val<=0.04045):
        val=val/12.92
    else:
        val=((val+0.055)/1.055)**2.4
    linLUTfloat[chan]=val
    linLUTint[chan]=int(round(val*255))
    if val==0:
        linLUTabs[chan]=255/64.0
    else:
        linLUTabs[chan]=-np.log10(val)

fileMode = input("Refresh Files? (Y/n): ")
if (fileMode == "y")|(fileMode == "Y"):
    runMode = input("Are you downloading from an email? (Y/n): ")
    if (runMode == "y")|(runMode == "Y"):
        FROM_EMAIL = input("Please enter your email address: ")
        FROM_PWD = input("Please enter your password: ")
        localFlag=False
    else:
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        video_file_path = askopenfilename(initialdir=os.getcwd(),filetypes=[('video files', '.MOV'),('all files', '.*')])
        localFlag=True
            
    useFile = input("Use settings saved in a file (f/F), or default (d/D)?")
    #read a limits file as well here to set upperLimitString
    if (useFile=="f") | (useFile=="F"):
        #include option of reading a default file on error
        root = tk.Tk()
        root.withdraw()
        root.wm_attributes('-topmost', 1)
        settings_file_path = askopenfilename(initialdir=filePathSettings,filetypes=[('settings files', '.set'),('all files', '.*')])
        settingsFile = open(settings_file_path,'r')
        settingString=settingsFile.read()
        settingsFile.close()
        dictSet=eval(settingString)
        print(dictSet)
        ActiveState="Process"
    else:
        settingsFile = open(filePathSettings+osSep+"default_settings.set",'r')
        settingString=settingsFile.read()
        settingsFile.close()
        dictSet=eval(settingString)
    settingsFile = open(filePathSettings+osSep+"upper_limit_settings.set",'r')
    settingString=settingsFile.read()
    settingsFile.close()
    dictUL=eval(settingString)