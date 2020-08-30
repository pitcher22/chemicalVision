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
    #fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    #fourcc = cv2.VideoWriter_fourcc(*'MP42')
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
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
absorbanceFlag=False
blankData=np.array([])

if dayLightSavings:
    greenwichTimeAdjustment = 7
else:
    greenwichTimeAdjustment = 8
    
# referenceFlag=True
# settingsFlag=False    
# RecordFlag=True
# overlayFlag=True
# displayHelp=True
# cmPerPixel=2.54/300

#ActiveState="Process"

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


root = tk.Tk()
root.withdraw()
root.wm_attributes('-topmost', 1)
video_file_path = askopenfilename(initialdir=os.getcwd(),filetypes=[('image files', '*.jpg | *.jpeg'),('video files', '*.mp4 | *.mkv | *.avi'),('all files', '.*')])
video_file_pathSplit = os.path.split(video_file_path)
video_file_dir=video_file_pathSplit[0]
video_file_file=video_file_pathSplit[1]
video_file_filename, video_file_file_extension = os.path.splitext(video_file_file)

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
    #ActiveState="Process"
else:
    settingsFile = open(filePathSettings+osSep+"default_settings.set",'r')
    settingString=settingsFile.read()
    settingsFile.close()
    dictSet=eval(settingString)
settingsFile = open(filePathSettings+osSep+"upper_limit_settings.set",'r')
settingString=settingsFile.read()
settingsFile.close()
dictUL=eval(settingString)

def FindLargestContour(mask):
    if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    if len(contours)>=1:
        maxArea=0
        contourIndex=0
        largestContourIndex=0
        for contour in contours:
            area=cv2.contourArea(contour)
            if area>maxArea:
                maxArea=area
                largestContourIndex=contourIndex
            contourIndex=contourIndex+1
        largestContour=contours[largestContourIndex]
        boundingRectangle=cv2.minAreaRect(largestContour)
        return(largestContour,maxArea,boundingRectangle)
    else:
        return(np.array([]),0,False)  

def FindContoursInside(mask,boundingContour,areaMin,areaMax,drawColor,frameForDrawing):
    ptsFound=np.zeros((40,4),dtype='float32')
    if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    circleIndex=0
    for contour in contours:
        area=cv2.contourArea(contour)
        if (area>=areaMin) & (area<=areaMax):
            M = cv2.moments(contour)
            if M['m00']>0:
                cx = int(M['m10']/M['m00'])
                cy = int(M['m01']/M['m00'])
                dist = cv2.pointPolygonTest(boundingContour,(cx,cy),False)
                if dist!=-1:
                    ptsFound[circleIndex,0]=cx
                    ptsFound[circleIndex,1]=cy
                    ptsFound[circleIndex,2]=area
                    ptsFound[circleIndex,3]=1
                    circleIndex=circleIndex+1
                    cv2.drawContours(frameForDrawing,[contour],0,drawColor,2)
                    cv2.circle(frameForDrawing,(int(cx),int(cy)), 2, drawColor, -1)
    return(ptsFound[0:circleIndex,:])

def RegisterImageColorRectangleFlex(frame,frameForDrawing,boxLL,boxUL,boxC1,boxC2,boxC3,boxC4,boxOR,boxWH):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvFrame, np.array(boxLL), np.array(boxUL)) 
    outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(boxMask)
    if outerBoxContour.size==0:
        return(np.array([0]),frameForDrawing)
    if outerBoxContour.size!=0:
        cv2.drawContours(frameForDrawing,[outerBoxContour],0,(255,0,255),10)
        epsilon = 0.1*cv2.arcLength(outerBoxContour,True)
        approx = cv2.approxPolyDP(outerBoxContour,epsilon,True)
        approx=approx[:,0,:]
        if approx.shape[0]!=4:
            approx = cv2.boxPoints(boxBoundingRectangle)
        position=np.sum(approx,axis=1)
        order=np.argsort(position)
        approxSort=np.copy(approx)
        approx[0,:] = approxSort[order[0],:]
        approx[1,:] = approxSort[order[1],:]
        approx[2,:] = approxSort[order[2],:]
        approx[3,:] = approxSort[order[3],:]
        distances=np.zeros(4)
        distances[1]= np.linalg.norm(approx[0,:]-approx[1,:])
        distances[2]= np.linalg.norm(approx[0,:]-approx[2,:])
        distances[3]= np.linalg.norm(approx[0,:]-approx[3,:])
        order=np.argsort(distances)
        ptsFound=np.copy(approx)
        ptsFound[0,:] = approx[order[0],:]
        ptsFound[1,:] = approx[order[1],:]
        ptsFound[2,:] = approx[order[2],:]
        ptsFound[3,:] = approx[order[3],:]
        orientation=boxOR[0]
        #these can likely be switched to the settings cl1, etc
        ptsCard = np.float32([[boxC1[0],boxC1[1]],[boxC2[0],boxC2[1]],[boxC3[0],boxC3[1]],[boxC4[0],boxC4[1]]])
        ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
        if ptsFound.shape[0]==4:
            if orientation==1:
                ptsImage[0,0]=ptsFound[0,0]
                ptsImage[0,1]=ptsFound[0,1]
                ptsImage[1,0]=ptsFound[1,0]
                ptsImage[1,1]=ptsFound[1,1]
                ptsImage[2,0]=ptsFound[2,0]
                ptsImage[2,1]=ptsFound[2,1]
                ptsImage[3,0]=ptsFound[3,0]
                ptsImage[3,1]=ptsFound[3,1]
            else:
                ptsImage[0,0]=ptsFound[3,0]
                ptsImage[0,1]=ptsFound[3,1]
                ptsImage[1,0]=ptsFound[2,0]
                ptsImage[1,1]=ptsFound[2,1]
                ptsImage[2,0]=ptsFound[1,0]
                ptsImage[2,1]=ptsFound[1,1]
                ptsImage[3,0]=ptsFound[0,0]
                ptsImage[3,1]=ptsFound[0,1]
            Mrot = cv2.getPerspectiveTransform(ptsImage,ptsCard)
            rotImage = cv2.warpPerspective(frame,Mrot,(boxWH[0],boxWH[1]))
            return(rotImage,frameForDrawing)
        else:
            return(np.array([0]),frameForDrawing)
    else:
        return(np.array([0]),frameForDrawing)

def RegisterImageColorRectangle(frame,frameForDrawing,dictSet):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvFrame, np.array(dictSet['bcr ll']), np.array(dictSet['bcr ul'])) 
    outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(boxMask)
    if outerBoxContour.size!=0:
        cv2.drawContours(frameForDrawing,[outerBoxContour],0,(255,0,255),10)
        epsilon = 0.01*cv2.arcLength(outerBoxContour,True)
        approx = cv2.approxPolyDP(outerBoxContour,epsilon,True)
        approx=approx[:,0,:]
        if approx.shape[0]!=4:
            approx = cv2.boxPoints(boxBoundingRectangle)
        position=np.sum(approx,axis=1)
        order=np.argsort(position)
        approxSort=np.copy(approx)
        approx[0,:] = approxSort[order[0],:]
        approx[1,:] = approxSort[order[1],:]
        approx[2,:] = approxSort[order[2],:]
        approx[3,:] = approxSort[order[3],:]
        distances=np.zeros(4)
        distances[1]= np.linalg.norm(approx[0,:]-approx[1,:])
        distances[2]= np.linalg.norm(approx[0,:]-approx[2,:])
        distances[3]= np.linalg.norm(approx[0,:]-approx[3,:])
        order=np.argsort(distances)
        ptsFound=np.copy(approx)
        ptsFound[0,:] = approx[order[0],:]
        ptsFound[1,:] = approx[order[1],:]
        ptsFound[2,:] = approx[order[2],:]
        ptsFound[3,:] = approx[order[3],:]
        orientation=dictSet['brt or'][0]
        #these can likely be switched to the settings cl1, etc
        ptsCard = np.float32([[dictSet['bl1 xy'][0],dictSet['bl1 xy'][1]],[dictSet['bl2 xy'][0],dictSet['bl2 xy'][1]],[dictSet['bl3 xy'][0],dictSet['bl3 xy'][1]],[dictSet['bl4 xy'][0],dictSet['bl4 xy'][1]]])
        ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
        if ptsFound.shape[0]==4:
            if orientation==1:
                ptsImage[0,0]=ptsFound[0,0]
                ptsImage[0,1]=ptsFound[0,1]
                ptsImage[1,0]=ptsFound[1,0]
                ptsImage[1,1]=ptsFound[1,1]
                ptsImage[2,0]=ptsFound[2,0]
                ptsImage[2,1]=ptsFound[2,1]
                ptsImage[3,0]=ptsFound[3,0]
                ptsImage[3,1]=ptsFound[3,1]
            elif orientation==2:
                ptsImage[0,0]=ptsFound[3,0]
                ptsImage[0,1]=ptsFound[3,1]
                ptsImage[1,0]=ptsFound[2,0]
                ptsImage[1,1]=ptsFound[2,1]
                ptsImage[2,0]=ptsFound[1,0]
                ptsImage[2,1]=ptsFound[1,1]
                ptsImage[3,0]=ptsFound[0,0]
                ptsImage[3,1]=ptsFound[0,1]
            elif orientation==3:
                ptsImage[0,0]=ptsFound[2,0]
                ptsImage[0,1]=ptsFound[2,1]
                ptsImage[1,0]=ptsFound[3,0]
                ptsImage[1,1]=ptsFound[3,1]
                ptsImage[2,0]=ptsFound[0,0]
                ptsImage[2,1]=ptsFound[0,1]
                ptsImage[3,0]=ptsFound[1,0]
                ptsImage[3,1]=ptsFound[1,1]
            elif orientation==4:
                ptsImage[0,0]=ptsFound[1,0]
                ptsImage[0,1]=ptsFound[1,1]
                ptsImage[1,0]=ptsFound[0,0]
                ptsImage[1,1]=ptsFound[0,1]
                ptsImage[2,0]=ptsFound[3,0]
                ptsImage[2,1]=ptsFound[3,1]
                ptsImage[3,0]=ptsFound[2,0]
                ptsImage[3,1]=ptsFound[2,1]
            Mrot = cv2.getPerspectiveTransform(ptsImage,ptsCard)
            rotImage = cv2.warpPerspective(frame,Mrot,(dictSet['box wh'][1],dictSet['box wh'][0]))
            return(rotImage,frameForDrawing)
        else:
            return(np.array([0]),frameForDrawing)
    else:
        return(np.array([0]),frameForDrawing)
    
def RegisterImageColorCard(frame,frameForDrawing,dictSet):
    hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    boxMask = cv2.inRange(hsvFrame, np.array(dictSet['box ll']), np.array(dictSet['box ul'])) 
    c12CircleMask = cv2.inRange(hsvFrame, np.array(dictSet['c12 ll']), np.array(dictSet['c12 ul'])) 
    c34CircleMask = cv2.inRange(hsvFrame, np.array(dictSet['c34 ll']), np.array(dictSet['c34 ul'])) 
    outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(boxMask)
    cv2.drawContours(frameForDrawing,[outerBoxContour],0,(0,255,0),2)
    ptsC12 = FindContoursInside(c12CircleMask,outerBoxContour,boxArea*0.005,boxArea*0.25,(255,0,0),frameForDrawing)    
    ptsC34 = FindContoursInside(c34CircleMask,outerBoxContour,boxArea*0.005,boxArea*0.25,(0,0,255),frameForDrawing)    
    ptsFound = np.concatenate((ptsC12, ptsC34), axis=0) 
    ptsCard = np.float32([[dictSet['cl1 xy'][0],dictSet['cl1 xy'][1]],[dictSet['cl2 xy'][0],dictSet['cl2 xy'][1]],[dictSet['cl3 xy'][0],dictSet['cl3 xy'][1]],[dictSet['cl4 xy'][0],dictSet['cl4 xy'][1]]])
    ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
    if ptsFound.shape[0]==4:
        if (cv2.pointPolygonTest(outerBoxContour,ip.MidPoint(ptsFound[0,0:2],ptsFound[2,0:2]),False)==-1) & (cv2.pointPolygonTest(outerBoxContour,ip.MidPoint(ptsFound[0,0:2],ptsFound[3,0:2]),False)==-1):
            ptsImage[0,0]=ptsFound[0,0]
            ptsImage[0,1]=ptsFound[0,1]
            ptsImage[1,0]=ptsFound[1,0]
            ptsImage[1,1]=ptsFound[1,1]
        else:
            ptsImage[1,0]=ptsFound[0,0]
            ptsImage[1,1]=ptsFound[0,1]
            ptsImage[0,0]=ptsFound[1,0]
            ptsImage[0,1]=ptsFound[1,1]
        if (cv2.pointPolygonTest(outerBoxContour,ip.MidPoint(ptsImage[1,0:2],ptsFound[2,0:2]),False)==-1):
            ptsImage[2,0]=ptsFound[2,0]
            ptsImage[2,1]=ptsFound[2,1]
            ptsImage[3,0]=ptsFound[3,0]
            ptsImage[3,1]=ptsFound[3,1]
        else:
            ptsImage[3,0]=ptsFound[2,0]
            ptsImage[3,1]=ptsFound[2,1]
            ptsImage[2,0]=ptsFound[3,0]
            ptsImage[2,1]=ptsFound[3,1]
        Mrot = cv2.getPerspectiveTransform(ptsImage,ptsCard)
        #the last tulpe below needs to be in settings
        rotImage = cv2.warpPerspective(frame,Mrot,(2600,900))
        return(rotImage,frameForDrawing)
    else:
        return(np.array([0]),frameForDrawing)

def WhiteBalanceFrame(displayFrame,rotImage,frame,frameForDrawing,dictSet,wbList=["WB1"]):
    rgbWBR=np.zeros((rotImage.shape),dtype='uint8')
    for wbRegion in wbList:
        rgbWBR[dictSet[wbRegion+' xy'][1]:dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1], dictSet[wbRegion+' xy'][0]:dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0]] = rotImage[dictSet[wbRegion+' xy'][1]:dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1], dictSet[wbRegion+' xy'][0]:dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0]]
        cv2.rectangle(frameForDrawing,(dictSet[wbRegion+' xy'][0],dictSet[wbRegion+' xy'][1]),(dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0],dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1]),(0,0,255),10 )
        if dictSet[wbRegion+' hs'][2]!=0:
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,wbRegion,dictSet,connectedOnly=False,histogramHeight=dictSet['dsp wh'][1])
            displayFrame=OpenCVComposite(histogramImage, displayFrame, dictSet[wbRegion+' hs'])
        else:
            valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,wbRegion,dictSet,connectedOnly=False)                
        if dictSet[wbRegion+' ds'][2]!=0:
            displayFrame=OpenCVComposite(resRGB, displayFrame, dictSet[wbRegion+' ds'])
    hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
    maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
    if np.sum(maskWBR)>0:
        RGBGreyWBR=cv2.mean(rgbWBR, mask=maskWBR)
        bscale=RGBGreyWBR[0]
        gscale=RGBGreyWBR[1]
        rscale=RGBGreyWBR[2]
        if dictSet['WBR sl'][0]!=0:
            scalemin=dictSet['WBR sl'][0]
        else:
            scalemin=min(rscale,gscale,bscale)
        if (scalemin!=0) & (min(rscale,gscale,bscale)!=0):
            rfactor=float(scalemin)/float(rscale)
            gfactor=float(scalemin)/float(gscale)
            bfactor=float(scalemin)/float(bscale)
        rgbWBR=ip.OpenCVRebalanceImage(rgbWBR,rfactor,gfactor,bfactor)
        rgbWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)
        rotImage=ip.OpenCVRebalanceImage(rotImage,rfactor,gfactor,bfactor)
        frame=ip.OpenCVRebalanceImage(frame,rfactor,gfactor,bfactor)
    return(rgbWBR,rotImage,frame,frameForDrawing)

def OpenCVComposite(sourceImage, targetImage,settingsWHS):
    if (sourceImage.size==0) or (sourceImage.shape[1]==0) or (sourceImage.shape[0]==0):
        return targetImage
    if settingsWHS[2]!=100:
        scaleFactor=settingsWHS[2]/100
        if (int(sourceImage.shape[1]*scaleFactor)>0) and (int(sourceImage.shape[0]*scaleFactor)>0):
            imageScaled = cv2.resize(sourceImage, (int(sourceImage.shape[1]*scaleFactor),int(sourceImage.shape[0]*scaleFactor)), interpolation = cv2.INTER_AREA)
        else:
            imageScaled=sourceImage
    else:
        imageScaled=sourceImage
    xTargetStart=int(targetImage.shape[0]*settingsWHS[1]/100)
    xTargetEnd=int((targetImage.shape[0]*settingsWHS[1]/100)+imageScaled.shape[0])
    yTargetStart=int(targetImage.shape[1]*settingsWHS[0]/100)
    yTargetEnd=int((targetImage.shape[1]*settingsWHS[0]/100)+imageScaled.shape[1])
    if xTargetEnd>targetImage.shape[0]:
        xTargetEnd=targetImage.shape[0]
        xSourceEnd=targetImage.shape[0]-int(targetImage.shape[0]*settingsWHS[1]/100)
    else:
        xSourceEnd=imageScaled.shape[0]
    
    if yTargetEnd>targetImage.shape[1]:
        yTargetEnd=targetImage.shape[1]
        ySourceEnd=targetImage.shape[1]-int(targetImage.shape[1]*settingsWHS[0]/100)
    else:
        ySourceEnd=imageScaled.shape[1]
    if len(imageScaled.shape)==3:
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,:]=imageScaled[0:xSourceEnd,0:ySourceEnd,:]
    else:
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,0]=imageScaled[0:xSourceEnd,0:ySourceEnd]
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,1]=imageScaled[0:xSourceEnd,0:ySourceEnd]
        targetImage[xTargetStart:xTargetEnd,yTargetStart:yTargetEnd,2]=imageScaled[0:xSourceEnd,0:ySourceEnd]
    return targetImage

def DisplayAllSettings(dictSet,parmWidth,parmHeight,displayFrame,fontScale):
    setRow=0
    activeSettingsRow=dictSet['set rc'][0]
    activeSettingsColumn=dictSet['set rc'][1]
    if activeSettingsColumn>len(dictSet[sorted(dictSet)[activeSettingsRow]])-1:
        activeSettingsColumn=len(dictSet[sorted(dictSet)[activeSettingsRow]])-1
        dictSet['set rc'][1]=activeSettingsColumn
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)): 
        if (activeSettingsRow==setRow):
            setColor=(0,0,255)
        else:
            setColor=(255,255,255)
        ip.OpenCVPutText(displayFrame, setting, (int(parmWidth*0.2),parmHeight*(setRow+1)), setColor, fontScale = fontScale)
        if activeSettingsColumn>len(dictSet[sorted(dictSet)[activeSettingsRow]])-1:
            activeSettingsColumn=len(dictSet[sorted(dictSet)[activeSettingsRow]])-1
        for setCol in range(len(dictSet[setting])):
            if (activeSettingsColumn==setCol) & (activeSettingsRow==setRow):
                setColor=(0,0,255)
            else:
                setColor=(255,255,255)
            ip.OpenCVPutText(displayFrame,str(dictSet[setting][setCol]),(parmWidth*(setCol+2),parmHeight*(setRow+1)),setColor, fontScale = fontScale)
    return displayFrame

def DisplaySomeSettings(dictSet,parmWidth,parmHeight,displayFrame,numRowsPad,fontScale):
    settings=sorted(dictSet)
    setRow=0
    activeSettingsRow=dictSet['set rc'][0]
    activeSettingsColumn=dictSet['set rc'][1]
    if activeSettingsRow-numRowsPad>=0:
        startRow=activeSettingsRow-numRowsPad
    else:
        startRow=0
    if activeSettingsRow+numRowsPad<=len(settings):
        endRow=activeSettingsRow+numRowsPad
    else:
        endRow=len(settings)    
    numRows=endRow-startRow
    if activeSettingsColumn>len(dictSet[settings[activeSettingsRow]])-1:
        activeSettingsColumn=len(dictSet[settings[activeSettingsRow]])-1
        dictSet['set rc'][1]=activeSettingsColumn
    for numRow,setRow,setting in zip(range(numRows),range(startRow,endRow),settings[startRow:endRow]): 
        if (activeSettingsRow==setRow):
            setColor=(0,0,255)
        else:
            setColor=(255,255,255)
        ip.OpenCVPutText(displayFrame, setting, (int(parmWidth*0.2),parmHeight*(numRow+1)), setColor, fontScale = fontScale)
        if activeSettingsColumn>len(dictSet[settings[activeSettingsRow]])-1:
            activeSettingsColumn=len(dictSet[settings[activeSettingsRow]])-1
        for setCol in range(len(dictSet[setting])):
            if (activeSettingsColumn==setCol) & (activeSettingsRow==setRow):
                setColor=(0,0,255)
            else:
                setColor=(255,255,255)
            ip.OpenCVPutText(displayFrame,str(dictSet[setting][setCol]),(parmWidth*(setCol+2),parmHeight*(numRow+1)),setColor, fontScale = fontScale)
    return displayFrame

def SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=True,histogramHeight=0):
    rgbROI = rotImage[dictSet[roiSetName+' xy'][1]:dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1], dictSet[roiSetName+' xy'][0]:dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0]]
    if rgbROI.size==0:
        #return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI,contourArea,boundingRectangle,False)
        return(np.array([0,0,0,0,0,0,0,0,0,0,0,0]),np.array([0,0,0,0,0,0,0,0,0,0,0,0]),np.array([]),np.array([]),np.array([]),np.array([]),np.array([]))
    hsvROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2HSV)
    hsvROI[:,:,0]=ip.ShiftHOriginToValue(hsvROI[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
    labROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2LAB)
    absROI=cv2.LUT(rgbROI, linLUTabs)*64
    if roiSetName[0:2]=="WB":
        maskROI = cv2.inRange(hsvROI, np.array(dictSet['WBR'+' ll']), np.array(dictSet['WBR'+' ul']))
    else:        
        maskROI = cv2.inRange(hsvROI, np.array(dictSet[roiSetName+' ll']), np.array(dictSet[roiSetName+' ul']))
    #following is only necessary if finding largest connected contour
    if connectedOnly:
        contourROI,contourArea,boundingRectangle=FindLargestContour(maskROI)
        contourMask=np.zeros((maskROI.shape),dtype='uint8')
        if contourArea>0:
            cv2.drawContours(contourMask,[contourROI],0,(255),-1)
        resMask = cv2.bitwise_and(maskROI,maskROI, mask= contourMask)
    else:
        resMask = maskROI
        boundingRectangle=((0, 0),(0,0),0)
        contourArea=0
    resFrameROI = cv2.bitwise_and(rgbROI,rgbROI, mask= resMask)
    rgbROIsummary=cv2.meanStdDev(rgbROI,mask=resMask)
    hsvROIsummary=cv2.meanStdDev(hsvROI,mask=resMask)
    labROIsummary=cv2.meanStdDev(labROI,mask=resMask)
    absROIsummary=cv2.meanStdDev(absROI,mask=resMask)
    allROIsummary=np.concatenate((rgbROIsummary,hsvROIsummary,labROIsummary,absROIsummary),axis=1)
    tempROIsummary=np.copy(allROIsummary)
    allROIsummary[:,0,:]=tempROIsummary[:,2,:]
    allROIsummary[:,2,:]=tempROIsummary[:,0,:]
    allROIsummary[:,9,:]=tempROIsummary[:,11,:]
    allROIsummary[:,11,:]=tempROIsummary[:,9,:]
    if histogramHeight!=0:
        inputImages= [rgbROI,rgbROI,rgbROI,hsvROI,hsvROI,hsvROI,labROI,labROI,labROI,absROI,absROI,absROI]
        rows=range(len(inputImages))
        displayColors=[(0,0,255),(0,255,0),(255,50,50),(255,255,0),(200,200,200),(128,128,128),(255,255,255),(255,0,255),(0,255,255),(0,0,255),(0,255,0),(255,50,50)]
        channels=[2,1,0,0,1,2,0,1,2,2,1,0]
        labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba"]
        singleHeight=int((histogramHeight-10)/len(inputImages))
        histogramFrame = np.zeros((histogramHeight, 255+20, 3), np.uint8)
        for row, displayColor, inputImage, channel, label in zip(rows, displayColors, inputImages, channels,labels):              
            mean,std,most=ip.OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,5,row*singleHeight+5,256,singleHeight-15,histogramFrame,displayColor,5,True,label)
        return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI,contourArea,boundingRectangle,histogramFrame)
    else:
        return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI,contourArea,boundingRectangle,False)
        
def ProcessOneFrame(frame,dictSet,displayFrame,wbList=["WB1"],roiList=["RO1"]):
    frameForDrawing=np.copy(frame)
    frameStats=np.zeros((16,6,len(roiList)))    
    if dictSet['flg rf'][0]==1:
        rotImage,frameForDrawing = RegisterImageColorCard(frame,frameForDrawing,dictSet)
        skipFrame=False
        if rotImage.size==1:
            skipFrame=True
            rotImage = np.copy(frame)
    if dictSet['flg rf'][0]==2:
        rotImage,frameForDrawing = RegisterImageColorRectangle(frame,frameForDrawing,dictSet)
        #rotImage,frameForDrawing = RegisterImageColorRectangleFlex(frame,frameForDrawing,?????)
        #RegisterImageColorRectangleFlex(frame,frameForDrawing,boxLL,boxUL,boxC1,boxC2,boxC3,boxC4,boxOR,boxWH)
        
        skipFrame=False
        if rotImage.size==1:
            skipFrame=True
            rotImage = np.copy(frame)
    else:
        rotImage = np.copy(frame)
        skipFrame=False
    rotForDrawing=np.copy(rotImage)
    if skipFrame==False:
        if dictSet['flg wb'][0]==1:
            rgbWBR,rotImage,frame,rotForDrawing=WhiteBalanceFrame(displayFrame,rotImage,frame,rotForDrawing,dictSet,wbList=wbList)
            if dictSet['flg di'][0]==1:
                cv2.imshow("WBR",rgbWBR)
            #if dictSet['WBR ds'][2]!=0:
            #    displayFrame=OpenCVComposite(rgbWBR, displayFrame, dictSet['WBR ds'])
            #hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
            #maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
            #rgbWBRsummary=cv2.meanStdDev(rgbWBR,mask=maskWBR)
            #resFrameWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)
        if dictSet['flg di'][0]==1:
            cv2.imshow("RotatedImage",rotImage)
        for roiSetName,roiNumber in zip(roiList,range(len(roiList))):
            if dictSet[roiSetName+' hs'][2]!=0:
                valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=dictSet[roiSetName+' ct'][0],histogramHeight=dictSet['dsp wh'][1])
                displayFrame=OpenCVComposite(histogramImage, displayFrame, dictSet[roiSetName+' hs'])
            else:
                valSummary,stdSummary,resMask,resRGB,contourArea,boundingRectangle,histogramImage=SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=dictSet[roiSetName+' ct'][0])
            cv2.rectangle(rotForDrawing,(dictSet[roiSetName+' xy'][0],dictSet[roiSetName+' xy'][1]),(dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0],dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1]),(0,255,0),10 )
            frameStats[0:12,0,roiNumber]=valSummary
            frameStats[0:12,1,roiNumber]=stdSummary
            area=cv2.countNonZero(resMask)
            frameStats[12,0,roiNumber]=area
            if area>0:
                frameStats[13,0,roiNumber]=boundingRectangle[1][0]
                frameStats[14,0,roiNumber]=boundingRectangle[1][1]
                frameStats[15,0,roiNumber]=contourArea
            if dictSet['flg di'][0]==1:
                cv2.imshow(roiSetName,resRGB)
            if dictSet[roiSetName+' ds'][2]!=0:
                displayFrame=OpenCVComposite(resRGB, displayFrame, dictSet[roiSetName+' ds'])
            if dictSet[roiSetName+' cs'][2]!=0:
                #box = cv2.boxPoints(boundingRectangle)
                if resMask.size>1:
                    x,y,w,h = cv2.boundingRect(resMask)
                #displayFrame=OpenCVComposite(resRGB[x:x+w,y:y+h,:], displayFrame, dictSet[roiSetName+' cs'])
                    displayFrame=OpenCVComposite(resRGB[y:y+h,x:x+w,:], displayFrame, dictSet[roiSetName+' cs'])
    return frameStats,displayFrame,frame,frameForDrawing,rotImage,rotForDrawing

def ToggleFlag(flagName,dictSet):
    if dictSet[flagName][0]==1:
        dictSet[flagName][0]=0
    else:
        dictSet[flagName][0]=1
    return dictSet
    
def CheckKeys(dictSet):
    keypress=cv2.waitKeyEx(1)
    #print(keypress)
    changeCameraFlag=False
    continueFlag=True
    frameJump=0
    #quit
    if keypress == ord('q'):
        continueFlag=False
    #display diagnostic regions
    if keypress == ord('i'):
        dictSet=ToggleFlag('flg di',dictSet)
    #display settings
    if keypress == ord('t'):
        dictSet=ToggleFlag('flg ds',dictSet)
    #record
    if keypress == ord('r'):
        dictSet=ToggleFlag('flg rc',dictSet)
    #register image
    if keypress == ord('c'):
        dictSet=ToggleFlag('flg rf',dictSet)
    #run video
    if keypress == ord('p'):
        dictSet=ToggleFlag('flg rn',dictSet)
    if keypress == ord('l'):
        frameJump=10
    if keypress == ord('h'):
        frameJump=-10
    if keypress == ord('k'):
        frameJump=1
    if keypress == ord('j'):
        frameJump=-1
    if dictSet['flg ds'][0]==1:
        row=dictSet['set rc'][0]
        col=dictSet['set rc'][1]
        if sorted(dictSet)[row][5]=='l':
            hLimit=255
            lLimit=0
        else:
            hLimit=7000
            lLimit=-100
        if (keypress==ord('+')) & (dictSet[sorted(dictSet)[row]][col]<hLimit):
            dictSet[sorted(dictSet)[row]][col]=dictSet[sorted(dictSet)[row]][col]+1
            if sorted(dictSet)[row].find('CAM')==0:
                changeCameraFlag=True
        if (keypress==ord('-')) & (dictSet[sorted(dictSet)[row]][col]>lLimit):
            dictSet[sorted(dictSet)[row]][col]=dictSet[sorted(dictSet)[row]][col]-1    
            if sorted(dictSet)[row].find('CAM')==0:
                changeCameraFlag=True
        if (keypress==ord('>')) & (dictSet[sorted(dictSet)[row]][col]<hLimit-9):
            dictSet[sorted(dictSet)[row]][col]=dictSet[sorted(dictSet)[row]][col]+10
            if sorted(dictSet)[row].find('CAM')==0:
                changeCameraFlag=True
        if (keypress==ord('<')) & (dictSet[sorted(dictSet)[row]][col]>lLimit):
            dictSet[sorted(dictSet)[row]][col]=dictSet[sorted(dictSet)[row]][col]-10   
            if sorted(dictSet)[row].find('CAM')==0:
                changeCameraFlag=True
        if (keypress==ord('.')) & (dictSet[sorted(dictSet)[row]][col]<hLimit-9):
            dictSet[sorted(dictSet)[row]][col]=dictSet[sorted(dictSet)[row]][col]+100
            if sorted(dictSet)[row].find('CAM')==0:
                changeCameraFlag=True
        if (keypress==ord(',')) & (dictSet[sorted(dictSet)[row]][col]>lLimit):
            dictSet[sorted(dictSet)[row]][col]=dictSet[sorted(dictSet)[row]][col]-100  
            if sorted(dictSet)[row].find('CAM')==0:
                changeCameraFlag=True
        if ((keypress==upArrow) | (keypress==ord('w'))) & (row>0):
            dictSet['set rc'][0]=row-1
        if ((keypress==dnArrow) | (keypress==ord('s'))) & (row<len(dictSet)-1):
            dictSet['set rc'][0]=row+1
        if ((keypress==ltArrow) | (keypress==ord('a'))) & (col>0):
            dictSet['set rc'][1]=col-1
        if ((keypress==rtArrow) | (keypress==ord('d'))) & (col<len(dictSet[sorted(dictSet)[row]])-1):
            dictSet['set rc'][1]=col+1
    return(keypress,dictSet,continueFlag,changeCameraFlag,frameJump)

def MakeTimePlots(parameterStats,dictSet,displayFrame,frameStart=0,frameEnd=0):
    pltList=[]
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)):
        if (setting[0:2]=="fg") & (setting[4:6]=="wh"):
            if (dictSet[setting][0]!=0) & (dictSet[setting][1]!=0):
                pltList.append(setting[0:3])
    if frameEnd==0:
        frameEnd=parameterStats.shape[2]
    for axis in pltList:
        if dictSet[axis+' ds'][2]!=0:
            xBool=parameterStats[31,0,frameStart:frameEnd,dictSet[axis+' xc'][2]]==1
            yBool=parameterStats[31,0,frameStart:frameEnd,dictSet[axis+' yc'][2]]==1
            if (xBool==yBool).all() & (np.sum(xBool)>1):               
                if dictSet[axis+' xs'][0]==0:
                    xMin=dictSet[axis+' xs'][1]
                    xMax=dictSet[axis+' xs'][2]
                else:
                    xMin=None
                    xMax=None          
                if dictSet[axis+' ys'][0]==0:
                    yMin=dictSet[axis+' ys'][1]
                    yMax=dictSet[axis+' ys'][2]
                else:
                    yMin=None
                    yMax=None
                xData=parameterStats[dictSet[axis+' xc'][0],dictSet[axis+' xc'][1],xBool,dictSet[axis+' xc'][2]]
                yData=parameterStats[dictSet[axis+' yc'][0],dictSet[axis+' yc'][1],yBool,dictSet[axis+' yc'][2]]
                scatterFrame = np.zeros((dictSet[axis+' wh'][1], dictSet[axis+' wh'][0], 3), np.uint8)
                ip.OpenCVDisplayedScatter(scatterFrame,xData,yData,0,0,dictSet[axis+' wh'][0],dictSet[axis+' wh'][1],(dictSet[axis+' cl'][0],dictSet[axis+' cl'][1],dictSet[axis+' cl'][2]),1,ydataRangemin=yMin,ydataRangemax=yMax,xdataRangemin=xMin,xdataRangemax=xMax)
                displayFrame=OpenCVComposite(scatterFrame, displayFrame, dictSet[axis+' ds'])
    return displayFrame

def MakeFramePlots(dictSet,displayFrame,rgbROI,blankData=np.array([]),calFlag=False):
    pltList=[]
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)):
        if (setting[0:2]=="ff") & (setting[4:6]=="wh"):
            if (dictSet[setting][0]!=0) & (dictSet[setting][1]!=0):
                pltList.append(setting[0:3])
    rgbROI=cv2.LUT(rgbROI, linLUTfloat)
    vSum=np.sum(rgbROI[:,:,:],axis=0)
    if calFlag==True:
        #xaxis=lastCal
        #labelFlag=True
        xLow=400
        xHigh=700
        #xFilter=[(xaxis>=xLow) & (xaxis<=xHigh)]
    else:
        xaxis=np.arange(rgbROI.shape[1])        
        #labelFlag=False
        xLow=0
        xHigh=rgbROI.shape[1]
        xFilter=(xaxis>=xLow) & (xaxis<=xHigh)
    for axis in pltList:
        if dictSet[axis+' ds'][2]!=0:
            if dictSet[axis+' xs'][0]==0:
                xMin=dictSet[axis+' xs'][1]
                xMax=dictSet[axis+' xs'][2]
            else:
                xMin=None
                xMax=None          
            if dictSet[axis+' ys'][0]==0:
                yMin=dictSet[axis+' ys'][1]
                yMax=dictSet[axis+' ys'][2]
            else:
                yMin=None
                yMax=None
            scatterFrame = np.zeros((dictSet[axis+' wh'][1], dictSet[axis+' wh'][0], 3), np.uint8)
            if blankData.size==0:
                xData=xaxis[xFilter]
                yData=vSum[xFilter][:,0]
                ip.OpenCVDisplayedScatter(scatterFrame,xData,yData,0,0,dictSet[axis+' wh'][0],dictSet[axis+' wh'][1],(255,100,100),1,ydataRangemin=yMin,ydataRangemax=yMax,xdataRangemin=xMin,xdataRangemax=xMax)
                yData=vSum[xFilter][:,1]
                ip.OpenCVDisplayedScatter(scatterFrame,xData,yData,0,0,dictSet[axis+' wh'][0],dictSet[axis+' wh'][1],(100,255,100),1,ydataRangemin=yMin,ydataRangemax=yMax,xdataRangemin=xMin,xdataRangemax=xMax)
                yData=vSum[xFilter][:,2]
                ip.OpenCVDisplayedScatter(scatterFrame,xData,yData,0,0,dictSet[axis+' wh'][0],dictSet[axis+' wh'][1],(100,100,255),1,ydataRangemin=yMin,ydataRangemax=yMax,xdataRangemin=xMin,xdataRangemax=xMax)
                yData=np.sum(vSum[xFilter],axis=1)
                ip.OpenCVDisplayedScatter(scatterFrame,xData,yData,0,0,dictSet[axis+' wh'][0],dictSet[axis+' wh'][1],(255,255,255),1,ydataRangemin=yMin,ydataRangemax=yMax,xdataRangemin=xMin,xdataRangemax=xMax)
                displayFrame=OpenCVComposite(scatterFrame, displayFrame, dictSet[axis+' ds'])
            else:
                xData=xaxis[xFilter]
                yData=np.sum(vSum[xFilter],axis=1)
                yData=-np.log(yData/blankData)
                ip.OpenCVDisplayedScatter(scatterFrame,xData,yData,0,0,dictSet[axis+' wh'][0],dictSet[axis+' wh'][1],(255,255,255),1,ydataRangemin=yMin,ydataRangemax=yMax,xdataRangemin=xMin,xdataRangemax=xMax)
                displayFrame=OpenCVComposite(scatterFrame, displayFrame, dictSet[axis+' ds'])
    return displayFrame,np.sum(vSum[xFilter],axis=1)

def WriteMultiFrameDataToExcel(parameterStats,roiNumber,outExcelFileName):
    #dfCollected=(parameterStats[31,0,:,0]==1) & (parameterStats[12,0,:,0]!=0)
    dfCollected=(parameterStats[31,0,:,0]==1)
    dfMean=pd.DataFrame(data=parameterStats[0:12,0,dfCollected,roiNumber].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=parameterStats[31,0,dfCollected,1])
    dfStdev=pd.DataFrame(data=parameterStats[0:12,1,dfCollected,roiNumber].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=parameterStats[31,0,dfCollected,1])
    dfMost=pd.DataFrame(data=parameterStats[0:12,2,dfCollected,roiNumber].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=parameterStats[31,0,dfCollected,1])
    writer = pd.ExcelWriter(outExcelFileName, engine='xlsxwriter')
    workbook  = writer.book
    dfMean.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=9,index=False)
    dfStdev.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=22,index=False)
    dfMost.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=35,index=False)
    worksheetData = writer.sheets['FrameData']
    worksheetData.write('J1', 'Means')
    worksheetData.write('W1', 'Standard Deviations')
    worksheetData.write('AJ1', 'Most Frequent Values')
    worksheetData.write('A2', 'FrameNumber')
    worksheetData.write('B2', 'FrameRate')
    worksheetData.write('C2', 'Time')
    worksheetData.write('D2', 'Area')
    worksheetData.write('E2', 'Height')
    worksheetData.write('F2', 'Width')
    worksheetData.write('G2', 'ContourArea')
    worksheetData.write('H2', 'Mass')
    worksheetData.write_column('A3', parameterStats[30,0,dfCollected,roiNumber])
    worksheetData.write_column('B3', parameterStats[29,0,dfCollected,roiNumber])
    worksheetData.write_column('C3', parameterStats[28,0,dfCollected,roiNumber])
    worksheetData.write_column('D3', parameterStats[12,0,dfCollected,roiNumber])
    worksheetData.write_column('E3', parameterStats[13,0,dfCollected,roiNumber])
    worksheetData.write_column('F3', parameterStats[14,0,dfCollected,roiNumber])
    worksheetData.write_column('G3', parameterStats[15,0,dfCollected,roiNumber])
    worksheetData.write_column('H3', parameterStats[16,0,dfCollected,roiNumber])
    workbook.close()
    writer.save()

def WriteSingleFrameDataToExcel(frameStats,roiList,outExcelFileName):
    dfMean=pd.DataFrame(data=frameStats[0:12,0,0:len(roiList)].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"])
    dfStdev=pd.DataFrame(data=frameStats[0:12,1,0:len(roiList)].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"])
    dfMost=pd.DataFrame(data=frameStats[0:12,2,0:len(roiList)].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"])
    writer = pd.ExcelWriter(outExcelFileName, engine='xlsxwriter')
    workbook  = writer.book
    dfMean.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=9,index=False)
    dfStdev.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=22,index=False)
    dfMost.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=35,index=False)
    worksheetData = writer.sheets['FrameData']
    worksheetData.write('J1', 'Means')
    worksheetData.write('W1', 'Standard Deviations')
    worksheetData.write('AJ1', 'Most Frequent Values')
    worksheetData.write('A2', 'FrameNumber')
    worksheetData.write('B2', 'FrameRate')
    worksheetData.write('C2', 'Time')
    worksheetData.write('D2', 'Area')
    worksheetData.write('E2', 'Height')
    worksheetData.write('F2', 'Width')
    worksheetData.write('G2', 'ContourArea')
    worksheetData.write('H2', 'Mass')
    worksheetData.write_column('A3', frameStats[30,0,0:len(roiList)])
    worksheetData.write_column('B3', frameStats[29,0,0:len(roiList)])
    worksheetData.write_column('C3', frameStats[28,0,0:len(roiList)])
    worksheetData.write_column('D3', frameStats[12,0,0:len(roiList)])
    worksheetData.write_column('E3', frameStats[13,0,0:len(roiList)])
    worksheetData.write_column('F3', frameStats[14,0,0:len(roiList)])
    worksheetData.write_column('G3', frameStats[15,0,0:len(roiList)])
    worksheetData.write_column('H3', frameStats[16,0,0:len(roiList)])
    workbook.close()
    writer.save()

def OpenCVDecodeSevenSegment(massFrame,decodeFrame,dictSet):
    massFrame = cv2.GaussianBlur(massFrame,(5,5),0)
    massDisplay=np.copy(massFrame)
    rotImage,massDisplay = RegisterImageColorRectangleFlex(massFrame,massDisplay,dictSet['7F1 ll'],dictSet['7F1 ul'],dictSet['7C1 xy'],dictSet['7C2 xy'],dictSet['7C3 xy'],dictSet['7C4 xy'],dictSet['7RT or'],dictSet['7BX wh'])
    if rotImage.size<=1:
        return -1,decodeFrame
    if dictSet['flg di'][0]==1:
        cv2.imshow('massReadout', rotImage)
    decodeFrame=OpenCVComposite(massDisplay, decodeFrame, dictSet['7M1 ds'])
    rotImageDisplay=np.copy(rotImage)
    rotImage = cv2.cvtColor(rotImage, cv2.COLOR_BGR2GRAY)
    digitHeight=dictSet['7DG wh'][1] 
    digitWidth=dictSet['7DG wh'][0]
    imageStart=dictSet['7DG sn'][0]
    digits=np.zeros((digitHeight, digitWidth, 5), np.uint8)
    total=0
    for digit in range(dictSet['7DG sn'][1]): 
        digits[:,:,digit]=rotImage[dictSet['7DG mr'][0]:dictSet['7DG mr'][0]+digitHeight,imageStart+(digit*digitWidth):imageStart+((digit+1)*digitWidth)]    
        #cv2.imshow(str(digit),digits[:,:,digit])
        cv2.rectangle(rotImageDisplay,(imageStart+(digit*digitWidth),dictSet['7DG mr'][0]),(imageStart+((digit+1)*digitWidth),dictSet['7DG mr'][0]+digitHeight),(0,0,255),4 )
        decodeFrame=OpenCVComposite(rotImageDisplay, decodeFrame, dictSet['7RT ds'])
        digitImage=digits[:,:,digit]
        digitMask = cv2.inRange(digitImage, np.array([dictSet['7D1 ll']]), np.array(dictSet['7D1 ul']))
        digitMask=ip.OpenCVRotateBound(digitMask, dictSet['7DG mr'][1])
        #boundingRectangle=cv2.minAreaRect(digitMask)
        x,y,w,h = cv2.boundingRect(digitMask)
        if (w<5) or (h<5):
            decode=0
        elif h>w*3:
            digitMask=digitMask[y:y+h,x:x+w]
            decodeFrame=OpenCVComposite(digitMask, decodeFrame,[dictSet['7NS ds'][0]*digit+dictSet['7NM ds'][0],dictSet['7NM ds'][1],dictSet['7NM ds'][2]])
            decode=36
        else:
            decode=0
            digitMask=digitMask[y:y+h,x:x+w]
            decodeFrame=OpenCVComposite(digitMask, decodeFrame,[dictSet['7NS ds'][0]*digit+dictSet['7NM ds'][0],dictSet['7NM ds'][1],dictSet['7NM ds'][2]])
            dH=digitMask.shape[0]
            dW=digitMask.shape[1]
            if (dH<digitHeight*.1) or (dW<digitWidth*.1):
                decode=-1
                break
            ttSegment=digitMask[0:int(dH*0.15),int(dW*0.25):dW-int(dW*0.25)]
            ttOn = cv2.countNonZero(ttSegment)/ttSegment.size
            if ttOn>0.4:
                decode=decode+1
            #cv2.imshow("top",ttSegment)
            tlSegment=digitMask[0:int(dH*0.45),0:int(dW*0.45)]
            tlOn = cv2.countNonZero(tlSegment)/tlSegment.size
            if tlOn>0.4:
                decode=decode+2
            #cv2.imshow("top left",tlSegment)
            trSegment=digitMask[0:int(dH*0.45),dW-int(dW*0.45):dW]
            trOn = cv2.countNonZero(trSegment)/trSegment.size
            if trOn>0.4:
                decode=decode+4
            #cv2.imshow("top right",trSegment)
            ccSegment=digitMask[int(dH/2)-int(dH*0.15):int(dH/2)+int(dH*0.15),int(dW*0.25):dW-int(dW*0.25)]
            ccOn = cv2.countNonZero(ccSegment)/ccSegment.size
            if ccOn>0.4:
                decode=decode+8
            #cv2.imshow("center",ccSegment)
            blSegment=digitMask[dH-int(dH*0.45):dH,0:int(dW*0.45)]
            blOn = cv2.countNonZero(blSegment)/blSegment.size
            if blOn>0.4:
                decode=decode+16
            #cv2.imshow("bottom left",blSegment)
            brSegment=digitMask[dH-int(dH*0.45):dH,dW-int(dW*0.45):dW]
            brOn = cv2.countNonZero(brSegment)/brSegment.size
            if brOn>0.4:
                decode=decode+32
            #cv2.imshow("bottom right",brSegment)
            bbSegment=digitMask[dH-int(dH*0.15):dH,int(dW*0.1):dW-int(dW*0.1)]
            bbOn = cv2.countNonZero(bbSegment)/bbSegment.size
            if bbOn>0.4:
                decode=decode+64
            #cv2.imshow("bottom",bbSegment)
        if (decode==0) or (decode==119):
            value=0
        elif decode==36:
            value=1
        elif decode==93:
            value=2
        elif decode==109:
            value=3
        elif decode==46:
            value=4
        elif decode==107:
            value=5
        elif decode==123:
            value=6
        elif decode==37:
            value=7
        elif decode==127:
            value=8
        elif decode==111:
            value=9
        else:
            value=-1
            total=-1
            #print('digit '+str(digit)+' is '+str(bin(decode))+' value '+str(value))
            cv2.imshow("digitMask",digitMask)
            break
        #print('digit '+str(digit)+' is '+str(bin(decode))+' value '+str(value))
        total=total+10**(2-digit)*value
    total=round(total,2)
    ip.OpenCVPutText(decodeFrame,str(total),(2,decodeFrame.shape[0]-16),(255,255,255),fontScale = 0.6)
    return total,decodeFrame




if len(video_file_path)!=0:
    cap = cv2.VideoCapture(video_file_path)
    totalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    liveFlag=False
    outFileName=video_file_path+'Processed.mp4'
else:
    liveFlag=True
    outFileName=os.getcwd()+'\\Processed.avi'
    #outFileName=os.getcwd()+'\\'+time.ctime()+'_Processed.avi'
    cap = cv2.VideoCapture(int(dictSet['CAM en'][0]))
    if dictSet['CM2 en'][0]!=-1:
        cap2 = cv2.VideoCapture(int(dictSet['CM2 en'][0]))
    if dictSet['CM3 en'][0]!=-1:
        cap3 = cv2.VideoCapture(int(dictSet['CM3 en'][0]))
    #cap = cv2.VideoCapture(0)
    if dictSet['CAM en'][1]==1:
        ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,dictSet['CAM wh'][0])
        ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,dictSet['CAM wh'][1])
        ret=cap.set(cv2.CAP_PROP_BRIGHTNESS,dictSet['CAM bc'][0])
        ret=cap.set(cv2.CAP_PROP_CONTRAST,dictSet['CAM bc'][1])
        ret=cap.set(cv2.CAP_PROP_SATURATION,dictSet['CAM bc'][2])
        ret=cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,dictSet['CAM ex'][0])
        ret=cap.set(cv2.CAP_PROP_EXPOSURE,dictSet['CAM ex'][1])
        ret=cap.set(cv2.CAP_PROP_AUTOFOCUS,dictSet['CAM fo'][0])
        ret=cap.set(cv2.CAP_PROP_FOCUS,dictSet['CAM fo'][1])
        ret=cap.set(cv2.CAP_PROP_AUTO_WB,dictSet['CAM wb'][0])
        ret=cap.set(cv2.CAP_PROP_WB_TEMPERATURE,dictSet['CAM wb'][1])
    totalFrames=10000

parameterStats=np.zeros((32,6,totalFrames,60))
if totalFrames==1:
    grabbedStats=np.zeros((32,6,100,60))
else:
    grabbedStats=np.zeros((32,6,totalFrames,60))
grabCount=0
    
#ParameterStats Map
#1st dimension 0 to 14: labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]
#1st dimension 15 to 17: labels=["RO1area","RO1BoundingRecHeight","RO1BoundingRecWidth"
#1st dimension 29 to 31: labels=["frameRate","frameNumber","frameTime"]
#2nd dimension 0 to 5: labels=["mean","stdev","dominant"]
#3rd dimension frame number
#4th dimension 0 to 4: labels=["WBR","RO1","RO2","RO3"]
frameNumber=0

if totalFrames!=1:
    videoFlag=True
    if liveFlag:
        frameRate=cap.get(cv2.CAP_PROP_FPS)
    else:
        frameRate=20
else:
    videoFlag=False
    frameRate=0
    ret, originalFrame = cap.read() 

outp = cv2.VideoWriter(outFileName,fourcc, frameRate, (dictSet['dsp wh'][0], dictSet['dsp wh'][1]))
    
while frameNumber<=totalFrames:
#for frameNumber in range(totalFrames):
    if videoFlag:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameNumber)
        frameRate=cap.get(cv2.CAP_PROP_FPS)
        if (dictSet['frm av'][0]>1):
            ret, frame = cap.read() 
            frameAcc=np.zeros((frame.shape), np.uint32)
            frameAcc=frameAcc+frame
            for frameNumber in range(dictSet['frm av'][0]-1):
                ret, frame = cap.read() 
                frameAcc=frameAcc+frame
            frame=(frameAcc/dictSet['frm av'][0]).astype(np.uint8)
        else:
            ret, frame = cap.read() 
        if (dictSet['CM2 en'][0]!=-1) and (liveFlag):
            ret2, frame2 = cap2.read() 
            if ret2==False:
                break
        if (dictSet['CM3 en'][0]!=-1) and (liveFlag):
            ret3, frame3 = cap3.read() 
            if ret3==False:
                break
        if ret==False:
            break
    else:
        frame = np.copy(originalFrame)
        
    #parameterStats[31,0,frameNumber,:]=currentTime

    displayWidth=dictSet['dsp wh'][0]
    displayHeight=dictSet['dsp wh'][1]
    displayFrame = np.zeros((displayHeight, displayWidth, 3), np.uint8)

    if dictSet['PRE ds'][2]!=0:
        displayFrame=OpenCVComposite(frame, displayFrame,dictSet['PRE ds'])

    if (dictSet['CM2 ds'][2]!=0) & (dictSet['CM2 en'][0]!=-1) and (liveFlag):
        frameCrop2=frame2[dictSet['CM2 xy'][0]:dictSet['CM2 xy'][0]+dictSet['CM2 wh'][0],dictSet['CM2 xy'][1]:dictSet['CM2 xy'][1]+dictSet['CM2 wh'][1],:]
        displayFrame=OpenCVComposite(frameCrop2, displayFrame,dictSet['CM2 ds'])
        if dictSet['7SG ds'][2]!=0:
            decodeFrame = np.zeros((300, 200, 3), np.uint8)
            mass,decodeFrame=OpenCVDecodeSevenSegment(frameCrop2,decodeFrame,dictSet)
            displayFrame=OpenCVComposite(decodeFrame, displayFrame,dictSet['7SG ds'])
        else:
            mass=-1
    else:
        mass=-1
        
    if (dictSet['CM3 ds'][2]!=0) & (dictSet['CM3 en'][0]!=-1) and (liveFlag):
        frameCrop3=frame3[dictSet['CM3 xy'][0]:dictSet['CM3 xy'][0]+dictSet['CM3 wh'][0],dictSet['CM3 xy'][1]:dictSet['CM3 xy'][1]+dictSet['CM3 wh'][1],:]
        displayFrame=OpenCVComposite(frameCrop3, displayFrame,dictSet['CM3 ds'])

    roiList=[]
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)):
        if (setting[0:2]=="RO") & (setting[4:6]=="wh"):
            if (dictSet[setting][0]!=0) & (dictSet[setting][1]!=0):
                roiList.append(setting[0:3])
    wbList=[]
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)):
        if (setting[0:2]=="WB") & (setting[4:6]=="wh"):
            if (dictSet[setting][0]!=0) & (dictSet[setting][1]!=0):
                wbList.append(setting[0:3])
                
    if dictSet['flg pf'][0]!=0:
        frameStats,displayFrame,frame,frameForDrawing,rotImage,rotForDrawing = ProcessOneFrame(frame,dictSet,displayFrame,wbList=wbList,roiList=roiList)
        parameterStats[0:16,:,frameNumber,0:frameStats.shape[2]]=frameStats
        parameterStats[16,0,frameNumber,:]=mass
        if liveFlag:
            parameterStats[28,0,frameNumber,:]=time.time()
        elif videoFlag:
            parameterStats[28,0,frameNumber,:]=frameNumber/frameRate
        else:
            parameterStats[28,0,frameNumber,:]=0
        parameterStats[29,0,frameNumber,:]=frameRate
        parameterStats[30,0,frameNumber,:]=frameNumber
        parameterStats[31,0,frameNumber,:]=1

    if dictSet['flg tp'][0]!=0:
        displayFrame=MakeTimePlots(parameterStats,dictSet,displayFrame)

    if dictSet['flg fp'][0]!=0:
        roiSetName='RO1'
        if absorbanceFlag==True:
            displayFrame,signal=MakeFramePlots(dictSet,displayFrame,frame[dictSet[roiSetName+' xy'][1]:dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1], dictSet[roiSetName+' xy'][0]:dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0]],blankData,calFlag=False)
        else:
            displayFrame,signal=MakeFramePlots(dictSet,displayFrame,frame[dictSet[roiSetName+' xy'][1]:dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1], dictSet[roiSetName+' xy'][0]:dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0]])
            
    if dictSet['PST ds'][2]!=0:
        displayFrame=OpenCVComposite(frame, displayFrame,dictSet['PST ds'])

    if dictSet['REG ds'][2]!=0:
        displayFrame=OpenCVComposite(rotImage, displayFrame,dictSet['REG ds'])

    if dictSet['FMK ds'][2]!=0:
        displayFrame=OpenCVComposite(frameForDrawing, displayFrame,dictSet['FMK ds'])
        
    if dictSet['RMK ds'][2]!=0:
        displayFrame=OpenCVComposite(rotForDrawing, displayFrame,dictSet['RMK ds'])
        
    if dictSet['flg ds'][0]==1:
        settingsFrame = np.zeros((300, 300, 3), np.uint8)
        settingsFrame=DisplaySomeSettings(dictSet,60,24,settingsFrame,5,0.6)
        cv2.imshow('Settings', settingsFrame)
    elif dictSet['flg ds'][0]==2:
        settingsFrame = np.zeros((1080, 300, 3), np.uint8)
        settingsFrame=DisplayAllSettings(dictSet,20,8,settingsFrame,0.2)
        cv2.imshow('Settings', settingsFrame)
        
    ip.OpenCVPutText(displayFrame,'frame '+str(frameNumber).zfill(5)+' grabbed '+str(grabCount).zfill(5),(2,displayHeight-8),(255,255,255))
    
    cv2.imshow('Display', displayFrame)

    if (dictSet['flg rn'][0]==1) & (dictSet['flg rc'][0]==1):
        outp.write(displayFrame)

    keypress,dictSet,continueFlag,changeCameraFlag,frameJump=CheckKeys(dictSet)
    
    if keypress == ord('b'):
        if absorbanceFlag==False:
            blankData=signal
            absorbanceFlag=True
        else:
            absorbanceFlag=False
    
    if keypress == ord('g'):
        cv2.imwrite(video_file_dir+'/'+video_file_filename+'_displayFrame'+str(grabCount).zfill(3)+'.jpg', displayFrame)
        grabbedStats[:,:,grabCount,:]=parameterStats[:,:,frameNumber,:]
        grabCount=grabCount+1
        
    if continueFlag==False:
        break
    if changeCameraFlag and dictSet['CAM en'][1]==1:
        if dictSet['CAM en'][2]==1:
            cap.release()
            cap = cv2.VideoCapture(int(dictSet['CAM en'][0]))
        ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,dictSet['CAM wh'][0])
        ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,dictSet['CAM wh'][1])
        ret=cap.set(cv2.CAP_PROP_BRIGHTNESS,dictSet['CAM bc'][0])
        ret=cap.set(cv2.CAP_PROP_CONTRAST,dictSet['CAM bc'][1])
        ret=cap.set(cv2.CAP_PROP_SATURATION,dictSet['CAM bc'][2])
        ret=cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,dictSet['CAM ex'][0])
        ret=cap.set(cv2.CAP_PROP_EXPOSURE,dictSet['CAM ex'][1])
        ret=cap.set(cv2.CAP_PROP_AUTOFOCUS,dictSet['CAM fo'][0])
        ret=cap.set(cv2.CAP_PROP_FOCUS,dictSet['CAM fo'][1])
        ret=cap.set(cv2.CAP_PROP_AUTO_WB,dictSet['CAM wb'][0])
        ret=cap.set(cv2.CAP_PROP_WB_TEMPERATURE,dictSet['CAM wb'][1])
    if (frameJump!=0) & (liveFlag==False):
        if np.abs(frameJump)==1:
            frameNumber=frameNumber+frameJump
        else:
            frameNumber=frameNumber+int(totalFrames*(frameJump/100))
        if frameNumber>totalFrames:
            frameNumber=totalFrames+1
        if frameNumber<0:
            frameNumber=0
    elif liveFlag==False and videoFlag:
        if dictSet['flg rn'][0]==1:
            frameNumber=frameNumber+dictSet['set fr'][0]
    elif liveFlag==True:
        if (dictSet['flg rn'][0]==1):
            frameNumber=frameNumber+1
            
cap.release()
outp.release()
if (dictSet['CM2 en'][0]!=-1) and (liveFlag):
    cap2.release()
if (dictSet['CM3 en'][0]!=-1) and (liveFlag):
    cap3.release()
cv2.destroyAllWindows()
    
saveSettings = input("Save current settings (Y/n)?")
if (saveSettings=="Y") | (saveSettings=="y"):
    root = tk.Tk()
    root.withdraw()
    settings_file_path = asksaveasfilename(initialdir=filePathSettings,filetypes=[('settings files', '.set'),('all files', '.*')],defaultextension='.set',initialfile=video_file_filename)
    settingsFile = open(settings_file_path,'w')
    sortedDictSet = sorted(dictSet)
    outString = '{' + "\n"
    for key in sorted(dictSet.keys()) :
        concatString = "'" + key + "'" + ':' + str(dictSet[key]) + ',' + "\n"
        outString = outString + concatString
    outString = outString + '}'    
    print(outString)
    settingsFile.write(outString)
    settingsFile.close()

if videoFlag==False:
    saveSettings = input("Save frame values (Y/n)?")
    if (saveSettings=="Y") | (saveSettings=="y"):
        root = tk.Tk()
        root.withdraw()
        data_file_path = asksaveasfilename(initialdir=video_file_dir,filetypes=[('Excel files', '.xlsx'),('all files', '.*')],initialfile=video_file_filename+'frameData',defaultextension='.xlsx')
        WriteSingleFrameDataToExcel(parameterStats[:,:,frameNumber,:],roiList,data_file_path)
        #WriteMultiFrameDataToExcel(grabbedStats[:,:,0:grabCount,:],0,data_file_path)
        
if grabCount!=0:
    saveSettings = input("Save grabbed values (Y/n)?")
    if (saveSettings=="Y") | (saveSettings=="y"):
        root = tk.Tk()
        root.withdraw()
        data_file_path = asksaveasfilename(initialdir=video_file_dir,filetypes=[('Excel files', '.xlsx'),('all files', '.*')],initialfile=video_file_filename+'grabbedData' ,defaultextension='.xlsx')
        #WriteSingleFrameDataToExcel(grabbedStats[:,:,0,:],roiList,data_file_path)
        WriteMultiFrameDataToExcel(grabbedStats[:,:,0:grabCount,:],0,data_file_path)