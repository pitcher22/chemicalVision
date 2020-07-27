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
        video_file_path = askopenfilename(initialdir=os.getcwd(),filetypes=[('video files', '.mp4'),('all files', '.*')])
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
        return(False,0,False)  

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
    
def RegisterImage(frame,frameForDrawing,dictSet):
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
        rotImage = cv2.warpPerspective(frame,Mrot,(2600,900))
        return(rotImage)
    else:
        return(np.array([0]))

def WhiteBalanceFrame(rotImage,frame,dictSet,wbList=["WB1"]):
    rgbWBR=np.zeros((rotImage.shape),dtype='uint8')
    for wbRegion in wbList:
        rgbWBR[dictSet[wbRegion+' xy'][1]:dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1], dictSet[wbRegion+' xy'][0]:dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0]] = rotImage[dictSet[wbRegion+' xy'][1]:dictSet[wbRegion+' xy'][1]+dictSet[wbRegion+' wh'][1], dictSet[wbRegion+' xy'][0]:dictSet[wbRegion+' xy'][0]+dictSet[wbRegion+' wh'][0]]
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
    return(rgbWBR,rotImage,frame)

def OpenCVComposite(sourceImage, targetImage,settingsWHS):
    if settingsWHS[2]!=100:
        scaleFactor=settingsWHS[2]/100
        imageScaled = cv2.resize(sourceImage, (int(sourceImage.shape[1]*scaleFactor),int(sourceImage.shape[0]*scaleFactor)), interpolation = cv2.INTER_AREA)
    else:
        imageScaled=sourceImage
    #needs index checking here so imageScaled fits inside targetImage
    targetImage[int(targetImage.shape[0]*settingsWHS[1]/100):int((targetImage.shape[0]*settingsWHS[1]/100)+imageScaled.shape[0]),int(targetImage.shape[1]*settingsWHS[0]/100):int((targetImage.shape[1]*settingsWHS[0]/100)+imageScaled.shape[1]),:]=imageScaled
    return targetImage

def DisplayAllSettings(dictSet,parmWidth,parmHeight,displayFrame):
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
        ip.OpenCVPutText(displayFrame, setting, (int(parmWidth*0.2),parmHeight*(setRow+1)), setColor)
        if activeSettingsColumn>len(dictSet[sorted(dictSet)[activeSettingsRow]])-1:
            activeSettingsColumn=len(dictSet[sorted(dictSet)[activeSettingsRow]])-1
        for setCol in range(len(dictSet[setting])):
            if (activeSettingsColumn==setCol) & (activeSettingsRow==setRow):
                setColor=(0,0,255)
            else:
                setColor=(255,255,255)
            ip.OpenCVPutText(displayFrame,str(dictSet[setting][setCol]),(parmWidth*(setCol+2),parmHeight*(setRow+1)),setColor)
    return displayFrame

def SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=True,histogramHeight=0):
    rgbROI = rotImage[dictSet[roiSetName+' xy'][1]:dictSet[roiSetName+' xy'][1]+dictSet[roiSetName+' wh'][1], dictSet[roiSetName+' xy'][0]:dictSet[roiSetName+' xy'][0]+dictSet[roiSetName+' wh'][0]]
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
    frameStats=np.zeros((16,6,len(roiList)))    
    img=np.copy(frame)
    if dictSet['flg rf'][0]:
        rotImage = RegisterImage(frame,img,dictSet)
        skipFrame=False
        if rotImage.size==1:
            skipFrame=True
            rotImage = np.copy(frame)
    else:
        rotImage = np.copy(frame)
        skipFrame=False
    if skipFrame==False:
        if dictSet['flg wb'][0]==1:
            rgbWBR,rotImage,frame=WhiteBalanceFrame(rotImage,frame,dictSet,wbList=wbList)
            if dictSet['flg di'][0]==1:
                cv2.imshow("WBR",rgbWBR)
            if dictSet['WBR ds'][2]!=0:
                displayFrame=OpenCVComposite(rgbWBR, displayFrame, dictSet['WBR ds'])
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
    return frameStats,displayFrame,frame,rotImage

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
            hLimit=4000
            lLimit=0
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
        if ((keypress==upArrow) | (keypress==ord('w'))) & (row>0):
            dictSet['set rc'][0]=row-1
        if ((keypress==dnArrow) | (keypress==ord('s'))) & (row<len(dictSet)-1):
            dictSet['set rc'][0]=row+1
        if ((keypress==ltArrow) | (keypress==ord('a'))) & (col>0):
            dictSet['set rc'][1]=col-1
        if ((keypress==rtArrow) | (keypress==ord('d'))) & (col<len(dictSet[sorted(dictSet)[row]])-1):
            dictSet['set rc'][1]=col+1
    return(dictSet,continueFlag,changeCameraFlag,frameJump)

def MakePlots(parameterStats,dictSet,displayFrame,frameStart=0,frameEnd=0):
    pltList=[]
    for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)):
        if (setting[0:2]=="fg") & (setting[4:6]=="wh"):
            if (dictSet[setting][0]!=0) & (dictSet[setting][1]!=0):
                pltList.append(setting[0:3])
    if frameEnd==0:
        frameEnd=parameterStats.shape[2]
    for axis in pltList:
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
    
cap = cv2.VideoCapture(video_file_path)
totalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
parameterStats=np.zeros((32,6,totalFrames,6))
#ParameterStats Map
#1st dimension 0 to 14: labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]
#1st dimension 15 to 17: labels=["RO1area","RO1BoundingRecHeight","RO1BoundingRecWidth"
#1st dimension 29 to 31: labels=["frameRate","frameNumber","frameTime"]
#2nd dimension 0 to 5: labels=["mean","stdev","dominant"]
#3rd dimension frame number
#4th dimension 0 to 4: labels=["WBR","RO1","RO2","RO3"]
frameNumber=0

if totalFrames>1:
    videoFlag=True
else:
    videoFlag=False
    frameRate=0
    ret, originalFrame = cap.read() 
    
while frameNumber<=totalFrames:
#for frameNumber in range(totalFrames):
    if videoFlag:
        cap.set(cv2.CAP_PROP_POS_FRAMES,frameNumber)
        frameRate=cap.get(cv2.CAP_PROP_FPS)
        ret, frame = cap.read() 
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
                
    frameStats,displayFrame,frame,rotImage = ProcessOneFrame(frame,dictSet,displayFrame,wbList=wbList,roiList=roiList)

    parameterStats[0:16,:,frameNumber,0:frameStats.shape[2]]=frameStats
    parameterStats[31,0,frameNumber,:]=1
    parameterStats[30,0,frameNumber,:]=frameNumber
    parameterStats[29,0,frameNumber,:]=frameRate
    if frameRate!=0:
        parameterStats[28,0,frameNumber,:]=frameNumber/frameRate
    else:
        parameterStats[28,0,frameNumber,:]=0
    
    displayFrame=MakePlots(parameterStats,dictSet,displayFrame)
    
    if dictSet['PST ds'][2]!=0:
        displayFrame=OpenCVComposite(frame, displayFrame,dictSet['PST ds'])

    if dictSet['REG ds'][2]!=0:
        displayFrame=OpenCVComposite(rotImage, displayFrame,dictSet['REG ds'])
        
    if dictSet['flg ds'][0]==1:
        settingsFrame = np.zeros((820, 150, 3), np.uint8)
        settingsFrame=DisplayAllSettings(dictSet,30,12,settingsFrame)
        cv2.imshow('Settings', settingsFrame)
        
    ip.OpenCVPutText(displayFrame,str(frameNumber),(2,displayHeight-8),(255,255,255))
    
    cv2.imshow('Display', displayFrame)
    dictSet,continueFlag,changeCameraFlag,frameJump=CheckKeys(dictSet)
    if continueFlag==False:
        break
    if changeCameraFlag:
        ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,dictSet['CAM wh'][0])
        ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,dictSet['CAM wh'][1])
        #ret=cap.set(cv2.CAP_PROP_BRIGHTNESS,dictSet['CAM bc'][0]/255.0)
        #ret=cap.set(cv2.CAP_PROP_CONTRAST,dictSet['CAM bc'][1]/255.0)
        #ret=cap.set(cv2.CAP_PROP_SATURATION,dictSet['CAM bc'][2]/255.0)
        #ret=cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,dictSet['CAM ex'][0]/2.0+0.25)
        #ret=cap.set(cv2.CAP_PROP_EXPOSURE,dictSet['CAM ex'][1]/255.0)
        #ret=cap.set(cv2.CAP_PROP_AUTOFOCUS,dictSet['CAM fo'][0])
        #ret=cap.set(cv2.CAP_PROP_FOCUS,dictSet['CAM fo'][1]/255.0)
    if frameJump!=0:
        if np.abs(frameJump)==1:
            frameNumber=frameNumber+frameJump
        else:
            frameNumber=frameNumber+int(totalFrames*(frameJump/100))
        if frameNumber>totalFrames:
            frameNumber=totalFrames+1
        if frameNumber<0:
            frameNumber=0
    else:
        if dictSet['flg rn'][0]==1:
            frameNumber=frameNumber+dictSet['set fr'][0]
            
cap.release()
cv2.destroyAllWindows()

saveSettings = input("Save current settings (Y/n)?")
if (saveSettings=="Y") | (saveSettings=="y"):
    root = tk.Tk()
    root.withdraw()
    settings_file_path = asksaveasfilename(initialdir=filePathSettings,filetypes=[('settings files', '.set'),('all files', '.*')])
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