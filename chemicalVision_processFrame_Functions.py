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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 19 17:15:17 2020

@author: kevin
"""
def FindLargestContour(mask):
    if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
        contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    else:
        image,contours,hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)   
    if len(contours)>=1:
        maxArea=0
        contourIndex=0
        largestContour=0
        for contour in contours:
            area=cv2.contourArea(contour)
            if area>maxArea:
                maxArea=area
                largestContour=contourIndex
            contourIndex=contourIndex+1
        largestContour=contours[largestContour]
        return(largestContour,maxArea)
    else:
        return(False,0)  

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

    outerBoxContour,boxArea=FindLargestContour(boxMask)
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
        #rotImage = cv2.warpPerspective(frame,Mrot,(1800,1200))
        rotImage = cv2.warpPerspective(frame,Mrot,(2600,900))
        return(rotImage)
    else:
        return(False)
    
    

def ProcessOneFrame(frame,dictSet):    
    displayWidth=dictSet['dsp wh'][0]
    displayHeight=dictSet['dsp wh'][1]
    flgFindReference=dictSet['flg rf'][0]
    displayFrame = np.zeros((displayHeight, displayWidth, 3), np.uint8)
    img=np.copy(frame)
    
    if flgFindReference:
        rotImage = RegisterImage(frame,img,dictSet)
        if rotImage==False:
            skipFrame=True
    else:
        rotImage = np.copy(frame)
        skipFrame=False
   
    if skipFrame==False:
        # Working here: Idea is to put multiple rectangular regions into rgbWBR
        rgbWBR=np.zeros((rotImage.shape),dtype='uint8')
        rgbWBR[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]] = rotImage[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]]
        rgbWBR[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]] = rotImage[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]]
        rgbWBR[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]] = rotImage[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]]
        hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
        maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
        
        if np.sum(maskWBR)>0:
            RGBGreyWBR=cv2.mean(rgbWBR, mask=maskWBR)
            bscale=RGBGreyWBR[0]
            gscale=RGBGreyWBR[1]
            rscale=RGBGreyWBR[2]
            scalemax=max(rscale,gscale,bscale)
            if dictSet['WBR sc'][0]!=0:
                scalemin=dictSet['WBR sc'][0]
            else:
                scalemin=min(rscale,gscale,bscale)
            if (scalemin!=0) & (min(rscale,gscale,bscale)!=0):
                rfactor=float(scalemin)/float(rscale)
                gfactor=float(scalemin)/float(gscale)
                bfactor=float(scalemin)/float(bscale)
            rotImage=ip.OpenCVRebalanceImage(rotImage,rfactor,gfactor,bfactor)
            frame=ip.OpenCVRebalanceImage(frame,rfactor,gfactor,bfactor)

        rgbWBR[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]] = rotImage[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]]
        rgbWBR[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]] = rotImage[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]]
        rgbWBR[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]] = rotImage[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]]
        hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
        maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
        for row, displayColor, channel in zip([0,1,2], [(0, 0, 128),(0, 128, 0),(128, 25, 25)], [2,1,0]):              
            #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskRO1,256,0,255,displayWidth/2,5+(row*displayHeight/10)+(row*5),256,(displayHeight/12),displayFrame,displayColor,5,True,label)
            ParameterStats[row,0,frameNumber,0],ParameterStats[row,1,frameNumber,0],ParameterStats[row,2,frameNumber,0]=ip.OpenCVDisplayedHistogram(rgbWBR,channel,maskWBR,256,0,255,displayWidth/2,5+(row*displayHeight/14)+(row*6),256,(displayHeight/16),displayFrame,displayColor,5,False)
        rgbRO2 = rotImage[dictSet['RO2 xy'][1]:dictSet['RO2 xy'][1]+dictSet['RO2 wh'][1], dictSet['RO2 xy'][0]:dictSet['RO2 xy'][0]+dictSet['RO2 wh'][0]]
        rgbRO3 = rotImage[dictSet['RO3 xy'][1]:dictSet['RO3 xy'][1]+dictSet['RO3 wh'][1], dictSet['RO3 xy'][0]:dictSet['RO3 xy'][0]+dictSet['RO3 wh'][0]]
        rgbRO1 = rotImage[dictSet['RO1 xy'][1]:dictSet['RO1 xy'][1]+dictSet['RO1 wh'][1], dictSet['RO1 xy'][0]:dictSet['RO1 xy'][0]+dictSet['RO1 wh'][0]]
        rgbRO2summary=cv2.meanStdDev(rgbRO2)
        rgbRO3summary=cv2.meanStdDev(rgbRO3)

        hsvRO1 = cv2.cvtColor(rgbRO1, cv2.COLOR_BGR2HSV)

        hsvRO1[:,:,0]=ip.ShiftHOriginToValue(hsvRO1[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])

        labRO1 = cv2.cvtColor(rgbRO1, cv2.COLOR_BGR2LAB)
        logsrgbRO1=cv2.LUT(rgbRO1, linLUTabs)*64
        maskRO1 = cv2.inRange(hsvRO1, np.array(dictSet['RO1 ll']), np.array(dictSet['RO1 ul']))
        #maskRO2 = cv2.inRange(hsvRO2, np.array(dictSet['RO2 ll']), np.array(dictSet['RO2 ul']))
        #maskRO3 = cv2.inRange(hsvRO3, np.array(dictSet['RO3 ll']), np.array(dictSet['RO3 ul']))
        resFrameWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)

        if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
            contours,hierarchy = cv2.findContours(maskRO1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else:
            image,contours,hierarchy = cv2.findContours(maskRO1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contourMask=np.zeros((maskRO1.shape),dtype='uint8')
        if len(contours)>=1:
            MaxRO1Area=0
            ContourIndex=0
            LargestContour=0
            for contour in contours:
                area=cv2.contourArea(contour)
                if area>MaxRO1Area:
                    MaxRO1Area=area
                    LargestContour=ContourIndex
                ContourIndex=ContourIndex+1
            outerRO1Contour=contours[LargestContour]
            boundingRectangle=cv2.minAreaRect(outerRO1Contour)
            #cv2.drawContours(resFrameRO1,[outerRO1Contour],0,(0,255,0),2)
            cv2.drawContours(contourMask,[outerRO1Contour],0,(255),-1)
            resMask = cv2.bitwise_and(maskRO1,maskRO1, mask= contourMask)
            resFrameRO1 = cv2.bitwise_and(rgbRO1,rgbRO1, mask= resMask)
            ParameterStats[15,0,frameNumber,1]=MaxRO1Area*cmPerPixel*cmPerPixel
            ParameterStats[16,0,frameNumber,1]=boundingRectangle[1][0]
            ParameterStats[17,0,frameNumber,1]=boundingRectangle[1][1]
            ParameterStats[0,0,frameNumber,2]=rgbRO2summary[0][2]
            ParameterStats[1,0,frameNumber,2]=rgbRO2summary[0][1]
            ParameterStats[2,0,frameNumber,2]=rgbRO2summary[0][0]
            ParameterStats[0,1,frameNumber,2]=rgbRO2summary[0][2]
            ParameterStats[1,1,frameNumber,2]=rgbRO2summary[0][1]
            ParameterStats[2,1,frameNumber,2]=rgbRO2summary[0][0]
            ParameterStats[0,0,frameNumber,3]=rgbRO3summary[0][2]
            ParameterStats[1,0,frameNumber,3]=rgbRO3summary[0][1]
            ParameterStats[2,0,frameNumber,3]=rgbRO3summary[0][0]
            ParameterStats[0,1,frameNumber,3]=rgbRO3summary[0][2]
            ParameterStats[1,1,frameNumber,3]=rgbRO3summary[0][1]
            ParameterStats[2,1,frameNumber,3]=rgbRO3summary[0][0]
            ParameterStats[31,0,frameNumber,:]=currentTime
            ParameterStats[30,0,frameNumber,:]=currentFrame
            ParameterStats[29,0,frameNumber,:]=frameRate
            #Analytical signal should be in channel 13 of parameter stats B absorbance -R absorbance for I2
            labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]
            ParameterStats[12,0,frameNumber,:]=ParameterStats[10,0,frameNumber,:]-ParameterStats[9,0,frameNumber,:]
            ParameterStats[13,0,frameNumber,:]=ParameterStats[11,0,frameNumber,:]-ParameterStats[9,0,frameNumber,:]
            ParameterStats[14,0,frameNumber,:]=ParameterStats[10,0,frameNumber,:]-ParameterStats[11,0,frameNumber,:]
            
            RO1Scale=max(resFrameRO1.shape[1]/(displayWidth/4),resFrameRO1.shape[0]/(displayHeight/4))
            RO1ImageScale = cv2.resize(resFrameRO1, (int(resFrameRO1.shape[1]/RO1Scale),int(resFrameRO1.shape[0]/RO1Scale)), interpolation = cv2.INTER_AREA)
            WBRScale=max(resFrameWBR.shape[1]/(displayWidth/4),resFrameWBR.shape[0]/(displayHeight/4))
            WBRImageScale = cv2.resize(resFrameWBR, (int(resFrameWBR.shape[1]/WBRScale),int(resFrameWBR.shape[0]/WBRScale)), interpolation = cv2.INTER_AREA)
            resFrameRO2=rgbRO2
            RO2Scale=max(resFrameRO2.shape[1]/(displayWidth/4),resFrameRO2.shape[0]/(displayHeight/4))
            RO2ImageScale = cv2.resize(resFrameRO2, (int(resFrameRO2.shape[1]/RO2Scale),int(resFrameRO2.shape[0]/RO2Scale)), interpolation = cv2.INTER_AREA)
            resFrameRO3=rotImage
            RO3Scale=max(resFrameRO3.shape[1]/(displayWidth/4),resFrameRO3.shape[0]/(displayHeight/4))
            RO3ImageScale = cv2.resize(resFrameRO3, (int(resFrameRO3.shape[1]/RO3Scale),int(resFrameRO3.shape[0]/RO3Scale)), interpolation = cv2.INTER_AREA)
            displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+RO1ImageScale.shape[0]),0:RO1ImageScale.shape[1],:]=RO1ImageScale
            displayFrame[int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4):int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4)+RO2ImageScale.shape[0],0:RO2ImageScale.shape[1],:]=RO2ImageScale
            displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+WBRImageScale.shape[0]) , int(displayFrame.shape[0]/2):int(displayFrame.shape[0]/2)+WBRImageScale.shape[1],:]=WBRImageScale
            displayFrame[int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4):int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4)+RO3ImageScale.shape[0] , int(displayFrame.shape[0]/2):int(displayFrame.shape[0]/2)+RO3ImageScale.shape[1],:]=RO3ImageScale
            inputImages= [rgbRO1,rgbRO1,rgbRO1,hsvRO1,hsvRO1,hsvRO1,labRO1,labRO1,labRO1,logsrgbRO1,logsrgbRO1,logsrgbRO1]
            for row, displayColor, inputImage, channel, label in zip(rows, displayColors, inputImages, channels,labels):              
                #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskRO1,256,0,255,displayWidth/2,5+(row*displayHeight/10)+(row*5),256,(displayHeight/12),displayFrame,displayColor,5,True,label)
                ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=ip.OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,displayWidth/2,5+(row*displayHeight/14)+(row*6),256,(displayHeight/16),displayFrame,displayColor,5,True,label)
            ParameterStats[12,0,frameNumber,1]=ParameterStats[10,0,frameNumber,1]-ParameterStats[9,0,frameNumber,1]
            ParameterStats[13,0,frameNumber,1]=ParameterStats[11,0,frameNumber,1]-ParameterStats[9,0,frameNumber,1]
            ParameterStats[14,0,frameNumber,1]=ParameterStats[10,0,frameNumber,1]-ParameterStats[11,0,frameNumber,1]
            maskRO1Volume = cv2.inRange(hsvRO1, np.array([int(ParameterStats[3,0,frameNumber,1]-ParameterStats[3,1,frameNumber,1]),int(ParameterStats[4,0,frameNumber,1]-ParameterStats[4,1,frameNumber,1]),int(ParameterStats[5,0,frameNumber,1]-ParameterStats[5,1,frameNumber,1])]), np.array([255,255,255]))
            if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
                contours,hierarchy = cv2.findContours(maskRO1Volume,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            else:
                image,contours,hierarchy = cv2.findContours(maskRO1Volume,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contourMask=np.zeros((maskRO1Volume.shape),dtype='uint8')
            if len(contours)>=1:
                MaxRO1AreaVolume=0
                ContourIndex=0
                LargestContour=0
                for contour in contours:
                    area=cv2.contourArea(contour)
                    if area>MaxRO1AreaVolume:
                        MaxRO1AreaVolume=area
                        LargestContour=ContourIndex
                    ContourIndex=ContourIndex+1
                outerRO1ContourVolume=contours[LargestContour]
            ParameterStats[18,0,frameNumber,1]=MaxRO1AreaVolume*cmPerPixel*cmPerPixel
            if frameNumber>=2:
                if dictSet['a1x sc'][0]==0:
                    xMin=dictSet['a1x sc'][1]
                    xMax=dictSet['a1x sc'][2]
                else:
                    xMin=None
                    xMax=None          
                if dictSet['a1y sc'][0]==0:
                    yMin=dictSet['a1y sc'][1]
                    yMax=dictSet['a1y sc'][2]
                else:
                    yMin=None
                    yMax=None     
                ip.OpenCVDisplayedScatter(displayFrame, ParameterStats[dictSet['a1x ch'][0],dictSet['a1x ch'][1],0:frameNumber,dictSet['a1x ch'][2]],ParameterStats[dictSet['a1y ch'][0],dictSet['a1y ch'][1],0:frameNumber,dictSet['a1y ch'][2]],dictSet['pl1 xy'][0],dictSet['pl1 xy'][1],dictSet['pl1 wh'][0],dictSet['pl1 wh'][1],(255,255,255), 1, ydataRangemin=yMin, ydataRangemax=yMax,xdataRangemin=xMin, xdataRangemax=xMax)
                if dictSet['a2x sc'][0]==0:
                    xMin=dictSet['a2x sc'][1]
                    xMax=dictSet['a2x sc'][2]
                else:
                    xMin=None
                    xMax=None          
                if dictSet['a2y sc'][0]==0:
                    yMin=dictSet['a2y sc'][1]
                    yMax=dictSet['a2y sc'][2]
                else:
                    yMin=None
                    yMax=None     
                ip.OpenCVDisplayedScatter(displayFrame, ParameterStats[dictSet['a2x ch'][0],dictSet['a2x ch'][1],0:frameNumber,dictSet['a2x ch'][2]],ParameterStats[dictSet['a2y ch'][0],dictSet['a2y ch'][1],0:frameNumber,dictSet['a2y ch'][2]],dictSet['pl2 xy'][0],dictSet['pl2 xy'][1],dictSet['pl2 wh'][0],dictSet['pl2 wh'][1],(255,255,255), 1, ydataRangemin=yMin, ydataRangemax=yMax,xdataRangemin=xMin, xdataRangemax=xMax)
            if ActiveState=="Process":
                frameNumber=frameNumber+1
    
    #single frame process ends here