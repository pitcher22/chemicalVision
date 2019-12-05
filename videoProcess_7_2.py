#!/usr/bin/env python2
# -*- coding: utf-8 -*-


#To do list:
#make script work with live video
#enable recording of: live, and diagnostic windows
    # - have a record flag that can record a video file that 
    # is the unadulturated unprocessed video input stream
    # - 
      
#all values set in separate file (displayColors)
    # - look through code for hard coded numbers (maybe a few)
    # - try to move into dictionary if you find hardcoded settings
#summary of hotkeys on display
    # - p pauses, j jumps backwards in time, etc.
    # - add this legend to the bottom of the page 
    
#test on Windows, Mac, and Linux
#test with Python 2.7 and 3.X
#clear map of the values in ParameterStats
    # - write a comprehensible map for what each 
    #   element is in the array ParameterStats
#clean up currentTime and timelaspeflags
    
#make OpenCVDisplayedScatter take an alpha value

#make sure settings in the dictionary are grouped in a rational way: Sarah

# Just have the settings that are highlighted show up in the top right 
# toggle on which setting is in top right to save real estate
# If you want to see all settings press i and get popout menu
# similar to the regions of interest and white balance (red button, etc.)

"""
Created on Tue Sep  4 13:00:43 2018

@author: sensor
"""
from __future__ import division
from __future__ import print_function

import time
import cv2
import numpy as np
import os
import datetime
#import time
try:
    import Tkinter as tk
    from tkFileDialog import askopenfilename
    from tkFileDialog import asksaveasfilename
except ImportError:
    import tkinter as tk
    from tkinter.filedialog import askopenfilename
    from tkinter.filedialog import asksaveasfilename

try:
    input=raw_input
except NameError:
    pass

import matplotlib.pyplot as plt
import pandas as pd
#import struct
try:
    import zbar
    decodedQR=False
except ImportError:
    print("QR not yet supported")
    decodedQR=True
    
from PIL import Image

import sys
if sys.platform=='win32':
    upArrow=38
    dnArrow=40
    ltArrow=37
    rtArrow=39
else:
    upArrow=82
    dnArrow=84
    ltArrow=81
    rtArrow=83

maskDiagnostic=False    
referenceFlag=True
settingsFlag=True    
font = cv2.FONT_HERSHEY_SIMPLEX
fourcc = cv2.VideoWriter_fourcc(*'XVID')
RecordFlag=True
overlayFlag=True
displayHelp=True
cmPerPixel=2.54/300
cellDepth=2
cellWidth=5.5
dropVolume=0.034

ActiveState="Process"
try:
    print(StoredSettingsFlag)
    if StoredSettingsFlag:
        dictSet=savedSet
except:
    print("Not Defined")
    StoredSettingsFlag=False
if StoredSettingsFlag==False:
    settingString='''
{'CAM bcs':[128,128,128],
 'CAM exp':[1,50],
 'CAM foc':[1,60],
 'ROI ll':[0,0,0],
 'ROI ul':[255,255,255],
 'ROI wh':[850,400],
 'ROI xy':[110,400],
 'RO2 wh':[40,40],
 'RO2 xy':[740,540],
 'RO3 wh': [42, 37],
 'RO3 xy': [849, 553],
 'WBR sc':[0],
 'WBR ll':[0,0,0],
 'WBR ul':[255,50,255],
 'WBR wh':[270,240],
 'WBR xy':[640,530],
 'box ll':[80,20,40],
 'box ul':[100,255,255],
 'cl1 xy':[600,225],
 'cl2 xy':[1500,225],
 'cl3 xy':[600,975],
 'cl4 xy':[1500,975],
 'c12 ll':[140,20,40],
 'c12 ul':[160,255,255],
 'c34 ll':[20,60,40],
 'c34 ul':[40,255,255],
 'hue lo':[180.0,150.0],
 'dsp wh':[1700,900],
 'xa1 ch':[30,0,1],
 'ya1 ch':[3,0,1],
 'xa1 sc':[1,0,0],
 'ya1 sc':[1,0,0],
 'pl1 xy':[1160,10],
 'pl1 wh':[220,220],
 'xa2 ch':[30,0,1],
 'ya2 ch':[3,0,1],
 'xa2 sc':[1,0,0],
 'ya2 sc':[1,0,0],
 'pl2 xy':[1160,250],
 'pl2 wh':[220,220]}
'''
    upperLimitString='''
{'CAM bcs':[255, 255, 255],
 'CAM exp':[  1, 255],
 'CAM foc':[ 1, 255],
 'ROI ll':[ 255, 255,  255],
 'ROI ul':[255, 255, 255],
 'ROI wh':[1800, 1200],
 'ROI xy':[1800, 1200],
 'RO2 wh':[1800, 1200],
 'RO2 xy':[1800, 1200],
 'RO3 wh':[1800, 1200],
 'RO3 xy':[1800, 1200],
 'WBR sc':[255],
 'WBR ll':[255, 255, 255],
 'WBR ul':[255, 255, 255],
 'WBR wh':[1800, 1200],
 'WBR xy':[1800, 1200],
 'box ll':[255, 255, 255],
 'box ul':[255, 255, 255],
 'cl1 xy':[1800,1800],
 'cl2 xy':[1800,1800],
 'cl3 xy':[1800,1800],
 'cl4 xy':[1800,1800],
 'c12 ll':[255, 255, 255],
 'c12 ul':[255, 255, 255],
 'c34 ll':[255, 255, 255],
 'c34 ul':[255, 255, 255],
 'hue lo':[180,180],
 'dsp wh':[2000,2000],
 'xa1 ch':[32,2,3],
 'ya1 ch':[32,2,3],
 'xa1 sc':[1,5000000,5000000],
 'ya1 sc':[1,5000000,5000000], 
 'pl1 xy':[2000,2000],
 'pl1 wh':[2000,2000],
 'xa2 ch':[32,2,3],
 'ya2 ch':[32,2,3],
 'xa2 sc':[1,5000000,5000000],
 'ya2 sc':[1,5000000,5000000], 
 'pl2 xy':[2000,2000],
 'pl2 wh':[2000,2000]}
'''
    dictSet=eval(settingString)
    dictUL=eval(upperLimitString)
    


def hyst(x, th_lo, th_hi, initial = False):
    # http://stackoverflow.com/questions/23289976/how-to-find-zero-crossings-with-hysteresis
    hi = x >= th_hi
    lo_or_hi = (x <= th_lo) | hi
    ind = np.nonzero(lo_or_hi)[0]
    if not ind.size: # prevent index error if ind is empty
        return np.zeros_like(x, dtype=bool) | initial
    cnt = np.cumsum(lo_or_hi) # from 0 to len(x)
    return np.where(cnt, hi[ind[cnt-1]], initial)
 
def crossBoolean(x, y, crossPoint=0, direction='cross'):
    """
    Given a Series returns all the index values where the data values equal 
    the 'cross' value. 
 
    Direction can be 'rising' (for rising edge), 'falling' (for only falling 
    edge), or 'cross' for both edges
    """
    # Find if values are above or bellow yvalue crossing:
    above=y > crossPoint
    below=np.logical_not(above)
    left_shifted_above = above[1:]
    left_shifted_below = below[1:]
    x_crossings = []
    # Find indexes on left side of crossing point
    if direction == 'rising':
        idxs = (left_shifted_above & below[0:-1]).nonzero()[0]
    elif direction == 'falling':
        idxs = (left_shifted_below & above[0:-1]).nonzero()[0]
    else:
        rising = left_shifted_above & below[0:-1]
        falling = left_shifted_below & above[0:-1]
        idxs = (rising | falling).nonzero()[0]
 
    # Calculate x crossings with interpolation using formula for a line:
    x1 = x[idxs]
    x2 = x[idxs+1]
    y1 = y[idxs]
    y2 = y[idxs+1]
    x_crossings = (crossPoint-y1)*(x2-x1)/(y2^y1) + x1
 
    return x_crossings,idxs

def PolyReg(X,Y,order):
    """
    Perform a least squares polynomial fit
    
    Parameters
    ----------
        X: a numpy array with shape M
            the independent variable 
        Y: a numpy array with shape M
            the dependent variable
        order: integer
            the degree of the fitting polynomial
    
    Returns
    -------
    a dict with the following keys:
        'coefs': a numpy array with length order+1 
            the coefficients of the fitting polynomial, higest order term first
        'errors': a numpy array with length order+1
            the standard errors of the calculated coefficients, 
            only returned if (M-order)>2
        'sy': float
            the standard error of the fit
        'n': integer
            number of data points (M)
        'poly':  class in numpy.lib.polynomial module
            a polynomial with coefficients (coefs) and degreee (order),
            see example below
        'res': a numpy array with length M
            the residuals of the fit
    
    Examples
    --------
    >>> x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
    >>> y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
    >>> fit = PolyReg(x, y, 2)
    >>> fit
    {'coefs': array([-0.16071429,  0.50071429,  0.22142857]),
     'errors': array([0.06882765, 0.35852091, 0.38115025]),
     'n': 6,
     'poly': poly1d([-0.16071429,  0.50071429,  0.22142857]),
     'res': array([-0.22142857,  0.23857143,  0.32      , -0.17714286, -0.45285714,
         0.29285714]),
     'sy': 0.4205438655564278}
    
    It is convenient to use the "poly" key for dealing with fit polynomials:
    
    >>> fit['poly'](0.5)
    0.43160714285714374
    >>> fit['poly'](10)
    -10.842857142857126
    >>> fit['poly'](np.linspace(0,10,11))
    array([  0.22142857,   0.56142857,   0.58      ,   0.27714286,
        -0.34714286,  -1.29285714,  -2.56      ,  -4.14857143,
        -6.05857143,  -8.29      , -10.84285714])
    """
    n=len(X)
    if X.shape!=Y.shape:
        raise Exception('The shape of X and Y should be the same')
    df=n-(order+1)
    if df<0:
        raise Exception('The number of data points is too small for that many coefficients')
    #if df = 0, 1, or 2 we call numpy's polyfit function without calculating the covariance matrix
    elif df<(3):
        coefs=np.polyfit(X,Y,order)
        p=np.poly1d(coefs)
        yFit=p(X)
        res=Y-yFit
        sy=np.sqrt( np.sum(res**2) / df )
        if order==1:
            #if the fit is linear we can explicitly calculate the standard errors of the slope and intercept
            #http://www.chem.utoronto.ca/coursenotes/analsci/stats/ErrRegr.html
            stdErrors=np.zeros((2))
            xVar=np.sum((X-np.mean(X))**2)
            sm=sy/np.sqrt(xVar)
            sb=np.sqrt(np.sum(X**2)/(n*xVar))*sy
            stdErrors[0]=sm
            stdErrors[1]=sb            
        else:
            stdErrors=np.full((order+1),np.inf)
    else:
        #The diagonal of the covariance matrix is the square of the standard error for each coefficent
        #NOTE 1: The polyfit function conservatively scales the covariance matrix. Dividing by (n-# coefs-2) rather than (n-# coefs)
        #NOTE 2: Because of this scaling factor, you can get division by zero in the covariance matrix when (# coefs-n)<2
        coefs,cov=np.polyfit(X,Y,order,cov=True)
        p=np.poly1d(coefs)
        yFit=p(X)
        res=Y-yFit
        sy=np.sqrt( np.sum(res**2) / df )
        stdErrors=np.sqrt(np.diagonal(cov)*(df-2)/df)
    return {'coefs':coefs,'errors':stdErrors,'sy':sy,'n':n,'poly':p,'res':res}

def FormatSciUsingError(x,e,withError=False,extraDigit=0):
    """
    Format the value, x, as a string using scientific notation and rounding appropriately based on the absolute error, e
    
    Parameters
    ----------
        x: number
            the value to be formatted 
        e: number
            the absolute error of the value
        withError: bool, optional
            When False (the default) returns a string with only the value. When True returns a string containing the value and the error
        extraDigit: int, optional
            number of extra digits to return in both value and error
    
    Returns
    -------
    a string
    
    Examples
    --------
    >>> FormatSciUsingError(3.141592653589793,0.02718281828459045)
    '3.14E+00'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045)
    '3.142E+00'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045,withError=True)
    '3.142E+00 (+/- 3E-03)'
    >>> FormatSciUsingError(3.141592653589793,0.002718281828459045,withError=True,extraDigit=1)
    '3.1416E+00 (+/- 2.7E-03)'
    >>> FormatSciUsingError(123456,123,withError=True)
    '1.235E+05 (+/- 1E+02)'
    """
    if abs(x)>=e:
        NonZeroErrorX=np.floor(np.log10(abs(e)))
        NonZeroX=np.floor(np.log10(abs(x)))
        formatCodeX="{0:."+str(int(NonZeroX-NonZeroErrorX+extraDigit))+"E}"
        formatCodeE="{0:."+str(extraDigit)+"E}"
    else:
        formatCodeX="{0:."+str(extraDigit)+"E}"
        formatCodeE="{0:."+str(extraDigit)+"E}"
    if withError==True:
        return formatCodeX.format(x)+" (+/- "+formatCodeE.format(e)+")"
    else:
        return formatCodeX.format(x)

def AnnotateFit(fit,axisHandle,annotationText='Eq',color='black',arrow=False,xArrow=0,yArrow=0,xText=0.5,yText=0.2,boxColor='0.9'):
    """
    Annotate a figure with information about a PolyReg() fit
    
    see https://matplotlib.org/api/_as_gen/matplotlib.pyplot.annotate.html
    https://matplotlib.org/examples/pylab_examples/annotation_demo3.html
    
    Parameters
    ----------
        fit: dict, returned by the function PolyReg(X,Y,order)
            the fit to be summarized in the figure annotation 
        axisHandle: a matplotlib axes class
            the axis handle to the figure to be annotated
        annotationText: string, optional
            When "Eq" (the default) displays a formatted polynomial with the coefficients (rounded according to their error) in the fit. When "Box" displays a formatted box with the coefficients and their error terms.  When any other string displays a text box with that string.
        color: a valid color specification in matplotlib, optional
            The color of the box outline and connecting arrow.  Default is black. See https://matplotlib.org/users/colors.html
        arrow: bool, optional
            If True (default=False) draws a connecting arrow from the annotation to a point on the graph.
        xArrow: float, optional 
            The X coordinate of the arrow head using units of the figure's X-axis data. If unspecified or 0 (and arrow=True), defaults to the center of the X-axis.
        yArrow: float, optional 
            The Y coordinate of the arrow head using units of the figure's Y-axis data. If unspecified or 0 (and arrow=True), defaults to the calculated Y-value at the center of the X-axis.
        xText: float, optional 
            The X coordinate of the annotation text using the fraction of the X-axis (0=left,1=right). If unspecified, defults to the center of the X-axis.
        yText: float, optional 
            The Y coordinate of the annotation text using the fraction of the Y-axis (0=bottom,1=top). If unspecified, defults to 20% above the bottom.
    
    Returns
    -------
    a dragable matplotlib Annotation class
    
    Examples
    --------
    >>> annLinear=AnnotateFit(fitLinear,ax)
    >>> annLinear.remove()
    """
    c=fit['coefs']
    e=fit['errors']
    t=len(c)
    if annotationText=='Eq':
        annotationText="y = "
        for order in range(t):
            exponent=t-order-1
            if exponent>=2:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])+"x$^{}$".format(exponent)+" + "
            elif exponent==1:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])+"x + "
            else:
                annotationText=annotationText+FormatSciUsingError(c[order],e[order])
        annotationText=annotationText+", sy={0:.1E}".format(fit['sy'])
    elif annotationText=='Box':
        annotationText="Fit Details:\n"
        for order in range(t):
            exponent=t-order-1
            annotationText=annotationText+"C$_{x^{"+str(exponent)+"}}$ = "+FormatSciUsingError(c[order],e[order],extraDigit=1)+' $\pm$ '+"{0:.1E}".format(e[order])+'\n'
        annotationText=annotationText+'n = {0:d}'.format(fit['n'])+', DoF = {0:d}'.format(fit['n']-t)+", s$_y$ = {0:.1E}".format(fit['sy'])
    if (arrow==True):
        if (xArrow==0):
            xSpan=axisHandle.get_xlim()
            xArrow=np.mean(xSpan)
        if (yArrow==0):    
            yArrow=fit['poly'](xArrow)
        annotationObject=axisHandle.annotate(annotationText, 
                xy=(xArrow, yArrow), xycoords='data',
                xytext=(xText, yText),  textcoords='axes fraction',
                arrowprops={'color': color, 'width':1, 'headwidth':5},
                bbox={'boxstyle':'round', 'edgecolor':color,'facecolor':boxColor}
                )
    else:
        xSpan=axisHandle.get_xlim()
        xArrow=np.mean(xSpan)
        ySpan=axisHandle.get_ylim()
        yArrow=np.mean(ySpan)
        annotationObject=axisHandle.annotate(annotationText, 
                xy=(xArrow, yArrow), xycoords='data',
                xytext=(xText, yText),  textcoords='axes fraction',
                ha="left", va="center",
                bbox={'boxstyle':'round', 'edgecolor':color,'facecolor':boxColor}
                )
    annotationObject.draggable()
    return annotationObject

def RebalanceImageCV(frame,rfactor,gfactor,bfactor):
    offset=np.zeros(frame[:,:,0].shape,dtype="uint8")
    frame[:,:,0]=cv2.scaleAdd(frame[:,:,0], bfactor, offset)
    frame[:,:,1]=cv2.scaleAdd(frame[:,:,1], gfactor, offset)
    frame[:,:,2]=cv2.scaleAdd(frame[:,:,2], rfactor, offset)
    return frame

def MidPoint(pt1,pt2):
    return ((pt1[0]+pt2[0])/2.0, (pt1[1]+pt2[1])/2.0)

def OpenCVDisplayedHistogram(image,channel,mask,NumBins,DataMin,DataMax,x,y,w,h,DisplayImage,color,integrationWindow,labelFlag,labelText=""):
    x=np.round(x,decimals=0).astype(int)
    y=np.round(y,decimals=0).astype(int)
    w=np.round(w,decimals=0).astype(int)
    h=np.round(h,decimals=0).astype(int)
    avgVal=cv2.meanStdDev(image,mask=mask)
    histdata = cv2.calcHist([image],[channel],mask,[NumBins],[DataMin,DataMax])
    #domValue=np.argmax(histdata)
    #domCount=np.max(histdata)/np.sum(histdata) 
    sortArg=np.argsort(histdata,axis=0)
    domValue=np.sum(histdata[sortArg[-5:][:,0]][:,0]*sortArg[-5:][:,0])/np.sum(histdata[sortArg[-5:][:,0]][:,0])
    domCount=np.sum(histdata[sortArg[-5:][:,0]][:,0])/np.sum(histdata)
    #numpixels=sum(np.array(histdata[domValue-integrationWindow:domValue+integrationWindow+1]))
    cv2.normalize(histdata, histdata, 0, h, cv2.NORM_MINMAX);
    if w>NumBins:
        binWidth = w/NumBins
    else:
        binWidth=1
    #img = np.zeros((h, NumBins*binWidth, 3), np.uint8)
    for i in range(NumBins):
        freq = int(histdata[i])
        cv2.rectangle(DisplayImage, ((i*binWidth)+x, y+h), (((i+1)*binWidth)+x, y+h-freq), color)
    if labelFlag:
        cv2.putText(DisplayImage,labelText+" m="+'{0:.2f}'.format(domValue/float(NumBins-1)*(DataMax-DataMin))+" p="+'{0:.2f}'.format(domCount)+" a="+'{0:.2f}'.format(avgVal[0][channel][0])+" s="+'{0:.2f}'.format(avgVal[1][channel][0]),(x,y+h+12), font, 0.4,color,1,cv2.LINE_AA)
    return (avgVal[0][channel][0],avgVal[1][channel][0],domValue/float(NumBins-1)*(DataMax-DataMin))

def OpenCVDisplayedScatter(img, xdata,ydata,x,y,w,h,color,ydataRangemin=None, ydataRangemax=None,xdataRangemin=None, xdataRangemax=None,labelFlag=True):      
    if xdataRangemin==None: 
         xdataRangemin=np.min(xdata)       
    if xdataRangemax==None: 
         xdataRangemax=np.max(xdata) 
    if ydataRangemin==None: 
         ydataRangemin=np.min(ydata) 
    if ydataRangemax==None: 
         ydataRangemax=np.max(ydata)
    xdataRange=xdataRangemax-xdataRangemin
    ydataRange=ydataRangemax-ydataRangemin
    if xdataRange!=0:
        xscale=float(w)/xdataRange
    else:
        xscale=1
    if ydataRange!=0:
        yscale=float(h)/ydataRange
    else:
        yscale=1
    xdata=((xdata-xdataRangemin)*xscale).astype(np.int)
    xdata[xdata>w]=w
    xdata[xdata<0]=0
    ydata=((ydataRangemax-ydata)*yscale).astype(np.int)
    ydata[ydata>h]=h
    ydata[ydata<0]=0
    img[y+ydata,x+xdata]=color
#    img[y+ydata,x+xdata]=img[y+ydata,x+xdata]+np.array([100,100,100])
    cv2.rectangle(img,(x,y),(x+w+1,y+h+1),color,1)
    if labelFlag:
        cv2.putText(img,str(round(xdataRangemax,0)),(x+w-15,y+h+15), font, 0.4,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(xdataRangemin,0)),(x-5,y+h+15), font, 0.4,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(ydataRangemax,0)),(x-40,y+10), font, 0.4,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(ydataRangemin,0)),(x-40,y+h-5), font, 0.4,color,1,cv2.LINE_AA)
        
def ShiftHOriginToGreen(hue,maxHue):
    shifthsv=np.copy(hue)
    shifthsv[hue>=maxHue/3.0]=shifthsv[hue>=maxHue/3.0]-maxHue/3.0
    shifthsv[hue<maxHue/3.0]=shifthsv[hue<maxHue/3.0]+maxHue*2/3.0
    hue=shifthsv
    return hue

def ShiftHOriginToValue(hue,maxHue,newOrigin):
    shifthsv=np.copy(hue).astype('float')
    shiftAmount=maxHue-newOrigin
    shifthsv[hue<newOrigin]=shifthsv[hue<newOrigin]+shiftAmount
    shifthsv[hue>=newOrigin]=shifthsv[hue>=newOrigin]-newOrigin
    hue=shifthsv
    return hue

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
        
#open a dialog to allow user to select a file
root = tk.Tk()
root.withdraw()
file_path = askopenfilename(initialdir='/home/sensor/Documents/',filetypes=[('video files', '.mkv'),('video files', '.MOV'),('all files', '.*')])
#if no file is selected try reading from a connected webcam for up to 5 minutes

if len(file_path)==0:
    frameWidth=1920
    frameHeight=1080
    frameRate=25
#    frameWidth=640
#    frameHeight=480
    cap = cv2.VideoCapture(1)
    ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,frameWidth)
    if ret==False:
        cap.release()
        cap = cv2.VideoCapture(0)
        ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,frameWidth)
    TotalFrames=30*60*5
    liveCapture=True
#    frameWidth=4096
#    frameHeight=2160
    ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frameHeight)
#    ret=cap.set(cv2.CAP_PROP_BRIGHTNESS,128/255.0)
#    ret=cap.set(cv2.CAP_PROP_CONTRAST,128/255.0)
#    ret=cap.set(cv2.CAP_PROP_SATURATION,128/255.0)
#    ret=cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,0.75)
#    ret=cap.set(cv2.CAP_PROP_EXPOSURE,128/255.0)
#    ret=cap.set(cv2.CAP_PROP_AUTOFOCUS,1)
#    ret=cap.set(cv2.CAP_PROP_FOCUS,128/255.0)
    f = ['/']
    fnames=['Recorded.mkv']
    useQRinImage = input("Use settings in the image's QR (i/I), saved in a file (f/F), or default (d/D)?")
    if (useQRinImage=="f") | (useQRinImage=="F"):
        root = tk.Tk()
        root.withdraw()
        settings_file_path = askopenfilename(initialdir='/home/sensor/',filetypes=[('settings files', '.set'),('all files', '.*')])
        settingsFile = open(settings_file_path,'r')
        settingString=settingsFile.read()
        settingsFile.close()
        dictSet=eval(settingString)
        print(dictSet)
        ActiveState="Process"
else:    
    allFiles = input("Root is "+os.path.dirname(file_path)+"\n"+"To import all files in this directory type (Y/y)\n"+"To import all files in this and subdirectories type (S/s)\n"+"To import a single file type (N/n)\n")
    useQRinImage = input("Use settings in the image's QR (i/I), saved in a file (f/F), or default (d/D)?")
    if (useQRinImage=="f") | (useQRinImage=="F"):
        root = tk.Tk()
        root.withdraw()
        settings_file_path = askopenfilename(initialdir='/home/sensor/',filetypes=[('settings files', '.set'),('all files', '.*')])
        settingsFile = open(settings_file_path,'r')
        settingString=settingsFile.read()
        settingsFile.close()
        dictSet=eval(settingString)
        print(dictSet)
        ActiveState="Process"
    
    f = []
    fnames=[]
    summaryNames=[]
    summaryRateLinear=[]
    summaryRatePoly=[]
    
    if (allFiles=='N') | (allFiles=='n'):
        f.append(file_path)
        dirpath=os.path.dirname(file_path)
        filename=os.path.basename(file_path)
        fnames.append(filename)
    else:
        for (dirpath, dirnames, filenames) in os.walk(os.path.dirname(file_path)):
            f=[]
            recNum=0
            colorDataMean=[]
            colorDataMost=[]
            for filename in filenames:
                if (filename[-4:]=='.mkv') | (filename[-4:]=='.MOV') | (filename[-4:]=='.mp4'): #make this a setting, just put frame rate in settings (dictionary)
                    print (os.path.join(os.path.normpath(dirpath), filename))
                    f.append(os.path.join(os.path.normpath(dirpath), filename))
                    fnames.append(filename)
    dfSummary=pd.DataFrame(columns=fnames)           
    for file_path,file_name in zip(f,fnames):
        if file_path[-4:]=='.MOV':
            iTimeLapseFlag=True
            aTimeLapseFlag=False
        elif file_path[-4:]=='.mp4':
            aTimeLapseFlag=True
            iTimeLapseFlag=False 
        else:
            iTimeLapseFlag=False 
            aTimeLapseFlag=False
        cap = cv2.VideoCapture(file_path) # change around this line
        TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        frameHeight=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frameRate=cap.get(cv2.CAP_PROP_FPS)
        liveCapture=False

for file_path,file_name in zip(f,fnames):    
    if (useQRinImage=="i") | (useQRinImage=="I"):
        ActiveState="FindQR"
    else:
        ActiveState="Process"
        
    ParameterStats=np.zeros((32,6,TotalFrames*2,5))
    frameNumber=0
    rows=range(12)
    displayColors=[(0,0,255),(0,255,0),(255,50,50),(255,255,0),(200,200,200),(128,128,128),(255,255,255),(255,0,255),(0,255,255),(0,0,255),(0,255,0),(255,50,50)]
    channels=[2,1,0,0,1,2,0,1,2,2,1,0]
    labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]

    DisplayWidth=dictSet['dsp wh'][0]
    DisplayHeight=dictSet['dsp wh'][1]
    activeSettingsColumn=1
    activeSettingsRow=0
    #this will run indefinately until the 'q' key is pressed when Live capturing
    changeCameraFlag=False
    if liveCapture:
        currentFrame=frameNumber
    else:
        currentFrame=cap.get(cv2.CAP_PROP_POS_FRAMES)
    startTime=time.time()
    if RecordFlag:
        outp = cv2.VideoWriter(file_path+"Processed.avi",fourcc, frameRate, (DisplayWidth, DisplayHeight))
        outd = cv2.VideoWriter(file_path+"Diagnostic.avi",fourcc, frameRate, (DisplayWidth, DisplayHeight))
    while(liveCapture | (currentFrame<=TotalFrames) ):
        DisplayWidth=dictSet['dsp wh'][0]
        DisplayHeight=dictSet['dsp wh'][1]
        if liveCapture:
            currentTime=(time.time()-startTime)
            currentFrame=frameNumber
        else:
            if iTimeLapseFlag:
                currentTime=currentFrame*0.5
            elif aTimeLapseFlag:
                currentTime=currentFrame*(1.0/30*8)
            else:
                currentTime=cap.get(cv2.CAP_PROP_POS_MSEC)/1000
            currentFrame=cap.get(cv2.CAP_PROP_POS_FRAMES)
            frameRate=cap.get(cv2.CAP_PROP_FPS)
            #currentTime=currentFrame*0.5
            #currentTime=currentFrame*1
        displayFrame = np.zeros((DisplayHeight, DisplayWidth, 3), np.uint8)
        if ActiveState=="Process":
            if liveCapture & changeCameraFlag:
                ret=cap.set(cv2.CAP_PROP_FRAME_WIDTH,frameWidth)
                ret=cap.set(cv2.CAP_PROP_FRAME_HEIGHT,frameHeight)
                #ret=cap.set(cv2.CAP_PROP_BRIGHTNESS,dictSet['CAM bcs'][0]/255.0)
                #ret=cap.set(cv2.CAP_PROP_CONTRAST,dictSet['CAM bcs'][1]/255.0)
                #ret=cap.set(cv2.CAP_PROP_SATURATION,dictSet['CAM bcs'][2]/255.0)
                #ret=cap.set(cv2.CAP_PROP_AUTO_EXPOSURE,dictSet['CAM exp'][0]/2.0+0.25)
                #ret=cap.set(cv2.CAP_PROP_EXPOSURE,dictSet['CAM exp'][1]/255.0)
                #ret=cap.set(cv2.CAP_PROP_AUTOFOCUS,dictSet['CAM foc'][0])
                #ret=cap.set(cv2.CAP_PROP_FOCUS,dictSet['CAM foc'][1]/255.0)
            ret, frame = cap.read()
            if ret==False:
                break
    
        if ActiveState=="Pause":
    #        cv2.rectangle(displayFrame, (int(DisplayWidth*0.425/4),int(DisplayHeight*0.2/4)), (int(DisplayWidth*0.475/4),int(DisplayHeight*0.8/4)), (255,255,255),-1)
    #        cv2.rectangle(displayFrame, (int(DisplayWidth*0.525/4),int(DisplayHeight*0.2/4)), (int(DisplayWidth*0.575/4),int(DisplayHeight*0.8/4)), (255,255,255),-1)
            ret, frame = cap.read()
            if ret==False:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame)
    
        if ActiveState=="FindQR":
    #        cv2.rectangle(displayFrame, (int(DisplayWidth*0.425/4),int(DisplayHeight*0.2/4)), (int(DisplayWidth*0.475/4),int(DisplayHeight*0.8/4)), (255,255,255),-1)
    #        cv2.rectangle(displayFrame, (int(DisplayWidth*0.525/4),int(DisplayHeight*0.2/4)), (int(DisplayWidth*0.575/4),int(DisplayHeight*0.8/4)), (255,255,255),-1)
            ret, frame = cap.read()
            if ret==False:
                break
        #by explictly copying changes to img will NOT change frame
        img=np.copy(frame)
        #converts to HSV colorspace
        hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if referenceFlag:
            #categorizes each pixel, either inside limits or outside limits; makes a B&W image with cyan as white
            boxMask = cv2.inRange(hsvFrame, np.array(dictSet['box ll']), np.array(dictSet['box ul'])) 
            #cv2.imshow('CyanMask', boxMask)
            #finds areas of connected pixels in the mask, https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html
            if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
                contours,hierarchy = cv2.findContours(boxMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            else:
                image,contours,hierarchy = cv2.findContours(boxMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours)>=1:
                #looks through the contours to identify the largest one
                MaxBoxArea=0
                ContourIndex=0
                LargestContour=0
                for contour in contours:
                    area=cv2.contourArea(contour)
                    if area>MaxBoxArea:
                        MaxBoxArea=area
                        LargestContour=ContourIndex
                    ContourIndex=ContourIndex+1
                outerBoxContour=contours[LargestContour]
                cv2.drawContours(img,[outerBoxContour],0,(0,255,0),2)
                
                ptsFound=np.zeros((40,4),dtype='float32')
                c12CircleMask = cv2.inRange(hsvFrame, np.array(dictSet['c12 ll']), np.array(dictSet['c12 ul'])) 
                c34CircleMask = cv2.inRange(hsvFrame, np.array(dictSet['c34 ll']), np.array(dictSet['c34 ul'])) 
                #cv2.imshow('c34Mask', c34CircleMask)
                #cv2.imshow('c12Mask', c12CircleMask)
                #finds areas of connected pixels in the mask, https://docs.opencv.org/3.3.1/d4/d73/tutorial_py_contours_begin.html
                if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
                    contoursc12,hierarchy = cv2.findContours(c12CircleMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                else:
                    image,contoursc12,hierarchy = cv2.findContours(c12CircleMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                #ContourIndex=0
                #LargestContour=0
                circleIndex=0
                for contour in contoursc12:
                    area=cv2.contourArea(contour)
                    if area>(MaxBoxArea*0.005):
                        M = cv2.moments(contour)
                        if M['m00']>0:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                            dist = cv2.pointPolygonTest(outerBoxContour,(cx,cy),False)
                            if dist!=-1:
                                ptsFound[circleIndex,0]=cx
                                ptsFound[circleIndex,1]=cy
                                ptsFound[circleIndex,2]=area
                                ptsFound[circleIndex,3]=1
                                circleIndex=circleIndex+1
                                cv2.drawContours(img,[contour],0,(255,0,0),2)
                                cv2.circle(img,(int(cx),int(cy)), 2, (255,0,0), -1)
                if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
                    contoursc34,hierarchy = cv2.findContours(c34CircleMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                else:
                    image,contoursc34,hierarchy = cv2.findContours(c34CircleMask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)                
                for contour in contoursc34:
                    area=cv2.contourArea(contour)
                    if area>(MaxBoxArea*0.005):
                        M = cv2.moments(contour)
                        if M['m00']>0:
                            #cx = int(M['m10']/M['m00'])
                            #cy = int(M['m01']/M['m00'])
                            cx = M['m10']/M['m00']
                            cy = M['m01']/M['m00']
                            dist = cv2.pointPolygonTest(outerBoxContour,(cx,cy),False)
                            if dist!=-1:
                                ptsFound[circleIndex,0]=cx
                                ptsFound[circleIndex,1]=cy
                                ptsFound[circleIndex,2]=area
                                ptsFound[circleIndex,3]=0
                                circleIndex=circleIndex+1
                                cv2.drawContours(img,[contour],0,(0,0,255),2)
                                cv2.circle(img,(int(cx),int(cy)), 2, (0,0,255), -1)
         
                ptsCard = np.float32([[600,225],[1500,225],[600,975],[1500,975]])
                ptsCard = np.float32([[dictSet['cl1 xy'][0],dictSet['cl1 xy'][1]],[dictSet['cl2 xy'][0],dictSet['cl2 xy'][1]],[dictSet['cl3 xy'][0],dictSet['cl3 xy'][1]],[dictSet['cl4 xy'][0],dictSet['cl4 xy'][1]]])
                #fptsCard = np.float32([[250,250],[1550,250],[250,950],[1550,950]])
                #ptsCard = np.float32([[150,150],[1650,150],[150,1050],[1650,1050]])
                ptsImage= np.float32([[150,150],[1650,150],[150,1050],[1650,1050]])
        
                if circleIndex==4:
                    if (cv2.pointPolygonTest(outerBoxContour,MidPoint(ptsFound[0,0:2],ptsFound[2,0:2]),False)==-1) & (cv2.pointPolygonTest(outerBoxContour,MidPoint(ptsFound[0,0:2],ptsFound[3,0:2]),False)==-1):
                        ptsImage[0,0]=ptsFound[0,0]
                        ptsImage[0,1]=ptsFound[0,1]
                        ptsImage[1,0]=ptsFound[1,0]
                        ptsImage[1,1]=ptsFound[1,1]
                    else:
                        ptsImage[1,0]=ptsFound[0,0]
                        ptsImage[1,1]=ptsFound[0,1]
                        ptsImage[0,0]=ptsFound[1,0]
                        ptsImage[0,1]=ptsFound[1,1]
                    if (cv2.pointPolygonTest(outerBoxContour,MidPoint(ptsImage[1,0:2],ptsFound[2,0:2]),False)==-1):
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
                    rotImage = cv2.warpPerspective(frame,Mrot,(1800,1200))

        else:
            rotImage = np.copy(frame)
        if (decodedQR==False) & (ActiveState=="FindQR"):
            bar=rotImage[395:395+400,1090:1090+400,:] #Change these hard coded numbers (its irrelevant)
            gray = cv2.cvtColor(bar, cv2.COLOR_BGR2GRAY)
            pil = Image.fromarray(gray)
            width, height = pil.size
            raw = pil.tobytes()
            
            image = zbar.Image(width, height, 'Y800', raw)
            scanner = zbar.Scanner()
            scanner.scan(image)
            for symbol in image:
                    # do something useful with results
                print ('decoded', symbol.type, 'symbol', '"%s"' % symbol.data)
                dictSet=eval(symbol.data)
                decodedQR=True
                ActiveState="Process"
            frameScale=max(frameWidth/(DisplayWidth/2.0),frameHeight/(DisplayHeight/2.0))
            frameImageScale = cv2.resize(frame, (int(frameWidth/frameScale),int(frameHeight/frameScale)), interpolation = cv2.INTER_AREA)
            displayFrame[0:frameImageScale.shape[0],0:frameImageScale.shape[1],:]=frameImageScale
            cv2.imshow('Display', displayFrame)
            keypress=cv2.waitKey(1) & 0xFF
            continue

        rgbWBR = rotImage[dictSet['WBR xy'][1]:dictSet['WBR xy'][1]+dictSet['WBR wh'][1], dictSet['WBR xy'][0]:dictSet['WBR xy'][0]+dictSet['WBR wh'][0]]
        hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
        maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
        
        if np.sum(maskWBR)>0:
            RGBGreyROI=cv2.mean(rgbWBR, mask=maskWBR)
            bscale=RGBGreyROI[0]
            gscale=RGBGreyROI[1]
            rscale=RGBGreyROI[2]
            scalemax=max(rscale,gscale,bscale)
            if dictSet['WBR sc'][0]!=0:
                scalemin=dictSet['WBR sc'][0]
            else:
                scalemin=min(rscale,gscale,bscale)
            if (scalemin!=0) & (min(rscale,gscale,bscale)!=0):
                rfactor=float(scalemin)/float(rscale)
                gfactor=float(scalemin)/float(gscale)
                bfactor=float(scalemin)/float(bscale)
            rotImage=RebalanceImageCV(rotImage,rfactor,gfactor,bfactor)
            frame=RebalanceImageCV(frame,rfactor,gfactor,bfactor)

        rgbWBR = rotImage[dictSet['WBR xy'][1]:dictSet['WBR xy'][1]+dictSet['WBR wh'][1], dictSet['WBR xy'][0]:dictSet['WBR xy'][0]+dictSet['WBR wh'][0]]
        hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
        maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
        for row, displayColor, channel in zip([0,1,2], [(0, 0, 128),(0, 128, 0),(128, 25, 25)], [2,1,0]):              
            #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskROI,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/10)+(row*5),256,(DisplayHeight/12),displayFrame,displayColor,5,True,label)
            ParameterStats[row,0,frameNumber,0],ParameterStats[row,1,frameNumber,0],ParameterStats[row,2,frameNumber,0]=OpenCVDisplayedHistogram(rgbWBR,channel,maskWBR,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/14)+(row*6),256,(DisplayHeight/16),displayFrame,displayColor,5,False)


#            labWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2LAB)
#            inputImages= [rgbWBR,rgbWBR,rgbWBR,hsvWBR,hsvWBR,hsvWBR,labWBR,labWBR,labWBR]
#            for row, displayColor, inputImage, channel, label in zip(rows[0:3], displayColors[0:3], inputImages[0:3], channels[0:3],labels[0:3]):              
#                ParameterStats[row,0,frameNumber,0],ParameterStats[row,1,frameNumber,0],ParameterStats[row,2,frameNumber,0]=OpenCVDisplayedHistogram(inputImage,channel,maskWBR,128,0,255,DisplayWidth/2+256+10,5+(row*45),128,40,displayFrame,displayColor,5,False,label)
#            frameWBR=cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR) 
#            scaleWBR=max(frameWBR.shape[1]/(128),frameWBR.shape[0]/(128))
#            imageScaleWBR = cv2.resize(frameWBR, (int(frameWBR.shape[1]/scaleWBR),int(frameWBR.shape[0]/scaleWBR)), interpolation = cv2.INTER_AREA)
#            displayFrame[int(5+(9*45)):int((5+(9*45))+imageScaleWBR.shape[0]),int(DisplayWidth/2+256+10):int(DisplayWidth/2+256+10+imageScaleWBR.shape[1]),:]=imageScaleWBR

        rgbRO2 = rotImage[dictSet['RO2 xy'][1]:dictSet['RO2 xy'][1]+dictSet['RO2 wh'][1], dictSet['RO2 xy'][0]:dictSet['RO2 xy'][0]+dictSet['RO2 wh'][0]]
        rgbRO3 = rotImage[dictSet['RO3 xy'][1]:dictSet['RO3 xy'][1]+dictSet['RO3 wh'][1], dictSet['RO3 xy'][0]:dictSet['RO3 xy'][0]+dictSet['RO3 wh'][0]]
        rgbROI = rotImage[dictSet['ROI xy'][1]:dictSet['ROI xy'][1]+dictSet['ROI wh'][1], dictSet['ROI xy'][0]:dictSet['ROI xy'][0]+dictSet['ROI wh'][0]]
        rgbRO2summary=cv2.meanStdDev(rgbRO2)
        rgbRO3summary=cv2.meanStdDev(rgbRO3)

        hsvROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2HSV)
        hsvROI[:,:,0]=ShiftHOriginToValue(hsvROI[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
        labROI = cv2.cvtColor(rgbROI, cv2.COLOR_BGR2LAB)
        logsrgbROI=cv2.LUT(rgbROI, linLUTabs)*64
        maskROI = cv2.inRange(hsvROI, np.array(dictSet['ROI ll']), np.array(dictSet['ROI ul']))
        resFrame = cv2.bitwise_and(rgbROI,rgbROI, mask= maskROI)
        if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
            contours,hierarchy = cv2.findContours(maskROI,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        else:
            image,contours,hierarchy = cv2.findContours(maskROI,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        contourMask=np.zeros((maskROI.shape),dtype='uint8')
        if len(contours)>=1:
            MaxROIArea=0
            ContourIndex=0
            LargestContour=0
            for contour in contours:
                area=cv2.contourArea(contour)
                if area>MaxROIArea:
                    MaxROIArea=area
                    LargestContour=ContourIndex
                ContourIndex=ContourIndex+1
            outerROIContour=contours[LargestContour]
            boundingRectangle=cv2.minAreaRect(outerROIContour)
            cv2.drawContours(resFrame,[outerROIContour],0,(0,255,0),2)
            cv2.drawContours(contourMask,[outerROIContour],0,(255),-1)
            maskROI=contourMask
            ParameterStats[15,0,frameNumber,1]=MaxROIArea*cmPerPixel*cmPerPixel
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
            
            roiScale=max(resFrame.shape[1]/(DisplayWidth/2.1),resFrame.shape[0]/(DisplayHeight/2.1))
            roiImageScale = cv2.resize(resFrame, (int(resFrame.shape[1]/roiScale),int(resFrame.shape[0]/roiScale)), interpolation = cv2.INTER_AREA)
            displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+roiImageScale.shape[0]),0:roiImageScale.shape[1],:]=roiImageScale
            inputImages= [rgbROI,rgbROI,rgbROI,hsvROI,hsvROI,hsvROI,labROI,labROI,labROI,logsrgbROI,logsrgbROI,logsrgbROI]
            for row, displayColor, inputImage, channel, label in zip(rows, displayColors, inputImages, channels,labels):              
                #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskROI,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/10)+(row*5),256,(DisplayHeight/12),displayFrame,displayColor,5,True,label)
                ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskROI,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/14)+(row*6),256,(DisplayHeight/16),displayFrame,displayColor,5,True,label)
            ParameterStats[12,0,frameNumber,1]=ParameterStats[10,0,frameNumber,1]-ParameterStats[9,0,frameNumber,1]
            ParameterStats[13,0,frameNumber,1]=ParameterStats[11,0,frameNumber,1]-ParameterStats[9,0,frameNumber,1]
            ParameterStats[14,0,frameNumber,1]=ParameterStats[10,0,frameNumber,1]-ParameterStats[11,0,frameNumber,1]

            maskROIVolume = cv2.inRange(hsvROI, np.array([int(ParameterStats[3,0,frameNumber,1]-ParameterStats[3,1,frameNumber,1]),int(ParameterStats[4,0,frameNumber,1]-ParameterStats[4,1,frameNumber,1]),int(ParameterStats[5,0,frameNumber,1]-ParameterStats[5,1,frameNumber,1])]), np.array([255,255,255]))
            if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
                contours,hierarchy = cv2.findContours(maskROIVolume,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            else:
                image,contours,hierarchy = cv2.findContours(maskROIVolume,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            contourMask=np.zeros((maskROIVolume.shape),dtype='uint8')
            if len(contours)>=1:
                MaxROIAreaVolume=0
                ContourIndex=0
                LargestContour=0
                for contour in contours:
                    area=cv2.contourArea(contour)
                    if area>MaxROIAreaVolume:
                        MaxROIAreaVolume=area
                        LargestContour=ContourIndex
                    ContourIndex=ContourIndex+1
                outerROIContourVolume=contours[LargestContour]
            ParameterStats[18,0,frameNumber,1]=MaxROIAreaVolume*cmPerPixel*cmPerPixel

            if frameNumber>=2:
                if dictSet['xa1 sc'][0]==0:
                    xMin=dictSet['xa1 sc'][1]
                    xMax=dictSet['xa1 sc'][2]
                else:
                    xMin=None
                    xMax=None          
                if dictSet['ya1 sc'][0]==0:
                    yMin=dictSet['ya1 sc'][1]
                    yMax=dictSet['ya1 sc'][2]
                else:
                    yMin=None
                    yMax=None     
                OpenCVDisplayedScatter(displayFrame, ParameterStats[dictSet['xa1 ch'][0],dictSet['xa1 ch'][1],0:frameNumber,dictSet['xa1 ch'][2]],ParameterStats[dictSet['ya1 ch'][0],dictSet['ya1 ch'][1],0:frameNumber,dictSet['ya1 ch'][2]],dictSet['pl1 xy'][0],dictSet['pl1 xy'][1],dictSet['pl1 wh'][0],dictSet['pl1 wh'][1],(255,255,255),ydataRangemin=yMin, ydataRangemax=yMax,xdataRangemin=xMin, xdataRangemax=xMax)
                if dictSet['xa2 sc'][0]==0:
                    xMin=dictSet['xa2 sc'][1]
                    xMax=dictSet['xa2 sc'][2]
                else:
                    xMin=None
                    xMax=None          
                if dictSet['ya2 sc'][0]==0:
                    yMin=dictSet['ya2 sc'][1]
                    yMax=dictSet['ya2 sc'][2]
                else:
                    yMin=None
                    yMax=None     
                OpenCVDisplayedScatter(displayFrame, ParameterStats[dictSet['xa2 ch'][0],dictSet['xa2 ch'][1],0:frameNumber,dictSet['xa2 ch'][2]],ParameterStats[dictSet['ya2 ch'][0],dictSet['ya2 ch'][1],0:frameNumber,dictSet['ya2 ch'][2]],dictSet['pl2 xy'][0],dictSet['pl2 xy'][1],dictSet['pl2 wh'][0],dictSet['pl2 wh'][1],(255,255,255),ydataRangemin=yMin, ydataRangemax=yMax,xdataRangemin=xMin, xdataRangemax=xMax)
            if ActiveState=="Process":
                frameNumber=frameNumber+1
    
        frameScale=max(frameWidth/(DisplayWidth/2.0),frameHeight/(DisplayHeight/2.0))
        frameImageScale = cv2.resize(frame, (int(frameWidth/frameScale),int(frameHeight/frameScale)), interpolation = cv2.INTER_AREA)
        displayFrame[0:frameImageScale.shape[0],0:frameImageScale.shape[1],:]=frameImageScale
        
        if settingsFlag:
            fontScale=0.3
            parmHeight=int(DisplayHeight/60.0)
            parmWidth=int(DisplayWidth/30.0)
            #cv2.putText(displayFrame,"Frame "+str(frameNumber),(DisplayWidth-128,12*1), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            setRow=0
            for setRow,setting in zip(range(len(dictSet)),sorted(dictSet)): 
                if (activeSettingsRow==setRow):
                    setColor=(0,0,255)
                else:
                    setColor=(255,255,255)
                cv2.putText(displayFrame,setting,(DisplayWidth-(parmWidth*5),parmHeight*(setRow+1)), font, fontScale,setColor,1,cv2.LINE_AA)
                if activeSettingsColumn>len(dictSet[sorted(dictSet)[activeSettingsRow]])-1:
                    activeSettingsColumn=len(dictSet[sorted(dictSet)[activeSettingsRow]])-1
                for setCol in range(len(dictSet[setting])):
                    if (activeSettingsColumn==setCol) & (activeSettingsRow==setRow):
                        setColor=(0,0,255)
                    else:
                        setColor=(255,255,255)
                    cv2.putText(displayFrame,str(dictSet[setting][setCol]),(DisplayWidth-(parmWidth*3)+(parmWidth*setCol),parmHeight*(setRow+1)), font, fontScale,setColor,1,cv2.LINE_AA)
                #cv2.putText(displayFrame,'LL: H '++', S '+str(SensorLower_lim[1])+', V '+str(SensorLower_lim[2]),(DisplayWidth-128,12*2), font, fontScale,(255,255,255),1,cv2.LINE_AA)
        
        if displayHelp:
            quitKey = ' to quit'
            pauseKey = ' to pause'
            rrBig = ' to rewind (big)'
            rrSmall= ' to rewind (small)'
            ffSmall = ' to fastforward (small)' #fastforward 1 frame
            ffBig = ' to fastforward (big)' #fastforward 10s
            showInfo = ' to show information' #show diagnostics 
            toggleSettingsKey = ' to toggle settings'
            increaseSettingValue = ' to increase value'
            decreaseSettingValue = ' to decrease value'
            moveUpSettings = ' to move up'
            moveDownSettings = ' to move down' 
            moveLeftSettings = ' to move left'
            moveRightSettings = ' to move right'
            
            textXlocation=2
            cv2.putText(displayFrame,'"q"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, quitKey, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(quitKey)) + 10
            
            
            cv2.putText(displayFrame,'"p"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, pauseKey, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(pauseKey)) + 10

            
            cv2.putText(displayFrame,'"j"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation + 15
            cv2.putText(displayFrame, rrSmall, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(rrSmall)) + 10
            
            cv2.putText(displayFrame,'"h"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, rrBig, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(rrBig)) + 10
            
            cv2.putText(displayFrame,'"k"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame,ffSmall, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(ffSmall)) + 10
            
            cv2.putText(displayFrame,'"l"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, ffBig, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(ffBig)) + 10
            
            cv2.putText(displayFrame,'"i"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, showInfo, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(showInfo)) + 10
            
            cv2.putText(displayFrame,'"t"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, toggleSettingsKey, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(toggleSettingsKey)) + 10
            
            cv2.putText(displayFrame,'"+"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, increaseSettingValue, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(increaseSettingValue)) + 10
            
            cv2.putText(displayFrame,'"-"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, decreaseSettingValue, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(decreaseSettingValue)) + 10
            
            cv2.putText(displayFrame,'"w"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, moveUpSettings, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(moveUpSettings)) + 10
            
            cv2.putText(displayFrame,'"a"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, moveLeftSettings, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(moveLeftSettings)) + 10
            
            cv2.putText(displayFrame,'"s"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, moveDownSettings, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(moveDownSettings)) + 10
            
            cv2.putText(displayFrame,'"d"', (textXlocation,DisplayHeight-8), font, fontScale,(255,255,255),1,cv2.LINE_AA)
            textXlocation=textXlocation+15
            cv2.putText(displayFrame, moveRightSettings, (textXlocation,DisplayHeight-8), font, fontScale,(128,128,128),1,cv2.LINE_AA)
            textXlocation=textXlocation + (5*len(moveRightSettings)) + 10
            
        
        
        else:
            cv2.putText(displayFrame,'type "?" for hotkeys', (2,DisplayHeight-8),font, fontScale,(255,255,255),1,cv2.LINE_AA)
            
        #cv2.imshow('Result', img)
        if ActiveState=="Pause":
            cv2.rectangle(displayFrame, (int(DisplayWidth*0.425/2),int(DisplayHeight*0.2/2)), (int(DisplayWidth*0.475/2),int(DisplayHeight*0.8/2)), (255,255,255),-1)
            cv2.rectangle(displayFrame, (int(DisplayWidth*0.525/2),int(DisplayHeight*0.2/2)), (int(DisplayWidth*0.575/2),int(DisplayHeight*0.8/2)), (255,255,255),-1)
    
        cv2.imshow('Display', displayFrame)
        if maskDiagnostic:
            cv2.imshow('Diagnostic', img)
            cv2.imshow('WB', rgbWBR)
            cv2.imshow('RO2', rgbRO2)
            cv2.imshow('RO3', rgbRO3)
            cv2.imshow('rotImage', rotImage)
        #reads the keyboard
        if RecordFlag & (ActiveState=="Process"):
            #cv2.putText(img,"REC",(10,10), font, .5,(0,0,255),2,cv2.LINE_AA) 
#            if overlayFlag:
#                dim2=frameImageScale.shape[0]
#                dim1=frameImageScale.shape[1]
#                #OpenCVDisplayedScatter(img, xdata,ydata,x,y,w,h,color,ydataRangemin=None, ydataRangemax=None,xdataRangemin=None, xdataRangemax=None,labelFlag=True)
#                OpenCVDisplayedScatter(frameImageScale, ParameterStats[dictSet['xax ch'][0],0,0:frameNumber,1],ParameterStats[dictSet['yax ch'][0],0,0:frameNumber,1],int(dim1/10),int(dim2/10),int(dim1/4),int(dim2/3),(255,255,255),ydataRangemin=10, ydataRangemax=45,xdataRangemin=xMin, xdataRangemax=xMax)
#                cv2.imshow('frameImageScale', frameImageScale)
#                if (currentTime>100) & (currentTime<400):
#                    outp.write(frameImageScale)
#            else:
                outp.write(displayFrame)
            #outr.write(frame)
        keypress=cv2.waitKey(1) & 0xFF
        #print(keypress)
        changeCameraFlag=False
        if keypress == ord('q'):
            break
        if keypress == ord('i'):
            maskDiagnostic=not(maskDiagnostic)
        if keypress == ord('t'):
            settingsFlag=not settingsFlag
        if keypress == ord('r'):
            RecordFlag=not RecordFlag
        if keypress == ord('c'):
            referenceFlag=not referenceFlag  
        if keypress == ord('p'):
            if ActiveState=="Pause":
                ActiveState="Process"
            else:
                ActiveState="Pause"
        if keypress == ord('l'):
    #        currentFrame=cap.get(cv2.CAP_PROP_POS_FRAMES)
            if currentFrame+(frameRate*10)<TotalFrames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame+(frameRate*10))
        if keypress == ord('h'):
    #        currentFrame=cap.get(cv2.CAP_PROP_POS_FRAMES)
            if currentFrame-(frameRate*10)>0:
                cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame-(frameRate*10))
        if keypress == ord('k'):
    #        currentFrame=cap.get(cv2.CAP_PROP_POS_FRAMES)
            if currentFrame<TotalFrames:
                cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame+1)
        if keypress == ord('j'):
    #        currentFrame=cap.get(cv2.CAP_PROP_POS_FRAMES)f
            if currentFrame>0:
                cap.set(cv2.CAP_PROP_POS_FRAMES,currentFrame-1)
        if settingsFlag:
            if (keypress==ord('+')) & (dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]<dictUL[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]):
                dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]=dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]+1
                if sorted(dictSet)[activeSettingsRow].find('CAM')==0:
                    changeCameraFlag=True
            if (keypress==ord('-')) & (dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]>0):
                dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]=dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]-1    
                if sorted(dictSet)[activeSettingsRow].find('CAM')==0:
                    changeCameraFlag=True
            if (keypress==ord('>')) & (dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]<dictUL[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]-9):
                dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]=dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]+10
                if sorted(dictSet)[activeSettingsRow].find('CAM')==0:
                    changeCameraFlag=True
            if (keypress==ord('<')) & (dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]>9):
                dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]=dictSet[sorted(dictSet)[activeSettingsRow]][activeSettingsColumn]-10   
                if sorted(dictSet)[activeSettingsRow].find('CAM')==0:
                    changeCameraFlag=True
            if ((keypress==upArrow) | (keypress==ord('w'))) & (activeSettingsRow>0):
                activeSettingsRow=activeSettingsRow-1
            if ((keypress==dnArrow) | (keypress==ord('s'))) & (activeSettingsRow<len(dictSet)-1):
                activeSettingsRow=activeSettingsRow+1
            if ((keypress==ltArrow) | (keypress==ord('a'))) & (activeSettingsColumn>0):
                activeSettingsColumn=activeSettingsColumn-1
            if ((keypress==rtArrow) | (keypress==ord('d'))) & (activeSettingsColumn<len(dictSet[sorted(dictSet)[activeSettingsRow]])-1):
                activeSettingsColumn=activeSettingsColumn+1
    cap.release()
    cv2.destroyAllWindows()
    if RecordFlag:
        outp.release()
        outd.release()
    saveSettings = input("Save current settings (Y/n)?")
    if (saveSettings=="Y") | (saveSettings=="y"):
        root = tk.Tk()
        root.withdraw()
        settings_file_path = asksaveasfilename(initialdir='/home/sensor/',filetypes=[('settings files', '.set'),('all files', '.*')])
        settingsFile = open(settings_file_path,'w')
        settingsFile.write(str(dictSet))
        settingsFile.close()
        
    dropSignal=ParameterStats[1,0,0:frameNumber,3]-ParameterStats[1,0,0:frameNumber,0]
    dropSignalMean=np.mean(dropSignal[int(len(dropSignal)/4):int(len(dropSignal)*3/4)])
    dropSignalStd=np.std(dropSignal[int(len(dropSignal)/4):int(len(dropSignal)*3/4)])
#    plt.plot(ParameterStats[31,0,0:frameNumber,3],dropSignal,'-.r')
#    plt.hlines(dropSignalMean,np.min(ParameterStats[31,0,0:frameNumber,3]),np.max(ParameterStats[31,0,0:frameNumber,3]))
#    plt.hlines(dropSignalMean-(dropSignalStd/2),np.min(ParameterStats[31,0,0:frameNumber,3]),np.max(ParameterStats[31,0,0:frameNumber,3]))
#    plt.hlines(dropSignalMean+(dropSignalStd/2),np.min(ParameterStats[31,0,0:frameNumber,3]),np.max(ParameterStats[31,0,0:frameNumber,3]))
    boolDrop=hyst(dropSignal, dropSignalMean-(dropSignalStd/2), dropSignalMean+(dropSignalStd/2))    
#    boolDrop=hyst(dropSignal, dropSignalMean-(dropSignalStd/2), dropSignalMean)    
    times,frames=crossBoolean(ParameterStats[31,0,0:frameNumber,3], boolDrop, crossPoint=0.5, direction='rising')
    drops=np.zeros((len(dropSignal)))
    signalPrior=np.zeros((4,len(frames)))
    for index,count in zip(frames,range(len(frames))):
        drops[index:]=drops[index:]+1
        for chan in range(4):
            signalPrior[chan,count]=np.mean(ParameterStats[chan,0,index-5:index,1])
    #plt.plot(ParameterStats[31,0,0:frameNumber,3],drops,'-ok')
    plt.scatter(range(len(frames)),signalPrior[3,:],c=np.transpose(signalPrior[0:3,:]/255))
    
    ledSignal=ParameterStats[0,0,0:frameNumber,2]-ParameterStats[2,0,0:frameNumber,2]
    #plt.plot(ParameterStats[31,0,0:frameNumber,3],ledSignal,'-og')
    ledSignalMean=np.mean(ledSignal)
    ledSignalStd=np.std(ledSignal)
    boolled=hyst(ledSignal, 10, 30)    
    times,frames=crossBoolean(ParameterStats[31,0,0:frameNumber,3], boolled, crossPoint=0.5, direction='rising')
    leds=np.zeros((len(ledSignal)))
    for index in frames:
        leds[index:]=leds[index:]+1
    #plt.plot(ParameterStats[31,0,0:frameNumber,3],leds,'-.g')
    
    #plt.figure()
    #plt.scatter(drops[1:30],ParameterStats[3,0,1:30,1],c=np.transpose(ParameterStats[0:3,0,1:30,1]/255))
    #plt.scatter(drops,ParameterStats[3,0,0:frameNumber,1],c=np.transpose(ParameterStats[0:3,0,0:frameNumber,1]/255),alpha=0.2)

    #h=1.11+(−0.17)/(1+𝑒^□(64&((𝑝𝐻−5.12))/0.39) )+(−0.32)/(1+𝑒^□(64&((𝑝𝐻−7.73))/0.50) )+(−0.32)/(1+𝑒^□(64&((𝑝𝐻−9.35))/0.16) )

    volumeChannel=ParameterStats[15,0,0:frameNumber,1]
    volumeScale=cellWidth/(np.mean(ParameterStats[16,0,0:frameNumber,1])*cmPerPixel)*cellDepth
    initialVolume=np.mean(volumeChannel[drops==0]*volumeScale)
    volumeAddedArea=volumeChannel*volumeScale-initialVolume
    volumeAddedDrops=drops*dropVolume
    plt.plot(volumeAddedDrops,volumeAddedArea,'or',alpha=0.05)
    
    #plt.plot(ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1],ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1],'or',alpha=0.2)
    dfMean=pd.DataFrame(data=ParameterStats[0:12,0,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=ParameterStats[31,0,0:frameNumber,1])
    dfStdev=pd.DataFrame(data=ParameterStats[0:12,1,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=ParameterStats[31,0,0:frameNumber,1])
    dfMost=pd.DataFrame(data=ParameterStats[0:12,2,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=ParameterStats[31,0,0:frameNumber,1])
     
    writer = pd.ExcelWriter(file_path+"Data.xlsx", engine='xlsxwriter')
    workbook  = writer.book
    dfMean.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=6,index=False)
    dfStdev.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=19,index=False)
    dfMost.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=32,index=False)
    worksheetData = writer.sheets['FrameData']
    worksheetData.write('G1', 'Means')
    worksheetData.write('T1', 'Standard Deviations')
    worksheetData.write('AG1', 'Most Frequent Values')
    worksheetData.write('A2', 'Time')
    worksheetData.write('B2', 'FrameNumber')
    worksheetData.write('C2', 'ROIArea')
    worksheetData.write('D2', 'Height')
    worksheetData.write('E2', 'DropSignal')
    worksheetData.write('F2', 'Drops')
    worksheetData.write_column('A3', ParameterStats[31,0,0:frameNumber,1])
    worksheetData.write_column('B3', ParameterStats[30,0,0:frameNumber,1])
    worksheetData.write_column('C3', ParameterStats[15,0,0:frameNumber,1])
    worksheetData.write_column('D3', ParameterStats[16,0,0:frameNumber,1])
    worksheetData.write_column('E3', dropSignal)
    worksheetData.write_column('F3', drops)
    
    worksheetSummary = workbook.add_worksheet("Summary")
    worksheetSummary.write('A1', 'Drop#')
    worksheetSummary.write('B1', 'H')
    worksheetSummary.write('C1', 'R')
    worksheetSummary.write('D1', 'G')
    worksheetSummary.write('E1', 'B')
    worksheetSummary.write_column('A2', np.arange(len(frames)))
    worksheetSummary.write_column('B2', signalPrior[3,:])
    worksheetSummary.write_column('C2', signalPrior[0,:])
    worksheetSummary.write_column('D2', signalPrior[1,:])
    worksheetSummary.write_column('E2', signalPrior[2,:])
    workbook.close()
    writer.save()
    
#    chart1 = workbook.add_chart({'type': 'scatter'})
#    numAllEntries=ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1].size
#    chart1.add_series({
#        'name': labels[dictSet['ya1 ch'][0]]+' linear',
#        'categories': ["Fit", 1, 2, 1+numEntries-1, 2],
#        'values': ["Fit", 1, 3, 1+numEntries-1, 3],
#        'trendline': {
#            'type': 'linear',
#            'display_equation': True,
#            'line': {
#            'color': 'black',
#            'width': 2,
#            },
#            'forward': ParameterStats[dictSet['xa1 ch'][0],0,frameNumber-1,1],
#            'backward': ParameterStats[dictSet['xa1 ch'][0],0,0,1],
#        },
#        'marker': {
#            'type': 'circle',
#            'size': 8,
#            'fill':   {'color': '#a66fb5'},
#        },
#    })
#    chart1.add_series({
#        'name': labels[dictSet['ya1 ch'][0]]+' all',
#        'categories': ["Fit", 1, 0, 1+numAllEntries-1, 0],
#        'values': ["Fit", 1, 1, 1+numAllEntries-1, 1],
#        'marker': {
#                'type': 'circle',
#                'size': 4,
#                'fill':   {'color': '#490648'},
#        },
#    })
#
#    #chart1.set_title ({'name': labels[dictSet['ya1 ch'][0]]+' Change'})
#    chart1.set_x_axis({
#            'name': 'Time (seconds)',
#            'min': np.min(np.floor(ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1])),
#            'max': np.max(np.ceil(ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1]))
#            })
#    chart1.set_y_axis({
#            'name': 'Signal',
#            'min': np.min(np.floor(ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1])),
#            'max': np.max(np.ceil(ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1])),
#            'major_gridlines': {
#                    'visible': False,
#                    },
#            })
#    #chart1.set_style(6)
#    chart1.set_legend({'position': 'none'})
#    worksheetFit.insert_chart('H8', chart1, {'x_offset': 25, 'y_offset': 10})
#    workbook.close()
#    writer.save()
#    
#    df=pd.DataFrame(data=ParameterStats[:,0,0:frameNumber,1])
#    df=df.T
#    
#    df=df.rename(index=int, columns={0:"R",1:"G",2:"B"})
#    df=df.rename(index=int, columns={3:'H',4:"S",5:"V"})
#    df=df.rename(index=int, columns={6:'L',7:"a*",8:"b*"})
#    df=df.rename(index=int, columns={9:'aR',10:"aG",11:"aB"})
#    df=df.rename(index=int, columns={15:'numPix',16:'volume'})
#    df=df.rename(index=int, columns={31:'t',30:'frame'})
#    df=df.rename(index=int, columns={18:'RO2'})
#
#    fig, axarr = plt.subplots(3, 1, sharex='col', sharey='row',figsize=(10, 10))
#    column=0
#    #dfBool=(df.R>0) & ((df.numPix/df.ROIPix)>minPixelFraction) & (df.t>=minTime) & (df.t<=maxTime)
#    #dfBool=(df.R>0) & (df.t>=minTime) & (df.t<=maxTime) 
#    minPix=30000
#    linFits=[]
#    polyFits=[]
#    polyTimeMid=[]
#    dfBool=(df.R>0) & (df.numPix>minPix)
#    
#    dfBoolGR=(dfBool) & ((df['aG']-df['aR'])<=25) & ((df['aG']-df['aR'])>=3)
#    X=np.array(df.t[dfBoolGR])
#    Y=np.array((df['aG']-df['aR'])[dfBoolGR])
#    fit1=PolyReg(X,Y,1)
#    fit2=PolyReg(X,Y,2)
#    linFits.append(fit1)
#    polyFits.append(fit2)
#    polyTimeMid.append(X[np.argmin(abs(Y-15))])
#    axarr[0].plot(df.t[dfBool],(df['aG']-df['aR'])[dfBool],'.',color='xkcd:deep teal')    
#    axarr[0].plot(X,Y,'.',color='xkcd:greenish teal')
#    Xrange=np.linspace(np.min(X),np.max(X),100)
#    axarr[0].set_ylabel('Green-Red Abs Video')
#    propError=np.sqrt(fit2['errors'][0]**2+fit2['errors'][1]**2)
#    FitAnnText=FormatSciUsingError(fit2['coefs'][0]*2*X[np.argmin(abs(Y-np.median(Y)))]+fit2['coefs'][1],propError,withError=True,extraDigit=1)
#    LinFitAnnText=FormatSciUsingError(fit1['coefs'][0],fit1['errors'][0],withError=True,extraDigit=1)
#    AnnotateFit(fit1,axarr[0],annotationText='poly2 rate = '+FitAnnText+'\nlinear rate = '+LinFitAnnText,color='black',xText=0.08,yText=0.2)
#    
#    dfBoolBR=(dfBool) & (df['aB']-df['aR']<=maxSignal) & (df['aB']-df['aR']>=minSignal)
#    X=np.array(df.t[dfBoolBR])
#    Y=np.array((df['aB']-df['aR'])[dfBoolBR])
#    fit1=PolyReg(X,Y,1)
#    fit2=PolyReg(X,Y,2)
#    linFits.append(fit1)
#    polyFits.append(fit2)
#    polyTimeMid.append(X[np.argmin(abs(Y-60))])
#    axarr[1].plot(df.t[dfBool],(df['aB']-df['aR'])[dfBool],'.',color='xkcd:deep violet')    
#    axarr[1].plot(X,Y,'.',color='xkcd:soft purple')
#    Xrange=np.linspace(np.min(X),np.max(X),100)
#    axarr[1].set_ylabel('Blue-Red Abs Video')
#    propError=np.sqrt(fit2['errors'][0]**2+fit2['errors'][1]**2)
#    FitAnnText=FormatSciUsingError(fit2['coefs'][0]*2*X[np.argmin(abs(Y-np.median(Y)))]+fit2['coefs'][1],propError,withError=True,extraDigit=1)
#    LinFitAnnText=FormatSciUsingError(fit1['coefs'][0],fit1['errors'][0],withError=True,extraDigit=1)
#    AnnotateFit(fit1,axarr[1],annotationText='poly2 rate = '+FitAnnText+'\nlinear rate = '+LinFitAnnText,color='black',xText=0.08,yText=0.2)
#    
#    summaryNames.append(file_name)
#    summaryRateLinear.append(fit1['coefs'][0])
#    summaryRatePoly.append(fit2['coefs'][0]*2*X[np.argmin(abs(Y-np.median(Y)))]+fit2['coefs'][1])
#    
#    dfBoolS=(dfBool) & (df['S']<=90) & (df['S']>=60)
#    X=np.array(df.t[dfBoolS])
#    Y=np.array(df['S'][dfBoolS])
#    fit1=PolyReg(X,Y,1)
#    fit2=PolyReg(X,Y,2)
#    linFits.append(fit1)
#    polyFits.append(fit2)
#    polyTimeMid.append(X[np.argmin(abs(Y-110))])
#    axarr[2].plot(df.t[dfBool],df['S'][dfBool],'.',color='xkcd:deep blue')    
#    axarr[2].plot(X,Y,'.',color='xkcd:soft blue')
#    Xrange=np.linspace(np.min(X),np.max(X),100)
#    axarr[2].set_ylabel('Saturation Video')
#    propError=np.sqrt(fit2['errors'][0]**2+fit2['errors'][1]**2)
#    FitAnnText=FormatSciUsingError(fit2['coefs'][0]*2*X[np.argmin(abs(Y-np.median(Y)))]+fit2['coefs'][1],propError,withError=True,extraDigit=1)
#    LinFitAnnText=FormatSciUsingError(fit1['coefs'][0],fit1['errors'][0],withError=True,extraDigit=1)
#    AnnotateFit(fit1,axarr[2],annotationText='poly2 rate = '+FitAnnText+'\nlinear rate = '+LinFitAnnText,color='black',xText=0.08,yText=0.2)
#    
#    fig.suptitle(file_name, fontsize=16)
#    fig.savefig(file_path+'plot.png', dpi=1000)

#summaryTemp=[]
#for name in summaryNames:
#    #temp=name[7:11]
#    #temp=name[8:12]
#    temp=name[name.find("T")+1:name.find("T")+5]
#    temp=float(temp)+273.15
#    summaryTemp.append(temp)
#    
#writer = pd.ExcelWriter(dirpath+"/SummaryData.xlsx", engine='xlsxwriter')
#workbook  = writer.book
#worksheetData = workbook.add_worksheet()
#worksheetData.write('A1', 'Filename')
#worksheetData.write('B1', 'Temp')
#worksheetData.write('C1', 'Slope')
#worksheetData.write_column('A2', summaryNames)
#worksheetData.write_column('B2', summaryTemp)
#worksheetData.write_column('C2', summaryRateLinear)
#workbook.close()
#
#fig, ax = plt.subplots()
#t=np.array(summaryTemp)
#X=1/t
#Y=np.log(-np.array(summaryRateLinear))
#arrFit=PolyReg(X,Y,1)
#ax.plot(X,Y,'ok')
#xLine=[np.min(X),np.max(X)]
#ax.plot(xLine,arrFit['poly'](xLine),'-k')
#AnnotateFit(arrFit,ax,annotationText='Box')
#fig.suptitle(fnames, fontsize=8)
#
#
#tFit=[303.35,304.15,305.15]
#xFit=[]
#yFit=[]
#for tempFit in tFit:
#    xFit.append(X[t==tempFit])
#    yFit.append(Y[t==tempFit])
