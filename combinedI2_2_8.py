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
import numpy as np
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
#import datetime
#import matplotlib.pyplot as plt

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
    
elif versionOS=='M':
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
{
'CAM bcs':[128, 128, 128],
'CAM exp':[1, 50],
'CAM foc':[1, 60],
'RO1 ll':[50, 80, 0],
'RO1 ul':[67, 255, 255],
'RO1 wh':[920, 420],
'RO1 xy':[1280, 230],
'RO2 ll':[0, 0, 0],
'RO2 ul':[227, 255, 255],
'RO2 wh':[850, 500],
'RO2 xy':[0, 325],
'RO3 ll':[0, 0, 0],
'RO3 ul':[255, 255, 255],
'RO3 wh':[280, 40],
'RO3 xy':[1310, 498],
'WBR ll':[0, 0, 0],
'WBR sc':[160],
'WBR ul':[255, 40, 255],
'WB1 wh':[470, 100],
'WB1 xy':[1310, 260],
'box ll':[70, 20, 40],
'box ul':[130, 255, 255],
'c12 ll':[20, 20, 0],
'c12 ul':[40, 255, 255],
'c34 ll':[120, 20, 0],
'c34 ul':[160, 255, 255],
'cl1 xy':[1420, 765],
'cl2 xy':[2295, 765],
'cl3 xy':[1420, 135],
'cl4 xy':[2295, 135],
'dsp wh':[1700, 900],
'hue lo':[180.0, 150.0],
'pl1 wh':[220, 220],
'pl1 xy':[1160, 10],
'pl2 wh':[220, 220],
'pl2 xy':[1160, 310],
'xa1 ch':[31, 0, 1],
'xa1 sc':[1, 0, 0],
'xa2 ch':[15, 0, 1],
'xa2 sc':[1, 0, 0],
'ya1 ch':[13, 0, 1],
'ya1 sc':[1, 20, 150],
'ya2 ch':[8, 0, 1],
'ya2 sc':[0, 20, 150],
}
'''
    upperLimitString='''
{'CAM bcs':[255, 255, 255],
 'CAM exp':[  1, 255],
 'CAM foc':[ 1, 255],
 'RO1 ll':[ 255, 255,  255],
 'RO1 ul':[255, 255, 255],
 'RO1 wh':[2600, 1200],
 'RO1 xy':[2600, 1200],
 'RO2 ll':[ 255, 255,  255],
 'RO2 ul':[255, 255, 255],
 'RO2 wh':[2600, 1200],
 'RO2 xy':[2600, 1200],
 'RO3 ll':[ 255, 255,  255],
 'RO3 ul':[255, 255, 255],
 'RO3 wh':[2600, 1200],
 'RO3 xy':[2600, 1200],
 'WBR sc':[255],
 'WBR ll':[255, 255, 255],
 'WBR ul':[255, 255, 255],
 'WB1 wh':[2600, 1200],
 'WB1 xy':[2600, 1200],
 'WB2 wh':[2600, 1200],
 'WB2 xy':[2600, 1200],
 'WB3 wh':[2600, 1200],
 'WB3 xy':[2600, 1200],
 'box ll':[255, 255, 255],
 'box ul':[255, 255, 255],
 'cl1 xy':[2600,2600],
 'cl2 xy':[2600,2600],
 'cl3 xy':[2600,2600],
 'cl4 xy':[2600,2600],
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
    
    see https://matplotlib.org/api/_as_gen/matplotlition='cw'):
    shifthsv=np.copy(hue).astype('float')b.pyplot.annotate.html
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

def OpenCVDisplayedScatter(img, xdata,ydata,x,y,w,h,color,ydataRangemin=None, ydataRangemax=None,xdataRangemin=None, xdataRangemax=None,alpha=1,labelFlag=True):      
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
    if alpha==1:
        img[y+ydata,x+xdata]=color
    else:
        xCords=xdata+x
        yCords=ydata+y
        #consider making this bigger
        colorApha=(int(color[0]*alpha),int(color[1]*alpha),int(color[2]*alpha))
        for ptx,pty in zip(xCords,yCords):
            img[yCords,xCords]=img[yCords,xCords]+colorApha
#    img[y+ydata,x+xdata]=img[y+ydata,x+xdata]+np.array([100,100,100])
    cv2.rectangle(img,(x,y),(x+w+1,y+h+1),color,1)
    if labelFlag:
        cv2.putText(img,str(round(xdataRangemax,0)),(x+w-15,y+h+15), font, 0.4,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(xdataRangemin,0)),(x-5,y+h+15), font, 0.4,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(ydataRangemax,0)),(x-40,y+10), font, 0.4,color,1,cv2.LINE_AA)
        cv2.putText(img,str(round(ydataRangemin,0)),(x-40,y+h-5), font, 0.4,color,1,cv2.LINE_AA)
        
def ShiftHOriginToValue(hue,maxHue,newOrigin,direction='cw'):
    shifthsv=np.copy(hue).astype('float')
    shiftAmount=maxHue-newOrigin
    shifthsv[hue<newOrigin]=shifthsv[hue<newOrigin]+shiftAmount
    shifthsv[hue>=newOrigin]=shifthsv[hue>=newOrigin]-newOrigin
    hue=shifthsv
    if direction=='ccw':
        hue=maxHue-hue
    return hue

def rotate_bound(image, angle):
    # grab the dimensions of the image and then determine the
    # center
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY
    # perform the actual rotation and return the image
    return cv2.warpAffine(image, M, (nW, nH))

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
        


useQRinImage = input("Use settings in the image's QR (i/I), saved in a file (f/F), or default (d/D)?")
if (useQRinImage=="f") | (useQRinImage=="F"):
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    
    if versionOS=='L':
        initialdir=r'/home/cantrell/Downloads/EmailedVideo'
    elif versionOS=='W':
        initialdir=r'C:\Users\cantrell\Dropbox (UofP)\ChemicalVision\chemicalVision'
    settings_file_path = askopenfilename(initialdir=os.getcwd(),filetypes=[('settings files', '.set'),('all files', '.*')])
    settingsFile = open(settings_file_path,'r')
    settingString=settingsFile.read()
    settingsFile.close()
    dictSet=eval(settingString)
    print(dictSet)
    ActiveState="Process"
    
mail = imaplib.IMAP4_SSL(SMTP_SERVER)
mail.login(FROM_EMAIL,FROM_PWD)
last_processed_email=0


runFlag=True
monitorEmailFlag=True

while runFlag:
    while monitorEmailFlag:
        mail.select('inbox')
        typ, data = mail.search(None, 'ALL')
        mail_ids = data[0]
        id_list = mail_ids.split()   
        first_email_id = int(id_list[0])
        latest_email_id = int(id_list[-1])
        if latest_email_id>last_processed_email:
            first_email_id=last_processed_email+1
            monitorEmailFlag=False
            break
            #downloadFlag=True
        else:
            print("Last email was " + str(last_processed_email) + " sleeping for one minute...zzz")
            time.sleep(60)
        if time.time()>endTime:
            runFlag=False
            break
        if runFlag==False:
            break
    monitorEmailFlag=True
    fileList=[]
    dateList=[]
    returnList=[]
    subjectList=[]
    summaryExcelList=[]
    processedFileList=[]
    
    #stopFlag=False       
    for i in np.arange(first_email_id,latest_email_id+1):
        if runFlag==False:
            break
        #type, data = mail.fetch(i, '(RFC822)' )
        typ, data = mail.fetch(str(i), '(RFC822)' )
        last_processed_email=i
        print("last_processed_email: ",last_processed_email)
        for response_part in data:
            if isinstance(response_part, tuple):
    #            msg = email.message_from_string(response_part[1])
                msg = email.message_from_string(response_part[1].decode())
                timestamp=time.mktime(email.utils.parsedate(msg['Date']))-(greenwichTimeAdjustment*60*60)
                print('Considering email: '+msg['subject'])
                if (timestamp>=emailStartTime) & (timestamp<=endTime):
                    email_subject = msg['subject']
                    if (email_subject==None) | (email_subject==''):
                        email_subject='No Subject'
                    print('In time range email: '+email_subject)
                    for part in msg.walk():
                        attachInfo=part['Content-Disposition']
                        attachFlag=False
                        if attachInfo!=None:
                            if part['Content-Disposition'][:10]=='attachment':
                                attachFlag=True
                        if (part.get_content_maintype() == 'video') | (attachFlag):
                                
                                #timeStruct=time.localtime(timestamp)
                                
                                #find creation date and look to the next semicolon
                                #Cut the string off at the beginning and at end to ensure 
                                #that only the creation date is taken
                                if attachInfo.find('creation-date="')!=-1:
                                    startPosition = attachInfo.find('creation-date="')
                                    stringAfterCreationDate = attachInfo[startPosition:]
                                    findCreationDate = stringAfterCreationDate.find(';')
                                    creationDate = attachInfo[startPosition+len('creation-date="'):findCreationDate + startPosition - 1] # - 1 to cut off other "
                                    attachmentDate = email.utils.parsedate(creationDate)
                                    attachmentStamp=time.mktime(attachmentDate)-(greenwichTimeAdjustment*60*60)
                                    attachmentDate=time.localtime(attachmentStamp)
                                    dateName=strftime("%Y_%m_%d_%H:%M", attachmentDate)
                                else:
                                    timeStruct=time.localtime(timestamp)
                                    dateName=strftime("%Y_%m_%d_%H:%M", timeStruct)
                                if msg['Return-Path']==None:
                                    returnAddress=msg['From'][msg['From'].find('<')+1:msg['From'].find('>')]
                                else:
                                    returnAddress =  msg['Return-Path']
                                    
                                if versionOS=='L':
                                    bad_chars=[":","<",">"]
                                    for c in bad_chars : 
                                        email_subject = email_subject.replace(c, ' ') 
                                    for c in bad_chars : 
                                        dateName = dateName.replace(c, '_') 
                                    for c in bad_chars : 
                                        returnAddress = returnAddress.replace(c, '') 
                                    fileName=filePath+'/'+ dateName + '#'+ returnAddress +'#'+ email_subject+'.MOV'
                                    
                                if versionOS=='W':
                                    bad_chars=[":","<",">"]
                                    for c in bad_chars : 
                                        email_subject = email_subject.replace(c, ' ') 
                                    for c in bad_chars : 
                                        dateName = dateName.replace(c, '_') 
                                    for c in bad_chars : 
                                        returnAddress = returnAddress.replace(c, '')
                                    fileName=filePath+'\\'+ dateName + '#'+ returnAddress +'#'+ email_subject+'.MOV'
                                    
                                if versionOS=='M':
                                    bad_chars=[":","<",">"]
                                    for c in bad_chars : 
                                        email_subject = email_subject.replace(c, ' ') 
                                    for c in bad_chars : 
                                        dateName = dateName.replace(c, '_') 
                                    for c in bad_chars : 
                                        returnAddress = returnAddress.replace(c, '') 
                                    fileName=filePath+'/'+ dateName + '#'+ returnAddress +'#'+ email_subject+'.MOV'
                                    
                                print ('Downloading and saving '+fileName)
                                fp = open(fileName, 'wb')
                                fileList.append(fileName)
                                dateList.append(dateName)
                                returnList.append(returnAddress)
                                subjectList.append(email_subject)
                                fp.write(part.get_payload(decode=True))
                                fp.close()

                                file_path=fileName
                                returnAddress=returnAddress
                                emailSubject=email_subject
                            #for file_path,returnAddress,emailSubject in zip(fileList,returnList,subjectList):    
                                cap = cv2.VideoCapture(file_path) # change around this line
                            
                                if file_path[-4:]=='.MOV':
                                    iTimeLapseFlag=True
                                    aTimeLapseFlag=False
                                    frameRate=2
                                elif file_path[-4:]=='.mp4':
                                    aTimeLapseFlag=True
                                    iTimeLapseFlag=False
                                    frameRate=30/8
                                else:
                                    iTimeLapseFlag=False 
                                    aTimeLapseFlag=False
                                    frameRate=cap.get(cv2.CAP_PROP_FPS)
                            
                                TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                                frameWidth=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                                frameHeight=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                            
                                liveCapture=False
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
                                videoStartTime=time.time()
                                if RecordFlag:
                                    if versionOS=='L':
                                        outFileName=filePath+'/Processed/'+ dateName + '#' + email_subject +'#'+'Processed.mp4'
                                    if versionOS=='W':
                                        outFileName=filePath+'\\Processed\\'+ dateName + '#'+ email_subject +'#'+'Processed.mp4'
                                    if versionOS=='M':
                                        outFileName=filePath+'/Processed/'+ dateName + '#' + email_subject +'#'+'Processed.mp4'
                                    if iTimeLapseFlag:
                                        outp = cv2.VideoWriter(outFileName,fourcc, 10, (DisplayWidth, DisplayHeight))
                                    elif aTimeLapseFlag:
                                        outp = cv2.VideoWriter(outFileName,fourcc, 10, (DisplayWidth, DisplayHeight))
                                    else:
                                        outp = cv2.VideoWriter(outFileName,fourcc, frameRate, (DisplayWidth, DisplayHeight))
                                while(liveCapture | (currentFrame<=TotalFrames) ):
                                    DisplayWidth=dictSet['dsp wh'][0]
                                    DisplayHeight=dictSet['dsp wh'][1]
                                    if liveCapture:
                                        currentTime=(time.time()-videoStartTime)
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
                                                if (area>(MaxBoxArea*0.005)) & (area<(MaxBoxArea*0.25)):
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
                                                if (area>(MaxBoxArea*0.005)) & (area<(MaxBoxArea*0.25)):
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
                                            #ptsImage= np.float32([[150,150],[1650,150],[150,1050],[1650,1050]]) 
                                            ptsImage= np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
                            
                                            skipFrame=False        
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
                                                #rotImage = cv2.warpPerspective(frame,Mrot,(1800,1200))
                                                rotImage = cv2.warpPerspective(frame,Mrot,(2600,900))
                                            else:
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
                                            rotImage=RebalanceImageCV(rotImage,rfactor,gfactor,bfactor)
                                            frame=RebalanceImageCV(frame,rfactor,gfactor,bfactor)
                                
                                        rgbWBR[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]] = rotImage[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]]
                                        rgbWBR[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]] = rotImage[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]]
                                        rgbWBR[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]] = rotImage[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]]
                                        hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
                                        maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
                                        for row, displayColor, channel in zip([0,1,2], [(0, 0, 128),(0, 128, 0),(128, 25, 25)], [2,1,0]):              
                                            #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskRO1,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/10)+(row*5),256,(DisplayHeight/12),displayFrame,displayColor,5,True,label)
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
                                        rgbRO1 = rotImage[dictSet['RO1 xy'][1]:dictSet['RO1 xy'][1]+dictSet['RO1 wh'][1], dictSet['RO1 xy'][0]:dictSet['RO1 xy'][0]+dictSet['RO1 wh'][0]]
                                        rgbRO2summary=cv2.meanStdDev(rgbRO2)
                                        rgbRO3summary=cv2.meanStdDev(rgbRO3)
                                
                                        hsvRO1 = cv2.cvtColor(rgbRO1, cv2.COLOR_BGR2HSV)
                                        #hsvRO2 = cv2.cvtColor(rgbRO2, cv2.COLOR_BGR2HSV)
                                        #hsvRO3 = cv2.cvtColor(rgbRO3, cv2.COLOR_BGR2HSV)
                            
                                        hsvRO1[:,:,0]=ShiftHOriginToValue(hsvRO1[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
                                        #hsvRO2[:,:,0]=ShiftHOriginToValue(hsvRO2[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
                                        #hsvRO3[:,:,0]=ShiftHOriginToValue(hsvRO3[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
                            
                                        labRO1 = cv2.cvtColor(rgbRO1, cv2.COLOR_BGR2LAB)
                                        logsrgbRO1=cv2.LUT(rgbRO1, linLUTabs)*64
                                        maskRO1 = cv2.inRange(hsvRO1, np.array(dictSet['RO1 ll']), np.array(dictSet['RO1 ul']))
                                        #maskRO2 = cv2.inRange(hsvRO2, np.array(dictSet['RO2 ll']), np.array(dictSet['RO2 ul']))
                                        #maskRO3 = cv2.inRange(hsvRO3, np.array(dictSet['RO3 ll']), np.array(dictSet['RO3 ul']))
                                        resFrameWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)
                                        #resFrameRO2 = cv2.bitwise_and(rgbRO2,rgbRO2, mask= maskRO2)
                                        #resFrameRO3 = cv2.bitwise_and(rgbRO3,rgbRO3, mask= maskRO3)
                            
                            #             if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
                            #                 contours,hierarchy = cv2.findContours(maskRO2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                            #             else:
                            #                 image,contours,hierarchy = cv2.findContours(maskRO2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                            #             contourMask=np.zeros((maskRO2.shape),dtype='uint8')
                            #             if len(contours)>=1:
                            #                 MaxRO2Area=0
                            #                 ContourIndex=0
                            #                 LargestContour=0
                            #                 for contour in contours:
                            #                     area=cv2.contourArea(contour)
                            #                     if area>MaxRO2Area:
                            #                         MaxRO2Area=area
                            #                         LargestContour=ContourIndex
                            #                     ContourIndex=ContourIndex+1
                            #                 outerRO2Contour=contours[LargestContour]
                            #                 boundingRectangle=cv2.minAreaRect(outerRO2Contour)
                                            
                                            
                            #                 #cv2.drawContours(resFrameRO2,[outerRO2Contour],0,(0,255,0),2)
                            #                 box = cv2.boxPoints(boundingRectangle) 
                            #                 box = np.int0(box)
                            #                 #box=box+20
                                            
                            #                 maskThermo=np.zeros((maskRO3.shape),dtype='uint8')
                            #                 cv2.drawContours(resFrameRO2,[box],0,(0,0,255),2)
                            #                 cv2.drawContours(maskThermo,[box],0,(255),-1)
                            #                 resFrameRO3 = cv2.bitwise_and(rgbRO3,rgbRO3, mask= maskThermo)
                            
                            #             #try using box to define points for 
                            # #                ptsThermo= np.float32([[600, 160],[220, 126],[221, 107],[849, 141]])
                            # #                ptsBox= np.float32(box)
                            # #                MrotTherm = cv2.getPerspectiveTransform(ptsThermo,ptsBox)
                            # #                rotImageTherm = cv2.warpPerspective(frame,Mrot,(2600,900))
                            
                                            
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
                                            
                                            RO1Scale=max(resFrameRO1.shape[1]/(DisplayWidth/4),resFrameRO1.shape[0]/(DisplayHeight/4))
                                            RO1ImageScale = cv2.resize(resFrameRO1, (int(resFrameRO1.shape[1]/RO1Scale),int(resFrameRO1.shape[0]/RO1Scale)), interpolation = cv2.INTER_AREA)
                            
                                            WBRScale=max(resFrameWBR.shape[1]/(DisplayWidth/4),resFrameWBR.shape[0]/(DisplayHeight/4))
                                            WBRImageScale = cv2.resize(resFrameWBR, (int(resFrameWBR.shape[1]/WBRScale),int(resFrameWBR.shape[0]/WBRScale)), interpolation = cv2.INTER_AREA)
                            
                                            resFrameRO2=rgbRO2
                                            RO2Scale=max(resFrameRO2.shape[1]/(DisplayWidth/4),resFrameRO2.shape[0]/(DisplayHeight/4))
                                            RO2ImageScale = cv2.resize(resFrameRO2, (int(resFrameRO2.shape[1]/RO2Scale),int(resFrameRO2.shape[0]/RO2Scale)), interpolation = cv2.INTER_AREA)
                                            
                                            resFrameRO3=rotImage
                                            RO3Scale=max(resFrameRO3.shape[1]/(DisplayWidth/4),resFrameRO3.shape[0]/(DisplayHeight/4))
                                            RO3ImageScale = cv2.resize(resFrameRO3, (int(resFrameRO3.shape[1]/RO3Scale),int(resFrameRO3.shape[0]/RO3Scale)), interpolation = cv2.INTER_AREA)
                            
                                            displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+RO1ImageScale.shape[0]),0:RO1ImageScale.shape[1],:]=RO1ImageScale
                                            displayFrame[int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4):int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4)+RO2ImageScale.shape[0],0:RO2ImageScale.shape[1],:]=RO2ImageScale
                            
                                            displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+WBRImageScale.shape[0]) , int(displayFrame.shape[0]/2):int(displayFrame.shape[0]/2)+WBRImageScale.shape[1],:]=WBRImageScale
                                            displayFrame[int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4):int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4)+RO3ImageScale.shape[0] , int(displayFrame.shape[0]/2):int(displayFrame.shape[0]/2)+RO3ImageScale.shape[1],:]=RO3ImageScale
                            
                                            inputImages= [rgbRO1,rgbRO1,rgbRO1,hsvRO1,hsvRO1,hsvRO1,labRO1,labRO1,labRO1,logsrgbRO1,logsrgbRO1,logsrgbRO1]
                                            for row, displayColor, inputImage, channel, label in zip(rows, displayColors, inputImages, channels,labels):              
                                                #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskRO1,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/10)+(row*5),256,(DisplayHeight/12),displayFrame,displayColor,5,True,label)
                                                ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/14)+(row*6),256,(DisplayHeight/16),displayFrame,displayColor,5,True,label)
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
                                    #frameImageScale = cv2.resize(frame, (int(frameWidth/frameScale),int(frameHeight/frameScale)), interpolation = cv2.INTER_AREA)
                                    frameImageScale = cv2.resize(img, (int(frameWidth/frameScale),int(frameHeight/frameScale)), interpolation = cv2.INTER_AREA)

                                    displayFrame[0:frameImageScale.shape[0],0:frameImageScale.shape[1],:]=frameImageScale
                                    
                                    fontScale=0.3
                                    if settingsFlag:
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
                            #                OpenCVDisplayedScatter(frameImageScale, ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1],ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1],int(dim1/10),int(dim2/10),int(dim1/4),int(dim2/3),(255,255,255),ydataRangemin=10, ydataRangemax=45,xdataRangemin=xMin, xdataRangemax=xMax)
                            #                cv2.imshow('frameImageScale', frameImageScale)
                            #                if (currentTime>100) & (currentTime<400):
                            #                    outp.write(frameImageScale)
                            #            else:
                                            outp.write(displayFrame)
                                        #outr.write(frame)
                                    keypress=cv2.waitKeyEx(1)
                                    #print(keypress)
                                    changeCameraFlag=False
                                    if keypress == ord('q'):
                                        runFlag=False
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
                                    #outd.release()
                            
                                    
                                dfMean=pd.DataFrame(data=ParameterStats[0:12,0,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=ParameterStats[31,0,0:frameNumber,1])
                                dfStdev=pd.DataFrame(data=ParameterStats[0:12,1,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=ParameterStats[31,0,0:frameNumber,1])
                                dfMost=pd.DataFrame(data=ParameterStats[0:12,2,0:frameNumber,1].transpose(),columns=["R","G","B","H","S","V","L*","a*","b*","Ra","Ga","Ba"],index=ParameterStats[31,0,0:frameNumber,1])

                                if versionOS=='L':
                                    outExcelFileName=filePath+'/Processed/'+ dateName + '#' + email_subject +'#'+'Data.xlsx'
                                if versionOS=='W':
                                    outExcelFileName=filePath+'\\Processed\\'+ dateName + '#'+ email_subject +'#'+'Data.xlsx'
                                if versionOS=='M':
                                    outExcelFileName=filePath+'/Processed/'+ dateName + '#' + email_subject +'#'+'Data.xlsx'
                                writer = pd.ExcelWriter(outExcelFileName, engine='xlsxwriter')
                                workbook  = writer.book
                                minArea=2
                                #maxArea=50000
                                maxSignal=40
                                minSignal=16
                                dfMinArea=ParameterStats[15,0,0:frameNumber,1]>minArea
                                dfHeightRange=(ParameterStats[16,0,0:frameNumber,1]>np.mean(ParameterStats[16,0,0:frameNumber,1][dfMinArea])*0.95) & (ParameterStats[16,0,0:frameNumber,1]<np.mean(ParameterStats[16,0,0:frameNumber,1][dfMinArea])*1.05)
                                #dfBool=dfMinArea & dfHeightRange
                                dfBool=(dfMinArea) & (ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1]<=maxSignal) & (ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1]>=minSignal)

                                worksheetFit = workbook.add_worksheet("Fit")
                                worksheetFit.write('A1', 'Time')
                                worksheetFit.write('B1', labels[dictSet['ya1 ch'][0]])
                                worksheetFit.write('C1', 'Time (linear range)')
                                worksheetFit.write('D1', labels[dictSet['ya1 ch'][0]]+' (linear range)')
                                worksheetFit.write_column('A2',ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1])
                                worksheetFit.write_column('B2',ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1])
                                worksheetFit.write_column('C2',ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1][dfBool])
                                worksheetFit.write_column('D2',ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1][dfBool])
                                numEntries=ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1][dfBool].size
                                numIndex=str(numEntries+1)
                                worksheetFit.write_array_formula('I3:J5', '{=LINEST(D2:D'+numIndex+',C2:C'+numIndex+',TRUE,TRUE)}')
                                worksheetFit.write('I2', 'Slope')
                                worksheetFit.write('J2', 'Intercept')
                                worksheetFit.write('H3', 'coefs')
                                worksheetFit.write('H4', 'errors')
                                worksheetFit.write('H5', 'r2, sy')
                                chart1 = workbook.add_chart({'type': 'scatter'})
                                numAllEntries=ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1].size
                                chart1.add_series({
                                    'name': labels[dictSet['ya1 ch'][0]]+' linear',
                                    'categories': ["Fit", 1, 2, 1+numEntries-1, 2],
                                    'values': ["Fit", 1, 3, 1+numEntries-1, 3],
                                    'trendline': {
                                        'type': 'linear',
                                        'display_equation': True,
                                        'line': {
                                        'color': 'black',
                                        'width': 2,
                                        },
                                        'forward': ParameterStats[dictSet['xa1 ch'][0],0,frameNumber-1,1],
                                        'backward': ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1][dfBool][0],
                                    },
                                    'marker': {
                                        'type': 'circle',
                                        'size': 8,
                                        'fill':   {'color': '#a66fb5'},
                                        'border':   {'color': '#a66fb5'},
                                    },
                                })
                                chart1.add_series({
                                    'name': labels[dictSet['ya1 ch'][0]]+' all',
                                    'categories': ["Fit", 1, 0, 1+numAllEntries-1, 0],
                                    'values': ["Fit", 1, 1, 1+numAllEntries-1, 1],
                                    'marker': {
                                            'type': 'circle',
                                            'size': 4,
                                            'fill':   {'color': '#490648'},
                                            'border':   {'color': '#490648'},
                                    },
                                })
                            
                                #chart1.set_title ({'name': labels[dictSet['ya1 ch'][0]]+' Change'})
                                if (ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1].size!=0) and (ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1].size!=0):
                                    chart1.set_x_axis({
                                            'name': 'Time (seconds)',
                                            'min': np.min(np.floor(ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1])),
                                            'max': np.max(np.ceil(ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1]))
                                            })
                                    chart1.set_y_axis({
                                            'name': 'Signal',
                                            'min': np.min(np.floor(ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1])),
                                            'max': np.max(np.ceil(ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1])),
                                            'major_gridlines': {
                                                    'visible': False,
                                                    },
                                            })
                                    #chart1.set_style(6)
                                    chart1.set_legend({'position': 'none'})
                                    worksheetFit.insert_chart('H8', chart1, {'x_offset': 25, 'y_offset': 10})
                                
                                dfMean.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=6,index=False)
                                dfStdev.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=19,index=False)
                                dfMost.to_excel(writer, sheet_name='FrameData',startrow=1,startcol=32,index=False)
                                worksheetData = writer.sheets['FrameData']
                                worksheetData.write('G1', 'Means')
                                worksheetData.write('K1', 'Standard Deviations')
                                worksheetData.write('AG1', 'Most Frequent Values')
                                worksheetData.write('A2', 'Time')
                                worksheetData.write('B2', 'FrameNumber')
                                worksheetData.write('C2', 'RO1Area')
                                worksheetData.write('D2', 'Height')
                                worksheetData.write('E2', 'Width')
                                worksheetData.write_column('A3', ParameterStats[31,0,0:frameNumber,1])
                                worksheetData.write_column('B3', ParameterStats[30,0,0:frameNumber,1])
                                worksheetData.write_column('C3', ParameterStats[15,0,0:frameNumber,1])
                                worksheetData.write_column('D3', ParameterStats[16,0,0:frameNumber,1])
                                worksheetData.write_column('E3', ParameterStats[17,0,0:frameNumber,1])
                                
                                workbook.close()
                                writer.save()
                            
                                
                                #returnAddress emailSubject
                            
                                sendemail = False
                                if sendemail == True:
                                    subject = "Processed I2 Data "+emailSubject
                                    X=ParameterStats[dictSet['xa1 ch'][0],0,0:frameNumber,1][dfBool]
                                    Y=ParameterStats[dictSet['ya1 ch'][0],0,0:frameNumber,1][dfBool]
                                    fit=PolyReg(X,Y,1)
                                    rate=-fit['coefs'][0]
                                    subjectPos=email_subject.lower().find('t')+1
                                    tempC=float(email_subject[subjectPos:subjectPos+2])
                                    tempK=tempC+273.15
                                    body = r"Based on your email subject the temperature is "+"{:g}".format(tempC)+" C. \n"
                                    body=body+r"Based on your video the rate is proportional to "+"{:.3g}".format(rate)+" dAU/dt. \n"
    
                                    body=body+r"For an  Arrhenius plot the x-coordinate (1/T where temp is in Kelvin) is "+"{:.4g}".format(1/tempK)+" 1/K. \n"
                                    body=body+r"For an  Arrhenius plot the y-coordinate (ln(rate)) is "+"{:.4g}".format(np.log(rate))
                                    
                                    sender_email = FROM_EMAIL
                                    #receiver_email = returnAddress
                                    receiver_email = "chem.sensor.up@gmail.com"
                                    password = FROM_PWD
                                    
                                    # Create a multipart message and set headers
                                    message = MIMEMultipart()
                                    message["From"] = sender_email
                                    message["To"] = receiver_email
                                    message["Subject"] = subject
                                    #message["Bcc"] = receiver_email  # Recommended for mass emails
                                    
                                    # Add body to email
                                    message.attach(MIMEText(body, "plain"))
                                    
                                    filename = "document.pdf"  # In same directory as script
                                    #filename = processedFileList[0]
                                    filename = file_path+"Data.xlsx"
                                    # Open PDF file in binary mode
                                    with open(filename, "rb") as attachment:
                                        # Add file as application/octet-stream
                                        # Email client can usually download this automatically as attachment
                                        part = MIMEBase("application", "octet-stream")
                                        part.set_payload(attachment.read())
                                    
                                    # Encode file in ASCII characters to send by email    
                                    encoders.encode_base64(part)
                                    
                                    # Add header as key/value pair to attachment part
                                    part.add_header(
                                        "Content-Disposition",
                                        #f"attachment; filename= {filename}",
                                        "attachment; filename={}".format(filename),
                                    )
                                    
                                    # Add attachment to message and convert message to string
                                    message.attach(part)
                                    
    
                                    filename = file_path+"Processed.mp4"
                                    # Open PDF file in binary mode
                                    with open(filename, "rb") as attachment:
                                        # Add file as application/octet-stream
                                        # Email client can usually download this automatically as attachment
                                        part = MIMEBase("application", "octet-stream")
                                        part.set_payload(attachment.read())
                                    
                                    # Encode file in ASCII characters to send by email    
                                    encoders.encode_base64(part)
                                    
                                    # Add header as key/value pair to attachment part
                                    part.add_header(
                                        "Content-Disposition",
                                        #f"attachment; filename= {filename}",
                                        "attachment; filename={}".format(filename),
                                    )
                                    
                                    # Add attachment to message and convert message to string
                                    message.attach(part)
                                    
                                    
                                    text = message.as_string()
                                    
                                    # Log in to server using secure context and send email
                                    context = ssl.create_default_context()
                                    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
                                        server.login(sender_email, password)
                                        server.sendmail(sender_email, receiver_email, text)
            
        # if stopFlag:
        #     break
        

saveSettings = input("Save current settings (Y/n)?")
if (saveSettings=="Y") | (saveSettings=="y"):
    root = tk.Tk()
    root.withdraw()
    settings_file_path = asksaveasfilename(initialdir='/home/sensor/',filetypes=[('settings files', '.set'),('all files', '.*')])
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
