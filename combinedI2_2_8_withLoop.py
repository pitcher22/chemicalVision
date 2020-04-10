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
    ltArrow=81
    upArrow=82
    rtArrow=83
    dnArrow=84
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
    
runMode = input("Are you downloading from an email? (Y/n): ")
if (runMode == "y")|(runMode == "Y"):
    FROM_EMAIL = input("Please enter your email address: ")
    FROM_PWD = input("Please enter your password: ")
else:
    video_file_path = askopenfilename(initialdir=os.getcwd(),filetypes=[('settings files', '.MOV'),('all files', '.*')])

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
                                    fileName=filePathEmail+'/'+ dateName + '#'+ returnAddress +'#'+ email_subject+'.MOV'
                                    
                                if versionOS=='W':
                                    bad_chars=[":","<",">"]
                                    for c in bad_chars : 
                                        email_subject = email_subject.replace(c, ' ') 
                                    for c in bad_chars : 
                                        dateName = dateName.replace(c, '_') 
                                    for c in bad_chars : 
                                        returnAddress = returnAddress.replace(c, '')
                                    fileName=filePathEmail+'\\'+ dateName + '#'+ returnAddress +'#'+ email_subject+'.MOV'
                                    
                                if versionOS=='M':
                                    bad_chars=[":","<",">"]
                                    for c in bad_chars : 
                                        email_subject = email_subject.replace(c, ' ') 
                                    for c in bad_chars : 
                                        dateName = dateName.replace(c, '_') 
                                    for c in bad_chars : 
                                        returnAddress = returnAddress.replace(c, '') 
                                    fileName=filePathEmail+'/'+ dateName + '#'+ returnAddress +'#'+ email_subject+'.MOV'
                                    
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
                                        outFileName=filePathEmail+'/Processed/'+ dateName + '#' + email_subject +'#'+'Processed.mp4'
                                    if versionOS=='W':
                                        outFileName=filePathEmail+'\\Processed\\'+ dateName + '#'+ email_subject +'#'+'Processed.mp4'
                                    if versionOS=='M':
                                        outFileName=filePathEmail+'/Processed/'+ dateName + '#' + email_subject +'#'+'Processed.mp4'
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
                                            rotImage=ip.OpenCVRebalanceImage(rotImage,rfactor,gfactor,bfactor)
                                            frame=ip.OpenCVRebalanceImage(frame,rfactor,gfactor,bfactor)
                                
                                        rgbWBR[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]] = rotImage[dictSet['WB1 xy'][1]:dictSet['WB1 xy'][1]+dictSet['WB1 wh'][1], dictSet['WB1 xy'][0]:dictSet['WB1 xy'][0]+dictSet['WB1 wh'][0]]
                                        rgbWBR[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]] = rotImage[dictSet['WB2 xy'][1]:dictSet['WB2 xy'][1]+dictSet['WB2 wh'][1], dictSet['WB2 xy'][0]:dictSet['WB2 xy'][0]+dictSet['WB2 wh'][0]]
                                        rgbWBR[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]] = rotImage[dictSet['WB3 xy'][1]:dictSet['WB3 xy'][1]+dictSet['WB3 wh'][1], dictSet['WB3 xy'][0]:dictSet['WB3 xy'][0]+dictSet['WB3 wh'][0]]
                                        hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
                                        maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
                                        for row, displayColor, channel in zip([0,1,2], [(0, 0, 128),(0, 128, 0),(128, 25, 25)], [2,1,0]):              
                                            #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskRO1,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/10)+(row*5),256,(DisplayHeight/12),displayFrame,displayColor,5,True,label)
                                            ParameterStats[row,0,frameNumber,0],ParameterStats[row,1,frameNumber,0],ParameterStats[row,2,frameNumber,0]=ip.OpenCVDisplayedHistogram(rgbWBR,channel,maskWBR,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/14)+(row*6),256,(DisplayHeight/16),displayFrame,displayColor,5,False)
                                
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
                            
                                        hsvRO1[:,:,0]=ip.ShiftHOriginToValue(hsvRO1[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
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
                                                ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=ip.OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,DisplayWidth/2,5+(row*DisplayHeight/14)+(row*6),256,(DisplayHeight/16),displayFrame,displayColor,5,True,label)
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
                                            ip.OpenCVPutText(displayFrame, setting, (DisplayWidth-(parmWidth*5),parmHeight*(setRow+1)), setColor)
                                            #cv2.putText(displayFrame,setting,(DisplayWidth-(parmWidth*5),parmHeight*(setRow+1)), font, fontScale,setColor,1,cv2.LINE_AA)
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
                                        moveSettings = ' to change selected setting'

                                        helpInfo = {quitKey: '"q"', pauseKey: '"p"', rrBig: '"j"', rrSmall: '"h"', ffSmall: '"k"', ffBig: '"l"', showInfo: '"i"', toggleSettingsKey: '"t"', increaseSettingValue: '"+"', decreaseSettingValue: '"-"', moveSettings: '"wasd"'}
                                        fontSpacing = int(1300)
                                        
                                        textXlocation = int(dictSet['dsp wh'][0] * (2/fontSpacing))
                                        textYlocation = DisplayHeight - int(dictSet['dsp wh'][0] * (8/fontSpacing))
                                        
                                        for key in helpInfo.keys():
                                            cv2.putText(displayFrame, helpInfo[key], (textXlocation,textYlocation), font, fontScale,(255,255,255),1,cv2.LINE_AA)
                                            textXlocation = textXlocation + (5*len(helpInfo[key])) + int(dictSet['dsp wh'][0] * (5/fontSpacing))
                                            cv2.putText(displayFrame, key, (textXlocation, textYlocation), font, fontScale,(0,255,0),1,cv2.LINE_AA)
                                            textXlocation = textXlocation + (5*len(key)) + int(dictSet['dsp wh'][0] * (5/fontSpacing))
                                    
                                    else:
                                        cv2.putText(displayFrame,'type "?" for hotkeys', (2,DisplayHeight-8),font, fontScale,(255,255,255),1,cv2.LINE_AA)
                                        
                                    #cv2.imshow('Result', img)
                                    if ActiveState=="Pause":
                                        cv2.rectangle(displayFrame, (int(DisplayWidth*0.425/2),int(DisplayHeight*0.2/2)), (int(DisplayWidth*0.475/2),int(DisplayHeight*0.8/2)), (255,255,255),-1)
                                        cv2.rectangle(displayFrame, (int(DisplayWidth*0.525/2),int(DisplayHeight*0.2/2)), (int(DisplayWidth*0.575/2),int(DisplayHeight*0.8/2)), (255,255,255),-1)
                                
                                    cv2.imshow('Display', displayFrame)
                                    if dictSet['flg di'][0]==1:
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
                            #                OpenCVDisplayedScatter(frameImageScale, ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1],ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1],int(dim1/10),int(dim2/10),int(dim1/4),int(dim2/3),(255,255,255),ydataRangemin=10, ydataRangemax=45,xdataRangemin=xMin, xdataRangemax=xMax)
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
                                        if dictSet['flg di'][0]==1:
                                            dictSet['flg di'][0]=0
                                        else:
                                            dictSet['flg di'][0]=1
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
                                    outExcelFileName=filePathEmail+'/Processed/'+ dateName + '#' + email_subject +'#'+'Data.xlsx'
                                if versionOS=='W':
                                    outExcelFileName=filePathEmail+'\\Processed\\'+ dateName + '#'+ email_subject +'#'+'Data.xlsx'
                                if versionOS=='M':
                                    outExcelFileName=filePathEmail+'/Processed/'+ dateName + '#' + email_subject +'#'+'Data.xlsx'
                                writer = pd.ExcelWriter(outExcelFileName, engine='xlsxwriter')
                                workbook  = writer.book
                                minArea=2
                                #maxArea=50000
                                maxSignal=40
                                minSignal=16
                                dfMinArea=ParameterStats[15,0,0:frameNumber,1]>minArea
                                dfHeightRange=(ParameterStats[16,0,0:frameNumber,1]>np.mean(ParameterStats[16,0,0:frameNumber,1][dfMinArea])*0.95) & (ParameterStats[16,0,0:frameNumber,1]<np.mean(ParameterStats[16,0,0:frameNumber,1][dfMinArea])*1.05)
                                #dfBool=dfMinArea & dfHeightRange
                                dfBool=(dfMinArea) & (ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1]<=maxSignal) & (ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1]>=minSignal)

                                worksheetFit = workbook.add_worksheet("Fit")
                                worksheetFit.write('A1', 'Time')
                                worksheetFit.write('B1', labels[dictSet['a1y ch'][0]])
                                worksheetFit.write('C1', 'Time (linear range)')
                                worksheetFit.write('D1', labels[dictSet['a1y ch'][0]]+' (linear range)')
                                worksheetFit.write_column('A2',ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1])
                                worksheetFit.write_column('B2',ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1])
                                worksheetFit.write_column('C2',ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1][dfBool])
                                worksheetFit.write_column('D2',ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1][dfBool])
                                numEntries=ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1][dfBool].size
                                numIndex=str(numEntries+1)
                                worksheetFit.write_array_formula('I3:J5', '{=LINEST(D2:D'+numIndex+',C2:C'+numIndex+',TRUE,TRUE)}')
                                worksheetFit.write('I2', 'Slope')
                                worksheetFit.write('J2', 'Intercept')
                                worksheetFit.write('H3', 'coefs')
                                worksheetFit.write('H4', 'errors')
                                worksheetFit.write('H5', 'r2, sy')
                                chart1 = workbook.add_chart({'type': 'scatter'})
                                numAllEntries=ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1].size
                                chart1.add_series({
                                    'name': labels[dictSet['a1y ch'][0]]+' linear',
                                    'categories': ["Fit", 1, 2, 1+numEntries-1, 2],
                                    'values': ["Fit", 1, 3, 1+numEntries-1, 3],
                                    'trendline': {
                                        'type': 'linear',
                                        'display_equation': True,
                                        'line': {
                                        'color': 'black',
                                        'width': 2,
                                        },
                                        'forward': ParameterStats[dictSet['a1x ch'][0],0,frameNumber-1,1],
                                        'backward': ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1][dfBool][0],
                                    },
                                    'marker': {
                                        'type': 'circle',
                                        'size': 8,
                                        'fill':   {'color': '#a66fb5'},
                                        'border':   {'color': '#a66fb5'},
                                    },
                                })
                                chart1.add_series({
                                    'name': labels[dictSet['a1y ch'][0]]+' all',
                                    'categories': ["Fit", 1, 0, 1+numAllEntries-1, 0],
                                    'values': ["Fit", 1, 1, 1+numAllEntries-1, 1],
                                    'marker': {
                                            'type': 'circle',
                                            'size': 4,
                                            'fill':   {'color': '#490648'},
                                            'border':   {'color': '#490648'},
                                    },
                                })
                            
                                #chart1.set_title ({'name': labels[dictSet['a1y ch'][0]]+' Change'})
                                if (ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1].size!=0) and (ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1].size!=0):
                                    chart1.set_x_axis({
                                            'name': 'Time (seconds)',
                                            'min': np.min(np.floor(ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1])),
                                            'max': np.max(np.ceil(ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1]))
                                            })
                                    chart1.set_y_axis({
                                            'name': 'Signal',
                                            'min': np.min(np.floor(ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1])),
                                            'max': np.max(np.ceil(ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1])),
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
                                    X=ParameterStats[dictSet['a1x ch'][0],0,0:frameNumber,1][dfBool]
                                    Y=ParameterStats[dictSet['a1y ch'][0],0,0:frameNumber,1][dfBool]
                                    fit=da.PolyReg(X,Y,1)
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