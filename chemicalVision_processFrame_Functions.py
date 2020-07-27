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
        rotImage = cv2.warpPerspective(frame,Mrot,(2600,900))
        return(rotImage)
    else:
        return(np.array([0]))

def WhiteBalanceFrame(rotImage,frame,dictSet):
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
    return(rgbWBR,rotImage,frame)

def OpenCVComposite(sourceImage, targetImage,settingsWHS):
    scaleFactor=settingsWHS[2]/100
    imageScaled = cv2.resize(sourceImage, (int(sourceImage.shape[1]*scaleFactor),int(sourceImage.shape[0]*scaleFactor)), interpolation = cv2.INTER_AREA)
    targetImage[int(targetImage.shape[0]*settingsWHS[1]/100):int((targetImage.shape[0]*settingsWHS[1]/100)+imageScaled.shape[0]),int(targetImage.shape[1]*settingsWHS[0]/100):int((targetImage.shape[1]*settingsWHS[0]/100)+imageScaled.shape[1]),:]=imageScaled
    return targetImage

def SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=True):
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
        contourROI,contourArea=FindLargestContour(maskROI)
        contourMask=np.zeros((maskROI.shape),dtype='uint8')
        if contourArea>0:
            cv2.drawContours(contourMask,[contourROI],0,(255),-1)
        resMask = cv2.bitwise_and(maskROI,maskROI, mask= contourMask)
    else:
        resMask = maskROI
    resFrameROI = cv2.bitwise_and(rgbROI,rgbROI, mask= resMask)
    rgbROIsummary=cv2.meanStdDev(rgbROI,mask=resMask)
    hsvROIsummary=cv2.meanStdDev(hsvROI,mask=resMask)
    labROIsummary=cv2.meanStdDev(labROI,mask=resMask)
    absROIsummary=cv2.meanStdDev(absROI,mask=resMask)
    allROIsummary=np.concatenate((rgbROIsummary,hsvROIsummary,labROIsummary,absROIsummary),axis=1)
    return(allROIsummary[0,:,0],allROIsummary[1,:,0],resMask,resFrameROI)

def ProcessOneFrame(frame,dictSet,displayFrame,roiList=["WB1","WB2","WB3","RO1","RO2","RO3"]):
    frameStats=np.zeros((32,6,len(roiList)))    
    #displayWidth=dictSet['dsp wh'][0]
    #displayHeight=dictSet['dsp wh'][1]
    flgFindReference=dictSet['flg rf'][0]
    #displayFrame = np.zeros((displayHeight, displayWidth, 3), np.uint8)
    img=np.copy(frame)
    if flgFindReference:
        rotImage = RegisterImage(frame,img,dictSet)
        skipFrame=False
        if rotImage.size==1:
            skipFrame=True
    else:
        rotImage = np.copy(frame)
        skipFrame=False
    if skipFrame==False:
        if dictSet['flg wb'][0]==1:
            rgbWBR,rotImage,frame=WhiteBalanceFrame(rotImage,frame,dictSet)
            if dictSet['flg di'][0]==1:
                cv2.imshow("WBR",rgbWBR)
            hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
            maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
            rgbWBRsummary=cv2.meanStdDev(rgbWBR,mask=maskWBR)
            resFrameWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)
        #for row, displayColor, channel in zip([0,1,2], [(0, 0, 128),(0, 128, 0),(128, 25, 25)], [2,1,0]):              
        #    ParameterStats[row,0,frameNumber,0],ParameterStats[row,1,frameNumber,0],ParameterStats[row,2,frameNumber,0]=ip.OpenCVDisplayedHistogram(rgbWBR,channel,maskWBR,256,0,255,displayWidth/2,5+(row*displayHeight/14)+(row*6),256,(displayHeight/16),displayFrame,displayColor,5,False)
        for roiSetName,roiNumber in zip(roiList,range(len(roiList))):
            valSummary,stdSummary,resMask,resRGB=SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=True)
            frameStats[0:12,0,roiNumber]=valSummary
            frameStats[0:12,1,roiNumber]=stdSummary
            area=cv2.countNonZero(resMask)
            frameStats[15,0,roiNumber]=area
            
            #check to see if these are calculated correctly in FindLargestContour
            #boundingRectangle=cv2.minAreaRect(resMask)
            #frameStats[16,0,roiNumber]=boundingRectangle[1][0]
            #frameStats[17,0,roiNumber]=boundingRectangle[1][1]
            if dictSet['flg di'][0]==1:
                cv2.imshow(roiSetName,resRGB)
            if dictSet[roiSetName+' ds'][2]!=0:
                displayFrame=OpenCVComposite(resRGB, displayFrame, dictSet[roiSetName+' ds'])
    return frameStats,displayFrame
   
cap = cv2.VideoCapture(video_file_path)
TotalFrames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
ParameterStats=np.zeros((32,6,TotalFrames*2,5))
#ParameterStats Map
#1st dimension 0 to 14: labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]
#1st dimension 15 to 17: labels=["RO1area","RO1BoundingRecHeight","RO1BoundingRecWidth"
#1st dimension 29 to 31: labels=["frameRate","frameNumber","frameTime"]
#2nd dimension 0 to 5: labels=["mean","stdev","dominant"]
#3rd dimension frame number
#4th dimension 0 to 4: labels=["WBR","RO1","RO2","RO3"]
frameNumber=0

displayWidth=dictSet['dsp wh'][0]
displayHeight=dictSet['dsp wh'][1]
displayFrame = np.zeros((displayHeight, displayWidth, 3), np.uint8)
ret, frame = cap.read()  
if dictSet['PRE ds'][2]!=0:
    displayFrame=OpenCVComposite(frame, displayFrame,dictSet['PRE ds'])
frameStats,displayFrame = ProcessOneFrame(frame,dictSet,displayFrame)
signal=frameStats[9,0,3]-frameStats[11,0,3]
print(signal)
if dictSet['PST ds'][2]!=0:
    displayFrame=OpenCVComposite(frame, displayFrame,dictSet['PST ds'])
cv2.imshow('Display', displayFrame)


# def ProcessOneFrame(frame,dictSet):    
#     displayWidth=dictSet['dsp wh'][0]
#     displayHeight=dictSet['dsp wh'][1]
#     flgFindReference=dictSet['flg rf'][0]
#     displayFrame = np.zeros((displayHeight, displayWidth, 3), np.uint8)
#     img=np.copy(frame)
#     if flgFindReference:
#         rotImage = RegisterImage(frame,img,dictSet)
#         skipFrame=False
#         if rotImage.size==1:
#             skipFrame=True
#     else:
#         rotImage = np.copy(frame)
#         skipFrame=False
#     if skipFrame==False:
#         rgbWBR,rotImage,frame=WhiteBalanceFrame(rotImage,frame,dictSet)
#         hsvWBR = cv2.cvtColor(rgbWBR, cv2.COLOR_BGR2HSV)
#         maskWBR = cv2.inRange(hsvWBR, np.array(dictSet['WBR ll']), np.array(dictSet['WBR ul']))
#         rgbWBRsummary=cv2.meanStdDev(rgbWBR,mask=maskWBR)
#         for row, displayColor, channel in zip([0,1,2], [(0, 0, 128),(0, 128, 0),(128, 25, 25)], [2,1,0]):              
#             ParameterStats[row,0,frameNumber,0],ParameterStats[row,1,frameNumber,0],ParameterStats[row,2,frameNumber,0]=ip.OpenCVDisplayedHistogram(rgbWBR,channel,maskWBR,256,0,255,displayWidth/2,5+(row*displayHeight/14)+(row*6),256,(displayHeight/16),displayFrame,displayColor,5,False)
#         resFrameWBR = cv2.bitwise_and(rgbWBR,rgbWBR, mask= maskWBR)

#         SummarizeROI(rotImage,roiSetName,dictSet,connectedOnly=True)

#         rgbRO1 = rotImage[dictSet['RO1 xy'][1]:dictSet['RO1 xy'][1]+dictSet['RO1 wh'][1], dictSet['RO1 xy'][0]:dictSet['RO1 xy'][0]+dictSet['RO1 wh'][0]]
#         hsvRO1 = cv2.cvtColor(rgbRO1, cv2.COLOR_BGR2HSV)
#         hsvRO1[:,:,0]=ip.ShiftHOriginToValue(hsvRO1[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
#         labRO1 = cv2.cvtColor(rgbRO1, cv2.COLOR_BGR2LAB)
#         logsrgbRO1=cv2.LUT(rgbRO1, linLUTabs)*64
#         maskRO1 = cv2.inRange(hsvRO1, np.array(dictSet['RO1 ll']), np.array(dictSet['RO1 ul']))
#         rgbRO1summary=cv2.meanStdDev(rgbRO1,mask=maskRO1)

#         rgbRO2 = rotImage[dictSet['RO2 xy'][1]:dictSet['RO2 xy'][1]+dictSet['RO2 wh'][1], dictSet['RO2 xy'][0]:dictSet['RO2 xy'][0]+dictSet['RO2 wh'][0]]
#         hsvRO2 = cv2.cvtColor(rgbRO2, cv2.COLOR_BGR2HSV)
#         hsvRO2[:,:,0]=ip.ShiftHOriginToValue(hsvRO2[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
#         labRO2 = cv2.cvtColor(rgbRO2, cv2.COLOR_BGR2LAB)
#         logsrgbRO2=cv2.LUT(rgbRO2, linLUTabs)*64
#         maskRO2 = cv2.inRange(hsvRO2, np.array(dictSet['RO2 ll']), np.array(dictSet['RO2 ul']))
#         rgbRO2summary=cv2.meanStdDev(rgbRO2,mask=maskRO2)

#         rgbRO3 = rotImage[dictSet['RO3 xy'][1]:dictSet['RO3 xy'][1]+dictSet['RO3 wh'][1], dictSet['RO3 xy'][0]:dictSet['RO3 xy'][0]+dictSet['RO3 wh'][0]]
#         hsvRO3 = cv2.cvtColor(rgbRO3, cv2.COLOR_BGR2HSV)
#         hsvRO3[:,:,0]=ip.ShiftHOriginToValue(hsvRO2[:,:,0],dictSet['hue lo'][0],dictSet['hue lo'][1])
#         labRO3 = cv2.cvtColor(rgbRO3, cv2.COLOR_BGR2LAB)
#         logsrgbRO3=cv2.LUT(rgbRO3, linLUTabs)*64
#         maskRO3 = cv2.inRange(hsvRO3, np.array(dictSet['RO3 ll']), np.array(dictSet['RO3 ul']))
#         rgbRO3summary=cv2.meanStdDev(rgbRO3,mask=maskRO3)

#         if float(float(cv2.__version__[0])+float(cv2.__version__[2])/10)>=4:
#             contours,hierarchy = cv2.findContours(maskRO1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         else:
#             image,contours,hierarchy = cv2.findContours(maskRO1,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#         contourMask=np.zeros((maskRO1.shape),dtype='uint8')
#         if len(contours)>=1:
#             MaxRO1Area=0
#             ContourIndex=0
#             LargestContour=0
#             for contour in contours:
#                 area=cv2.contourArea(contour)
#                 if area>MaxRO1Area:
#                     MaxRO1Area=area
#                     LargestContour=ContourIndex
#                 ContourIndex=ContourIndex+1
#             outerRO1Contour=contours[LargestContour]
#             boundingRectangle=cv2.minAreaRect(outerRO1Contour)
#             #cv2.drawContours(resFrameRO1,[outerRO1Contour],0,(0,255,0),2)
#             cv2.drawContours(contourMask,[outerRO1Contour],0,(255),-1)
#             resMask = cv2.bitwise_and(maskRO1,maskRO1, mask= contourMask)
#             resFrameRO1 = cv2.bitwise_and(rgbRO1,rgbRO1, mask= resMask)
#             ParameterStats[15,0,frameNumber,1]=MaxRO1Area*cmPerPixel*cmPerPixel
#             ParameterStats[16,0,frameNumber,1]=boundingRectangle[1][0]
#             ParameterStats[17,0,frameNumber,1]=boundingRectangle[1][1]
#             ParameterStats[0,0,frameNumber,2]=rgbRO2summary[0][2]
#             ParameterStats[1,0,frameNumber,2]=rgbRO2summary[0][1]
#             ParameterStats[2,0,frameNumber,2]=rgbRO2summary[0][0]
#             ParameterStats[0,1,frameNumber,2]=rgbRO2summary[0][2]
#             ParameterStats[1,1,frameNumber,2]=rgbRO2summary[0][1]
#             ParameterStats[2,1,frameNumber,2]=rgbRO2summary[0][0]
#             ParameterStats[0,0,frameNumber,3]=rgbRO3summary[0][2]
#             ParameterStats[1,0,frameNumber,3]=rgbRO3summary[0][1]
#             ParameterStats[2,0,frameNumber,3]=rgbRO3summary[0][0]
#             ParameterStats[0,1,frameNumber,3]=rgbRO3summary[0][2]
#             ParameterStats[1,1,frameNumber,3]=rgbRO3summary[0][1]
#             ParameterStats[2,1,frameNumber,3]=rgbRO3summary[0][0]
#             ParameterStats[31,0,frameNumber,:]=currentTime
#             ParameterStats[30,0,frameNumber,:]=currentFrame
#             ParameterStats[29,0,frameNumber,:]=frameRate
#             #Analytical signal should be in channel 13 of parameter stats B absorbance -R absorbance for I2
#             labels=["R","G","B","H","S","V","L","a","b","Ra","Ga","Ba","Ga-Ra","Ba-Ra","Ga-Ba"]
#             #ParameterStats[12,0,frameNumber,:]=ParameterStats[10,0,frameNumber,:]-ParameterStats[9,0,frameNumber,:]
#             #ParameterStats[13,0,frameNumber,:]=ParameterStats[11,0,frameNumber,:]-ParameterStats[9,0,frameNumber,:]
#             #ParameterStats[14,0,frameNumber,:]=ParameterStats[10,0,frameNumber,:]-ParameterStats[11,0,frameNumber,:]
            
#             RO1Scale=max(resFrameRO1.shape[1]/(displayWidth/4),resFrameRO1.shape[0]/(displayHeight/4))
#             RO1ImageScale = cv2.resize(resFrameRO1, (int(resFrameRO1.shape[1]/RO1Scale),int(resFrameRO1.shape[0]/RO1Scale)), interpolation = cv2.INTER_AREA)
#             WBRScale=max(resFrameWBR.shape[1]/(displayWidth/4),resFrameWBR.shape[0]/(displayHeight/4))
#             WBRImageScale = cv2.resize(resFrameWBR, (int(resFrameWBR.shape[1]/WBRScale),int(resFrameWBR.shape[0]/WBRScale)), interpolation = cv2.INTER_AREA)
#             resFrameRO2=rgbRO2
#             RO2Scale=max(resFrameRO2.shape[1]/(displayWidth/4),resFrameRO2.shape[0]/(displayHeight/4))
#             RO2ImageScale = cv2.resize(resFrameRO2, (int(resFrameRO2.shape[1]/RO2Scale),int(resFrameRO2.shape[0]/RO2Scale)), interpolation = cv2.INTER_AREA)
#             resFrameRO3=rotImage
#             RO3Scale=max(resFrameRO3.shape[1]/(displayWidth/4),resFrameRO3.shape[0]/(displayHeight/4))
#             RO3ImageScale = cv2.resize(resFrameRO3, (int(resFrameRO3.shape[1]/RO3Scale),int(resFrameRO3.shape[0]/RO3Scale)), interpolation = cv2.INTER_AREA)
#             displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+RO1ImageScale.shape[0]),0:RO1ImageScale.shape[1],:]=RO1ImageScale
#             displayFrame[int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4):int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4)+RO2ImageScale.shape[0],0:RO2ImageScale.shape[1],:]=RO2ImageScale
#             displayFrame[int(displayFrame.shape[0]/2):int((displayFrame.shape[0]/2)+WBRImageScale.shape[0]) , int(displayFrame.shape[0]/2):int(displayFrame.shape[0]/2)+WBRImageScale.shape[1],:]=WBRImageScale
#             displayFrame[int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4):int(displayFrame.shape[0]/2)+int(displayFrame.shape[0]/4)+RO3ImageScale.shape[0] , int(displayFrame.shape[0]/2):int(displayFrame.shape[0]/2)+RO3ImageScale.shape[1],:]=RO3ImageScale
#             inputImages= [rgbRO1,rgbRO1,rgbRO1,hsvRO1,hsvRO1,hsvRO1,labRO1,labRO1,labRO1,logsrgbRO1,logsrgbRO1,logsrgbRO1]
#             for row, displayColor, inputImage, channel, label in zip(rows, displayColors, inputImages, channels,labels):              
#                 #ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=OpenCVDisplayedHistogram(inputImage,channel,maskRO1,256,0,255,displayWidth/2,5+(row*displayHeight/10)+(row*5),256,(displayHeight/12),displayFrame,displayColor,5,True,label)
#                 ParameterStats[row,0,frameNumber,1],ParameterStats[row,1,frameNumber,1],ParameterStats[row,2,frameNumber,1]=ip.OpenCVDisplayedHistogram(inputImage,channel,resMask,256,0,255,displayWidth/2,5+(row*displayHeight/14)+(row*6),256,(displayHeight/16),displayFrame,displayColor,5,True,label)
#             ParameterStats[12,0,frameNumber,1]=ParameterStats[10,0,frameNumber,1]-ParameterStats[9,0,frameNumber,1]
#             ParameterStats[13,0,frameNumber,1]=ParameterStats[11,0,frameNumber,1]-ParameterStats[9,0,frameNumber,1]
#             ParameterStats[14,0,frameNumber,1]=ParameterStats[10,0,frameNumber,1]-ParameterStats[11,0,frameNumber,1]

#             if frameNumber>=2:
#                 if dictSet['a1x sc'][0]==0:
#                     xMin=dictSet['a1x sc'][1]
#                     xMax=dictSet['a1x sc'][2]
#                 else:
#                     xMin=None
#                     xMax=None          
#                 if dictSet['a1y sc'][0]==0:
#                     yMin=dictSet['a1y sc'][1]
#                     yMax=dictSet['a1y sc'][2]
#                 else:
#                     yMin=None
#                     yMax=None     
#                 ip.OpenCVDisplayedScatter(displayFrame, ParameterStats[dictSet['a1x ch'][0],dictSet['a1x ch'][1],0:frameNumber,dictSet['a1x ch'][2]],ParameterStats[dictSet['a1y ch'][0],dictSet['a1y ch'][1],0:frameNumber,dictSet['a1y ch'][2]],dictSet['pl1 xy'][0],dictSet['pl1 xy'][1],dictSet['pl1 wh'][0],dictSet['pl1 wh'][1],(255,255,255), 1, ydataRangemin=yMin, ydataRangemax=yMax,xdataRangemin=xMin, xdataRangemax=xMax)
#                 if dictSet['a2x sc'][0]==0:
#                     xMin=dictSet['a2x sc'][1]
#                     xMax=dictSet['a2x sc'][2]
#                 else:
#                     xMin=None
#                     xMax=None          
#                 if dictSet['a2y sc'][0]==0:
#                     yMin=dictSet['a2y sc'][1]
#                     yMax=dictSet['a2y sc'][2]
#                 else:
#                     yMin=None
#                     yMax=None     
#                 ip.OpenCVDisplayedScatter(displayFrame, ParameterStats[dictSet['a2x ch'][0],dictSet['a2x ch'][1],0:frameNumber,dictSet['a2x ch'][2]],ParameterStats[dictSet['a2y ch'][0],dictSet['a2y ch'][1],0:frameNumber,dictSet['a2y ch'][2]],dictSet['pl2 xy'][0],dictSet['pl2 xy'][1],dictSet['pl2 wh'][0],dictSet['pl2 wh'][1],(255,255,255), 1, ydataRangemin=yMin, ydataRangemax=yMax,xdataRangemin=xMin, xdataRangemax=xMax)
#             if ActiveState=="Process":
#                 frameNumber=frameNumber+1
    
#     #single frame process ends here