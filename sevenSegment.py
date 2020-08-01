# -*- coding: utf-8 -*-
"""
Created on Thu Jul 30 17:07:44 2020

@author: cantr
"""


cap2 = cv2.VideoCapture(int(dictSet['CM2 en'][0]))

ret, frame = cap2.read() 
frameCrop2=frame2[dictSet['CM2 xy'][0]:dictSet['CM2 xy'][0]+dictSet['CM2 wh'][0],dictSet['CM2 xy'][1]:dictSet['CM2 xy'][1]+dictSet['CM2 wh'][1],:]
hsvFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

dictSet['bal ll']=[0,200,200]
dictSet['bal ul']=[255,255,255]
massMask = cv2.inRange(hsvFrame, np.array([dictSet['bal ll']]), np.array(dictSet['bal ul'])) 
cv2.imshow('massMask',massMask)
outerBoxContour,boxArea,boxBoundingRectangle=FindLargestContour(massMask)
#cv2.drawContours(frame,[outerBoxContour],0,(0,255,0),2)
ptsFound = cv2.boxPoints(boxBoundingRectangle)

#epsilon = 0.1*cv2.arcLength(outerBoxContour,True)
#ptsFound = cv2.approxPolyDP(outerBoxContour,epsilon,True)

#box = np.int0(ptsFound)
#cv2.drawContours(frame,[box],0,(0,0,255),2)
#cv2.drawContours(frame,[ptsFound],0,(0,0,255),2)


dictSet['cl1 xy']=[300,100]
dictSet['cl2 xy']=[0,100]
dictSet['cl3 xy']=[0,0]
dictSet['cl4 xy']=[300,0]
orientation=1
ptsCard = np.float32([[dictSet['cl1 xy'][0],dictSet['cl1 xy'][1]],[dictSet['cl2 xy'][0],dictSet['cl2 xy'][1]],[dictSet['cl3 xy'][0],dictSet['cl3 xy'][1]],[dictSet['cl4 xy'][0],dictSet['cl4 xy'][1]]])
ptsImage = np.float32([[135,220],[765,220],[135,1095],[765,1095]]) 
if ptsFound.shape[0]==4:
    if orientation==1:
        ptsImage[0,0]=ptsFound[0,0]
        ptsImage[0,1]=ptsFound[0,1]
        ptsImage[1,0]=ptsFound[1,0]
        ptsImage[1,1]=ptsFound[1,1]
    else:
        ptsImage[1,0]=ptsFound[0,0]
        ptsImage[1,1]=ptsFound[0,1]
        ptsImage[0,0]=ptsFound[1,0]
        ptsImage[0,1]=ptsFound[1,1]
    if orientation==1:
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
    rotImage = cv2.warpPerspective(frame,Mrot,(300,100))


cv2.imshow('frame',frame)
rotImage = cv2.cvtColor(rotImage, cv2.COLOR_BGR2GRAY)
cv2.imshow('rotImage',rotImage)
digitHeight=85
digitWidth=45
imageStart=55
digits=np.zeros((digitHeight, digitWidth, 5), np.uint8)
for digit in range(5): 
    digits[:,:,digit]=rotImage[10:95,imageStart+(digit*digitWidth):imageStart+((digit+1)*digitWidth)]    
    cv2.imshow(str(digit),digits[:,:,digit])
    digitImage=digits[:,:,digit]
    dictSet['seg ll']=[0]
    dictSet['seg ul']=[150]
    digitMask = cv2.inRange(digitImage, np.array([dictSet['seg ll']]), np.array(dictSet['seg ul'])) 
    cv2.imshow('digitMask',digitMask)


edges = cv2.Canny(frame2,150,200)
cv2.imshow('Edges',edges)

cap2.release()
