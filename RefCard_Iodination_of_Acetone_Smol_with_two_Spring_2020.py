import numpy as np
import cv2
#import qrcode
#from PIL import Image
font = cv2.FONT_HERSHEY_SIMPLEX

settingString='''
{'ROI xy': array([ 30, 300]), 'CAM exp': array([ 1, 50]), 'WBR xy': array([810, 370]), 'yax ch': array([4]), 'xax sc': array([1, 0, 0]), 'box ul': array([100, 255, 255]), 'WBR sc': array([0]), 'yax sc': array([1, 0, 0]), 'CAM bcs': array([128, 128, 128]), 'hue lo': array([180., 150.]), 'c34 ul': array([ 40, 255, 255]), 'c12 ll': array([140,  20,  40]), 'CAM foc': array([ 1, 60]), 'ROI wh': array([920, 590]), 'xax ch': array([31]), 'c12 ul': array([160, 255, 255]), 'box ll': array([80, 20, 40]), 'ROI ll': array([ 0, 70,  0]), 'c34 ll': array([20, 60, 40]), 'WBR ul': array([255,  30, 255]), 'WBR wh': array([270, 450]), 'ROI ul': array([180, 255, 255]), 'WBR ll': array([0, 0, 0])}
'''

#qrimg = qrcode.make(settingString)
#open_qr_image = np.array(qrimg.convert('RGB'))
circler=55
#open_qr_image = cv2.resize(open_qr_image, (400, 400))
paperWidth=1200
paperHeight=1800
shrinkFactor=0.1
recFraction=0.20
ReferenceImage = np.full((paperWidth, paperHeight, 3), 255,np.uint8)

#cv2.circle(ReferenceImage,(200,200), circler*3, (255,255,0), -1)
#cv2.circle(ReferenceImage,(1600,200), circler*3, (0,255,255), -1)
#cv2.circle(ReferenceImage,(1600,1000), circler*3, (255,0,255), -1)
#cv2.circle(ReferenceImage,(200,1000), circler*3, (255,0,255), -1)
#
#cv2.circle(ReferenceImage,(1600,200), circler*2, (255,0,255), -1)
#cv2.circle(ReferenceImage,(1600,1000), circler*2, (255,255,0), -1)
#cv2.circle(ReferenceImage,(200,1000), circler*2, (0,255,255), -1)

#cv2.rectangle(ReferenceImage,(450,450), (1100,750), (255,128,128), 5)

#cv2.rectangle(ReferenceImage,(int(0+(paperHeight*shrinkFactor)),int(0+(paperWidth*shrinkFactor))), (900,300), (255,255,0), -1)

#left card
cv2.rectangle(ReferenceImage, (30,30), (240,1200), (255,255,0), -1) #left side of rectangle
cv2.rectangle(ReferenceImage, (30, 990), (870,1200), (255,255,0), -1) #bottom rectangle in digital view
cv2.rectangle(ReferenceImage, (660,30), (870,1200), (255,255,0), -1) #right side of rectangle

#duplicate right card
cv2.rectangle(ReferenceImage, (30 + 900, 30), (240 + 900,1200), (255,255,0), -1) #left side of rectangle
cv2.rectangle(ReferenceImage, (30 + 900, 990), (870 + 900,1200), (255,255,0), -1) #bottom rectangle in digital view
cv2.rectangle(ReferenceImage, (660 + 900, 30), (870 + 900,1200), (255,255,0), -1) #right side of rectangle


#cv2.rectangle(ReferenceImage,(int(0+(paperHeight*shrinkFactor)),int(0+(400))), (int(paperHeight-(paperHeight*shrinkFactor)),int((paperWidth*recFraction))+(int(paperWidth*shrinkFactor))), (255,255,0), -1)
#cv2.rectangle(ReferenceImage,(int(0+(paperHeight*shrinkFactor)),int(0+(paperWidth*shrinkFactor))), (int(paperHeight-(paperHeight*shrinkFactor)),int((paperWidth*recFraction))+(int(paperWidth*shrinkFactor))), (255,255,0), -1)
#cv2.rectangle(ReferenceImage,(0,800), (1800,1200), (255,255,0), -1)
#cv2.rectangle(ReferenceImage,(1400,0), (1800,1200), (255,255,0), -1)
#cv2.rectangle(ReferenceImage,(50,50), (1750,1150), (255,255,0), 50)

#magentaCircles of left card
cv2.circle(ReferenceImage,(765,1095), circler, (255,0,255), -1) 
cv2.circle(ReferenceImage,(765,220), circler, (255,0,255), -1) 

#yellowCircles of left card
cv2.circle(ReferenceImage,(135,220), circler, (0,255,255), -1) 
cv2.circle(ReferenceImage,(135,1095), circler, (0,255,255), -1) 

#magentaCircles of right card
cv2.circle(ReferenceImage,(765 + 900,1095), circler, (255,0,255), -1) 
cv2.circle(ReferenceImage,(765 + 900,220), circler, (255,0,255), -1) 

#yellowCircles of right card
cv2.circle(ReferenceImage,(135 + 900,220), circler, (0,255,255), -1) 
cv2.circle(ReferenceImage,(135 + 900,1095), circler, (0,255,255), -1) 


#ReferenceImage[395:395+open_qr_image.shape[0],1090:1090+open_qr_image.shape[1],:]=open_qr_image
#cv2.putText(ReferenceImage,"Iodination",(1100,380), font, 1,(0,0,0),1,cv2.LINE_AA)
cv2.imshow('RefCard', ReferenceImage)
keypress=cv2.waitKey(0) & 0xFF
cv2.imwrite("ArchRefCard.jpg", ReferenceImage)
cv2.destroyAllWindows()
