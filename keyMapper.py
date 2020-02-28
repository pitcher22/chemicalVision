#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 18:11:59 2020

@author: sensor
"""
import cv2
import numpy as np

runFlag=True
while runFlag:
    displayFrame = np.zeros((200, 200, 3), np.uint8)
    keypress=cv2.waitKeyEx(20000) 
    print(keypress)
    cv2.putText(displayFrame,str(keypress),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 0.4,(255,255,255),1,cv2.LINE_AA)
    cv2.imshow('Display', displayFrame)
    if keypress == ord('q'):
        runFlag=False
        break