# -*- coding: utf-8 -*-
"""
Fonctions pour le script MainTd1.py

@author: kooky
"""
import numpy as np
import cv2


def NivGris(img):
    img = img.astype(float)
    
    n = len(img[:,0,0])
    p = len(img[0,:,0])
    result = np.empty((n,p))
    print(f"Lignes : {n} | Colonnes : {p}")
    
    for i in range(n):
        for j in range(p):
            temp=sum(img[i,j,:])/3
            result[i][j]=temp
    result=result.astype(np.uint8)
    
    return result

def NivGrisM(img):
    n = len(img[:,0,0])
    p = len(img[0,:,0])
    result = np.empty((n,p))
    
    for i in range(n):
        for j in range(p):
            result[i][j]=MoyPondImg(img[i,j,0], img[i,j,1], img[i,j,2])
    result=result.astype(np.uint8)
    
    return result
    
def MoyPondImg(R,G,B):
    return 0.2989*R + 0.5870*G + 0.1140*B

def Morphing(I1, I2, alpha):
    I1 = I1.astype(float)
    I2 = I2.astype(float)
    return (alpha*I1 + (1-alpha) * I2).astype(np.uint8)

def Quantize(img, n):
    img_reshape = img.reshape(-1,3)
    img_reshape = img_reshape.astype(np.float32)
    
    # Define criteria = ( type, max_iter = 100 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1.0)
    
    ret,label,center = cv2.kmeans(img_reshape,n,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
    
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    
    return res2
    