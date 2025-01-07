# -*- coding: utf-8 -*-
"""
Fonctions pour le script MainTd1.py

@author: kooky
"""
import numpy as np


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