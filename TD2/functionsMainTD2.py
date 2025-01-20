# -*- coding: utf-8 -*-
"""
Fonctions pour le script MainTD2.py

@author: kooky
"""

import numpy as np

def ImgNegative(img):
    # On inverse l'image -> Valeur max - Intensité lumineuse du pixel
    return np.max(img) - img

def ImgLogarithme(img, c):
    img.astype(np.double)
    return (c * np.log10(img)).astype(np.uint8)

def ImgPower(img, p):
    img.astype(np.double)
    return (255 * np.power(img/255, p)).astype(np.uint8)

def ImgCut(img, s1, s2):
    tmp = img.copy()
    tmp[np.where(tmp < s1)] = 0 # valeur inférieure à s1 -> 0
    tmp[np.where(tmp > s2)] = 255 # valeur supérieure à s2 -> 255
    return tmp

def ImgSeuil(img, s):
    tmp = img.copy()
    tmp[np.where(tmp < s)] = 0 # valeur inférieure à s -> 0
    tmp[np.where(tmp > s)] = 255 # valeur supérieure à s -> 255
    return tmp

def MyHistColor(img):
    H = np.zeros((256,3)) # Création d'une matrice 255 * 3 rempli de 0
    tmp = img.reshape(-1,3) # "Vectorisation de l'image" avec les colonnes correspondant aux couleurs RGB
    
    for i in range(tmp.shape[0]):
        H[tmp[i,0], 0] = H[tmp[i,0], 0]+1 # Colonne histogramme rouge
        #print(f"tmp[i,0] : {tmp[i,0]} | i : {i}")
        H[tmp[i,1], 1] = H[tmp[i,1], 1]+1 # Colonne histogramme vert
        #print(f"tmp[i,1] : {tmp[i,1]} | i : {i}")
        H[tmp[i,2], 2] = H[tmp[i,2], 2]+1 # Colonne histogramme blue
        #print(f"tmp[i,2] : {tmp[i,2]} | i : {i}")
    return H

def MyHistGrey(img):
    H = np.zeros(256)
    tmp = img.reshape(-1,1)
    
    for i in range(tmp.shape[0]):
        H[tmp[i]] = H[tmp[i]] + 1    
    return H

def HistCumul(img):
    hist = MyHistGrey(img) # Usage de fonction pour calculer l'histogramme de mon image
    
    hist_cumul = np.zeros(256)
    hist_cumul[0] = hist[0]
    
    for i in range(1, 256):
        hist_cumul[i] = hist_cumul[i-1] + hist[i]
        
    return hist_cumul
        