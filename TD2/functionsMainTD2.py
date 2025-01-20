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