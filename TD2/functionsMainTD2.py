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
    return np.power(img/255, p)