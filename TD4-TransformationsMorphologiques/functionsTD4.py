# -*- coding: utf-8 -*-
"""
Fonctions pour script pour le TD4 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

def MyOpen(img, form=cv2.MORPH_RECT, kernel_size=(3,3)):
    """
    Fonction qui réalise l'ouverture

    Input :
        img -> Image de travail
        form -> Forme du noyau à utiliser
        kernel_size = -> Taille du noyau à utiliser
    
    Output :
        img_opened -> Image ouverte    
    """

    # Création du noyau
    kernel = cv2.getStructuringElement(form, kernel_size)

    img_eroded = cv2.erode(img, kernel) # Erosion de l'image
    img_opened = cv2.dilate(img_eroded, kernel) # Dilatation de l'image
    
    return img_opened

def MyClose(img, form=cv2.MORPH_RECT, kernel_size=(3,3)):
    """
    Fonction qui réalise la fermeture

    Input :
        img -> Image de travail
        form -> Forme du noyau à utiliser
        kernel_size = -> Taille du noyau à utiliser
    
    Output :
        img_closed -> Image fermée    
    """

    # Création du noyau
    kernel = cv2.getStructuringElement(form, kernel_size)

    img_dilated = cv2.dilate(img, kernel) # Dilatation de l'image
    img_closed = cv2.erode(img_dilated, kernel) # Erosion de l'image
    
    return img_closed
