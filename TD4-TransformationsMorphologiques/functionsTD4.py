# -*- coding: utf-8 -*-
"""
Fonctions pour script pour le TD4 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.morphology import reconstruction

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

def MyGradMorph(img, form=cv2.MORPH_RECT, kernel_size=(3,3)):
    """
    Fonction qui réalise le gradient morphologique

    Input :
        img -> Image de travail
        form -> Forme du noyau à utiliser
        kernel_size = -> Taille du noyau à utiliser
    
    Output :
        img_morph -> Image calculée   
    """
    # Création du noyau
    kernel = cv2.getStructuringElement(form, kernel_size)

    img_dilated = cv2.dilate(img, kernel) # Dilatation de l'image
    img_eroded = cv2.erode(img, kernel) # Erosion de l'image

    img_morph = img_dilated - img_eroded

    return img_morph

def MyReconstruct(marker, form=cv2.MORPH_RECT, kernel_size=(3,3), method="dilation"):
    """
    Fonction qui réalise la reconstruction morphologique

    Input :
        marker -> Image marqueur
        form -> Forme du noyau à utiliser
        kernel_size = -> Taille du noyau à utiliser
        methode -> Méthode à utiliser
    
    Output :
        img_reconstruct -> Image calculée  
    """

    # Création du masque
    kernel = cv2.getStructuringElement(form, kernel_size)

    # Dilatation de l'image marker
    mask = cv2.dilate(marker, kernel)

    # Reconstruction morphologique de l'image
    img_reconstruct = reconstruction(marker, mask, method=method)

    return img_reconstruct

def MyContour(img, form=cv2.MORPH_RECT, kernel_size=(3,3)):
    """
    Fonction qui réalise l'extraction de contours

    Input :
        img -> Image de travail
        form -> Forme du noyau à utiliser
        kernel_size = -> Taille du noyau à utiliser

    Output :
        img_contour -> Image avec contours extraits  
    """

    # Création du noyau
    kernel = cv2.getStructuringElement(form, kernel_size)

    # Dilatation de l'image marker
    img_eroded = cv2.erode(img, kernel)

    img_contour = img - img_eroded

    return img_contour