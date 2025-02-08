# -*- coding: utf-8 -*-
"""
Fonctions pour script pour le TD3 Image&Vision

@author: kooky
"""
import numpy as np
import cv2

def EdgeDetection(gradient_x, gradient_y):
    """
    Ma Fonction qui permet de détecter les contours d'une image
    
    Input :
        gradient_x -> Image filtrée en x
        gradient_y -> Image filtrée en y 

    Output :
        binary_contours -> Image binaire avec contours detectés
    """

    # Calcul du maximum entre les deux gradients
    gradient_combined = np.maximum(np.abs(gradient_x), np.abs(gradient_y))

    # Calcul du négatif de l'image
    gradient_negative = cv2.bitwise_not(gradient_combined.astype(np.uint8))

    # Binarisation de l'image avec méthode d'Otsu
    _, binary_contours = cv2.threshold(gradient_negative, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Calcul du négatif de l'image (pour avoir le fond noir et les contours blancs)
    binary_contours = cv2.bitwise_not(binary_contours)

    return binary_contours