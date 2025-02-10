# -*- coding: utf-8 -*-
"""
Script pour le TD4 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2


###########################################################
# Q2 :
###########################################################

filename_img = "TD4-TransformationsMorphologiques/coins.jpg"
img = cv2.imread(filename_img) # Lecture image
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Conversion de l'image en RGB
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) # Conversion de l'image en niveau de gris
ret2, img_binarized=cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU) # Binarisation de l'image avec la méthode d'Otsu

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,15))
fig = plt.figure(1)
fig.suptitle("Affichage de l'image")

# Image de base
axes[0].imshow(img)
axes[0].set_title("Image de base")

# Image en niveau de gris
axes[1].imshow(img_gray, cmap='gray')
axes[1].set_title("Image Niv. de Gris")

# Image binaire
axes[2].imshow(img_binarized, cmap='gray')
axes[2].set_title("Image binarisé")

###########################################################
# Q2 :
###########################################################
kernel_size = (3,3) # Taille du noyau 
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # Création d'un noyau carré avec OpenCv

img_dilated = cv2.dilate(img_binarized, kernel) # Dilatation de l'image
img_eroded = cv2.erode(img_binarized, kernel) # Erosion de l'image

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,15))
fig = plt.figure(2)
fig.suptitle("Dilatation/Erosion")

# Image Binaire
axes[0].imshow(img_binarized, cmap="gray")
axes[0].set_title("Image binaire")

# Image dilatée
axes[1].imshow(img_dilated, cmap='gray')
axes[1].set_title("Image dilatée")

# Image erodée
axes[2].imshow(img_eroded, cmap='gray')
axes[2].set_title("Image erodée")

plt.tight_layout()
plt.show()
