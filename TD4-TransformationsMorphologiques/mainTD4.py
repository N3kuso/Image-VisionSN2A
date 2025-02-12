# -*- coding: utf-8 -*-
"""
Script pour le TD4 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import functionsTD4

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
# Q3 :
###########################################################
kernel_size = (3,3) # Taille du noyau 
kernel =  cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) # Création d'un noyau carré avec OpenCv

img_dilated = cv2.dilate(img_binarized, kernel) # Dilatation de l'image
img_eroded = cv2.erode(img_binarized, kernel) # Erosion de l'image

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,15))
fig = plt.figure(2)
fig.suptitle(f"Dilatation/Erosion avec noyau carré de taille {kernel_size}")

# Image Binaire
axes[0].imshow(img_binarized, cmap="gray")
axes[0].set_title("Image binaire")

# Image dilatée
axes[1].imshow(img_dilated, cmap='gray')
axes[1].set_title("Image dilatée")

# Image erodée
axes[2].imshow(img_eroded, cmap='gray')
axes[2].set_title("Image erodée")

###########################################################
# Q4 :
###########################################################
# Noyau circulaire de rayon 5
kernel_size = (5,5) # Taille du noyau 
kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size) # Création d'un noyau circulaire avec OpenCv

img_dilated = cv2.dilate(img_binarized, kernel) # Dilatation de l'image
img_eroded = cv2.erode(img_binarized, kernel) # Erosion de l'image

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,15))
fig = plt.figure(3)
fig.suptitle(f"Dilatation/Erosion avec noyau disque de rayon {kernel_size}")

# Image Binaire
axes[0].imshow(img_binarized, cmap="gray")
axes[0].set_title("Image binaire")

# Image dilatée
axes[1].imshow(img_dilated, cmap='gray')
axes[1].set_title("Image dilatée")

# Image erodée
axes[2].imshow(img_eroded, cmap='gray')
axes[2].set_title("Image erodée")

###########################################################
# Q5 :
###########################################################
# Ouverture de l'image
img_opened = functionsTD4.MyOpen(img_binarized)

# Fermeture de l'image
img_closed = functionsTD4.MyClose(img_binarized)

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,15))
fig = plt.figure(4)
fig.suptitle(f"Ouverture/Fermeture")

# Image Binaire
axes[0].imshow(img_binarized, cmap="gray")
axes[0].set_title("Image binaire")

# Image Ouverte
axes[1].imshow(img_opened, cmap='gray')
axes[1].set_title("Image ouverte")

# Image Fermée
axes[2].imshow(img_closed, cmap='gray')
axes[2].set_title("Image fermée")

###########################################################
# Q6 :
###########################################################
# Calcul du gradient morphologique sur img niv de gris
img_morph = functionsTD4.MyGradMorph(img_gray)

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(10,15))
fig = plt.figure(5)
fig.suptitle(f"Gradient morphologique")

# Image de base
axes[0].imshow(img)
axes[0].set_title("Image de base")

# Image Binaire
axes[1].imshow(img_gray, cmap="gray")
axes[1].set_title("Image Niv. de gris")

# Gradient morphologique
axes[2].imshow(img_morph, cmap='gray')
axes[2].set_title("Gradient morphologique")

###########################################################
# Q7 :
###########################################################
# Paramètres pour la reconstruction morphologique
form = cv2.MORPH_RECT
kernel_size = (3,3)
method="dilation"

# Reconstruction morphologique
img_reconstruct = functionsTD4.MyReconstruct(img_binarized, form=form, kernel_size=kernel_size, method=method)

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,15))
fig = plt.figure(6)
fig.suptitle(f"Reconstruction morphologique")

# Image de base
axes[0].imshow(img_binarized, cmap="gray")
axes[0].set_title("Image binaire")

# Reconstruction Morphologique
axes[1].imshow(img_reconstruct, cmap="gray")
axes[1].set_title("Reconstruction Morphologique")

###########################################################
# Q8 :
###########################################################
# Extraction des contours
img_contour = functionsTD4.MyContour(img_binarized)

### Affichage ###
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10,15))
fig = plt.figure(7)
fig.suptitle(f"Extraction des contours")

# Gradient Morphologique
axes[0].imshow(img_morph, cmap="gray")
axes[0].set_title("Gradient Morphologique")

# Extraction des contours
axes[1].imshow(img_contour, cmap="gray")
axes[1].set_title("Contours Extraits")

###########################################################
# Q9 :
###########################################################
# Réalisation des questions précédentes avec l'image en niveau de gris

# Paramètres 
form = cv2.MORPH_RECT
kernel_size = (3,3)
method="dilation"
kernel = cv2.getStructuringElement(form, kernel_size)

# Dilatation
img_dilated = cv2.dilate(img_gray, kernel)

# Erosion
img_eroded = cv2.erode(img_gray, kernel)

# Ouverture
img_opened = functionsTD4.MyOpen(img_gray, form=form, kernel_size=kernel_size)

# Fermeture
img_closed = functionsTD4.MyClose(img_gray, form=form, kernel_size=kernel_size)

# Gradient Morphologique
img_morph = functionsTD4.MyGradMorph(img_gray, form=form, kernel_size=kernel_size)

# Reconstruction Morphologique
img_reconstruct = functionsTD4.MyReconstruct(img_gray, form=form, kernel_size=kernel_size, method=method)

# Extraction des contours
img_contour = functionsTD4.MyContour(img_gray, form=form, kernel_size=kernel_size)

### Affichage ###
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,15))
fig = plt.figure(8)
fig.suptitle(f"Question 9")

# Image Niv. de gris
axes[0, 0].imshow(img_gray, cmap="gray")
axes[0, 0].set_title("Image Niv. de gris")
 
# Dilatation
axes[0, 1].imshow(img_dilated, cmap="gray")
axes[0, 1].set_title("Dilatation")

# Erosion
axes[0, 2].imshow(img_eroded, cmap="gray")
axes[0, 2].set_title("Erosion")

# Ouverture
axes[0, 3].imshow(img_opened, cmap="gray")
axes[0, 3].set_title("Ouverture")

# Fermeture
axes[1, 0].imshow(img_closed, cmap="gray")
axes[1, 0].set_title("Fermeture")

# Gradient Morphologique
axes[1, 1].imshow(img_morph, cmap="gray")
axes[1, 1].set_title("Gradient Morphologique")

# Recontruction Morphologique
axes[1, 2].imshow(img_reconstruct, cmap="gray")
axes[1, 2].set_title("Reconstruction Morphologique")

# Extraction de contours
axes[1, 3].imshow(img_contour, cmap="gray")
axes[1, 3].set_title("Extraction de contours")

plt.tight_layout()
plt.show()