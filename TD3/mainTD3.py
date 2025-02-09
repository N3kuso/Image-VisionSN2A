# -*- coding: utf-8 -*-
"""
Script pour le TD3 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as ski
import os

###########################################################
# Q2 :
###########################################################
file_ballon = "TD3/snake.png" # variable contenant le nom du fichier
img_ballon = cv2.imread(file_ballon) # Lecture du fichier avec opencv, on obtient une image BGR
img_ballon = cv2.cvtColor(img_ballon, cv2.COLOR_BGR2RGB) # Conversion de l'image en RGB

# Affichage
plt.figure(2)
plt.imshow(img_ballon)
plt.title("Ballon.jpg")
#plt.show()

###########################################################
# Q3 :
###########################################################
saltPepper_rate = 0.05 #float(input("Entrez le taux de bruit Salt & Pepper: "))
# Ajout d'un bruit Salt & Pepper
img_ballon_salt_pepper = ski.util.random_noise(img_ballon, mode='s&p', amount=saltPepper_rate )
img_ballon_salt_pepper = np.uint8(255 * img_ballon_salt_pepper) # Conversion vers des valeurs entre 0 et 255

# Affichage
plt.figure(3)
plt.imshow(img_ballon_salt_pepper)
plt.title("Ballon.jpg Salt & Pepper")
#plt.show()

###########################################################
# Q4 :
###########################################################
variance_gaussian = 0.01 #float(input("Entrez la variance du bruit Gaussien: "))
# Ajout d'un bruit Gaussien
img_ballon_gaussian = ski.util.random_noise(img_ballon, mode='gaussian', mean= 0.001, var=variance_gaussian) 
img_ballon_gaussian = np.uint8(255 * img_ballon_gaussian)# Conversion vers des valeurs entre 0 et 255

# Affichage
plt.figure(4)
plt.imshow(img_ballon_gaussian)
plt.title("Ballon.jpg Gaussien")
#plt.show()

###########################################################
# Q5 :
###########################################################
## Filtre passe-bas moyenneur ##
# Image Salt & Pepper
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10,15))
fig = plt.figure(5)
fig.suptitle("Filtre Moyenneur")
axes[0,0].imshow(img_ballon_salt_pepper)
axes[0,0].set_title("Ballon.jpg Salt & pepper")

# Image Gaussien
# plt.subplot(5, 2, 2)
axes[0,1].imshow(img_ballon_gaussian)
axes[0,1].set_title("Ballon.jpg gaussian")

kernel_size_range = [(3,3), (5,5), (7,7), (11,11)] # Plage de taille de kernel
for i,kernel_size in enumerate(kernel_size_range):
    print(f"Kernel_size : {kernel_size}")
    
    # Filtrage de l'image Salt & Pepper
    img_sp_filtered = cv2.blur(img_ballon_salt_pepper, kernel_size) # Filtrage de l'image
    # plt.subplot(5, 2, 3+i)
    axes[1+i,0].imshow(img_sp_filtered)
    axes[1+i,0].set_title(f"{kernel_size}")

    # Filtrage de l'image Gaussien
    img_gaussian_filtered = cv2.blur(img_ballon_gaussian, kernel_size) # Filtrage de l'image
    # plt.subplot(5, 2, 4+i)
    axes[1+i,1].imshow(img_gaussian_filtered)
    axes[1+i,1].set_title(f"{kernel_size}")

plt.tight_layout()
#plt.show()
 
## Filtre passe-bas gaussien ##
sigma = 15
# Image Salt & Pepper
fig, axes = plt.subplots(nrows=5, ncols=2, figsize=(10,15))
fig = plt.figure(6)
fig.suptitle("Filtre Gaussien")
axes[0,0].imshow(img_ballon_salt_pepper)
axes[0,0].set_title("Ballon.jpg Salt & pepper")

# Image Gaussien
# plt.subplot(5, 2, 2)
axes[0,1].imshow(img_ballon_gaussian)
axes[0,1].set_title("Ballon.jpg gaussian")

kernel_size_range = [(3,3), (5,5), (7,7), (11,11)] # Plage de taille de kernel
for i,kernel_size in enumerate(kernel_size_range):
    print(f"Kernel_size : {kernel_size}")
    
    # Filtrage de l'image Salt & Pepper
    img_sp_filtered = cv2.GaussianBlur(img_ballon_salt_pepper, kernel_size, sigma) # Filtrage de l'image
    # plt.subplot(5, 2, 3+i)
    axes[1+i,0].imshow(img_sp_filtered)
    axes[1+i,0].set_title(f"{kernel_size}")

    # Filtrage de l'image Gaussien
    img_gaussian_filtered = cv2.GaussianBlur(img_ballon_gaussian, kernel_size, sigma) # Filtrage de l'image
    # plt.subplot(5, 2, 4+i)
    axes[1+i,1].imshow(img_gaussian_filtered)
    axes[1+i,1].set_title(f"{kernel_size}")

plt.tight_layout()
#plt.show()

## Filtre passe-bas conique ##
# Image Salt & Pepper
fig, axes = plt.subplots(nrows=2, ncols=2)
fig = plt.figure(7)
fig.suptitle("Filtre Conique")
axes[0,0].imshow(img_ballon_salt_pepper)
axes[0,0].set_title("Ballon.jpg Salt & pepper")

# Image Gaussien
# plt.subplot(5, 2, 2)
axes[0,1].imshow(img_ballon_gaussian)
axes[0,1].set_title("Ballon.jpg gaussian")


# Création du masque du filtre
kernel_conique = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float32)

kernel_conique /= kernel_conique.sum()

print(kernel_conique)

# Filtrage de l'image Salt & Pepper
img_sp_filtered = cv2.filter2D(src=img_ballon_salt_pepper, ddepth=-1, kernel=kernel_conique) # Filtrage de l'image
# plt.subplot(5, 2, 3+i)
axes[1,0].imshow(img_sp_filtered)
axes[1,0].set_title(f"{kernel_size}")

# Filtrage de l'image Gaussien
img_gaussian_filtered = cv2.filter2D(src=img_ballon_gaussian, ddepth=-1, kernel=kernel_conique) # Filtrage de l'image
# plt.subplot(5, 2, 4+i)
axes[1,1].imshow(img_gaussian_filtered)
axes[1,1].set_title(f"{kernel_size}")

plt.tight_layout()
#plt.show()

###########################################################
# Q6 :
###########################################################
## Filtre Moyenneur ##
kernel_size_range = [(3,3), (5,5), (7,7), (11,11)] # Plage de taille de kernel

fig, axes = plt.subplots(nrows=4, ncols=3, figsize=(10,15))
fig = plt.figure(8)

for i,kernel_size in enumerate(kernel_size_range):
    # Création du masque
    kernel = np.ones(kernel_size)
    kernel /= kernel.sum()
    print(kernel)

    # Calcul de la fft2D
    y = np.fft.fft2(kernel, s=(64,64))

    # Affichage
    axes[i,0].imshow(kernel, cmap='jet')

    axes[i,1].imshow(np.fft.fftshift(np.abs(y)),cmap="jet")

    axes[i,2].imshow(np.fft.fftshift(np.angle(y)), cmap="jet")

plt.tight_layout()
#plt.show()

###########################################################
# Q8 :
###########################################################
## Filtre median ##
# Image Salt & Pepper
fig, axes = plt.subplots(nrows=4, ncols=2, figsize=(10,15))
fig = plt.figure(9)
fig.suptitle("Filtre median")
axes[0,0].imshow(img_ballon_salt_pepper)
axes[0,0].set_title("Snake.png Salt & pepper")

# Image Gaussien
# plt.subplot(5, 2, 2)
axes[0,1].imshow(img_ballon_gaussian)
axes[0,1].set_title("Snake.png gaussian")

kernel_size_range = [3,5,7] # Plage de taille de kernel
for i,kernel_size in enumerate(kernel_size_range):
    print(f"Kernel_size : {kernel_size}")
    
    # Filtrage de l'image Salt & Pepper
    img_sp_filtered = cv2.medianBlur(img_ballon_salt_pepper, kernel_size) # Filtrage de l'image
    # plt.subplot(5, 2, 3+i)
    axes[1+i,0].imshow(img_sp_filtered)
    axes[1+i,0].set_title(f"{kernel_size}")

    # Filtrage de l'image Gaussien
    img_gaussian_filtered = cv2.medianBlur(img_ballon_gaussian, kernel_size) # Filtrage de l'image
    # plt.subplot(5, 2, 4+i)
    axes[1+i,1].imshow(img_gaussian_filtered)
    axes[1+i,1].set_title(f"{kernel_size}")

plt.tight_layout()
#plt.show()

###########################################################
# Q9 :
###########################################################
## Filtre median successif ##
fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(10,15))
fig = plt.figure(10)
# Image Salt & Pepper
axes[0].imshow(img_ballon_salt_pepper)
axes[0].set_title("Snake.png Salt & pepper")


kernel_size = 7
# Filtrage de l'image Salt & Pepper
# 1 fois
img_sp_filtered = cv2.medianBlur(img_ballon_salt_pepper, kernel_size) # Filtrage de l'image
axes[1].imshow(img_sp_filtered)

# 2 fois
img_sp_filtered = cv2.medianBlur(img_sp_filtered, kernel_size) # Filtrage de l'image
axes[2].imshow(img_sp_filtered)

# 3 fois
img_sp_filtered = cv2.medianBlur(img_sp_filtered, kernel_size) # Filtrage de l'image
axes[3].imshow(img_sp_filtered)

plt.tight_layout()
#plt.show()

###########################################################
# Q10 :
###########################################################
filename = "TD3/colorful.jpg"
img = cv2.imread(filename, 0) # Lecture du fichier avec opencv, image en niveau de gris

# Filtre gradient
gx_mask = np.array([[0, -1], [0, 1]], dtype=float) # Masque 2x2 gradient direction x
gy_mask = np.array([[0, 0], [1, -1]], dtype=float) # Masque 2x2 gradient direction
gradient_x = cv2.filter2D(img, cv2.CV_64F, gx_mask) # Dérivée selon x
gradient_y = cv2.filter2D(img, cv2.CV_64F, gy_mask) # Dérivée selon y

# Filtre Roberts
roberts = ski.filters.roberts(img)

# Filtre Prewitt
prewitt = ski.filters.prewitt(img)

# Filtre Sobel
sobel_x = cv2.Sobel(img, cv2.CV_64F, dx=1, dy=0, ksize=3) # Dérivée selon X
sobel_y = cv2.Sobel(img, cv2.CV_64F, dx=0, dy=1, ksize=3) # Dérivée selon X

# Filtre Laplacien
laplacian = cv2.Laplacian(img, cv2.CV_64F)

### Affichage ###
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(10,15))
fig = plt.figure(11)
fig.suptitle("Filtrage 'Passe-haut'")
# Image de base
axes[0,0].imshow(img, cmap="gray")
axes[0,0].set_title("Image de base")

# Gradient x
axes[0,1].imshow(gradient_x, cmap="gray")
axes[0,1].set_title("Gradient x")

# Gradient y
axes[0,2].imshow(gradient_y, cmap="gray")
axes[0,2].set_title("Gradient y")

# Roberts
axes[0,3].imshow(roberts, cmap="gray")
axes[0,3].set_title("Roberts")

# Prewitt
axes[1,0].imshow(prewitt, cmap="gray")
axes[1,0].set_title("Prewitt")

# Sobel x
axes[1,1].imshow(sobel_x, cmap="gray")
axes[1,1].set_title("Sobel x")

# Sobel y
axes[1,2].imshow(sobel_y, cmap="gray")
axes[1,2].set_title("Sobel y")

# Laplacien
axes[1,3].imshow(laplacian, cmap="gray")
axes[1,3].set_title("Laplacien")

plt.tight_layout()
#plt.show()

###########################################################
# Q11 :
###########################################################
# Définition des noyaux de chaque filtre, pris dans le cours donc peut-être différent de ceux utilisés dans la question précédente

# Initialisation d'une liste de noyau
kernel_list = []
# Initialisation d'une liste de titre, associé à chaque noyau
title_list = []

## Noyau gradient 2x2 ##
# Noyau gradient x
kernel_gradient_x = np.array([[0, -1], [0, 1]], dtype=float)

kernel_list.append(kernel_gradient_x)
title_list.append("Noyau gradient x")

## Noyau Roberts 2x2 ##
# Noyau Roberts x
kernel_roberts_x = np.array([[1, 0], [0, -1]], dtype=float)

kernel_list.append(kernel_roberts_x)
title_list.append("Noyau Roberts x")

## Noyau Prewitt 3x3 ##
# Noyau Prewitt x
kernel_prewitt_x = 1/3 * np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=float)

kernel_list.append(kernel_prewitt_x)
title_list.append("Noyau Prewitt x")

## Noyau Sobel ##
# Noyau Sobel x
kernel_sobel_x = 1/4 * np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=float)

kernel_list.append(kernel_sobel_x)
title_list.append("Noyau Sobel x")

## Noyau Laplacien ##
kernel_laplacian = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=float)

kernel_list.append(kernel_laplacian)
title_list.append("Noyau Laplacien")

print(kernel_list)
print(title_list)

### Affichage ###
fig, axes = plt.subplots(nrows=5, ncols=3, figsize=(10,15))
fig = plt.figure(12)
fig.suptitle("Fonction de transfert")

for i, kernel in enumerate(kernel_list):
    # Calcul de la fonction de transfert
    ft = np.fft.fft2(kernel, s=(64,64))

    # Affichage du noyau
    axes[i,0].imshow(kernel, cmap='jet')

    # Affichage du module
    axes[i,1].imshow(np.fft.fftshift(np.abs(ft)),cmap="jet")
    axes[i,1].set_title(title_list[i])

    # Affichage de l'angle
    axes[i,2].imshow(np.fft.fftshift(np.angle(ft)), cmap="jet")

plt.tight_layout()
#plt.show()

###########################################################
# Q12 :
###########################################################
from functionsTD3 import EdgeDetection

### Affichage ###
fig, axes = plt.subplots(nrows=2, ncols=3)
fig = plt.figure(13)
fig.suptitle("Détection de contour")

# Image de base
axes[0,0].imshow(img, cmap="gray")
axes[0,0].set_title("Image de base")

# Detection de contour gradient simple
axes[0,1].imshow(EdgeDetection(gradient_x, gradient_y), cmap="gray")
axes[0,1].set_title("Gradient")

# Detection de contour Roberts 
axes[0,2].imshow(roberts, cmap="gray")
axes[0,2].set_title("Roberts")

# Detection de contour Prewitt
axes[1,0].imshow(prewitt, cmap="gray")
axes[1,0].set_title("Prewitt")

# Detection de contour sobel
axes[1,1].imshow(EdgeDetection(sobel_x,sobel_y), cmap="gray")
axes[1,1].set_title("Sobel x")

# Laplacien
axes[1,2].imshow(EdgeDetection(laplacian), cmap="gray")
axes[1,2].set_title("Laplacien")

plt.tight_layout()
plt.show()