# -*- coding: utf-8 -*-
"""
Script pour le TD3 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2
import skimage as ski

###########################################################
# Q2 :
###########################################################
file_ballon = "TD3/colorful.jpg" # variable contenant le nom du fichier
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
plt.show()