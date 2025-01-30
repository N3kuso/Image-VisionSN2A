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
file_ballon = "TD3/ballon.jpg" # variable contenant le nom du fichier
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
saltPepper_rate = 0.15 #float(input("Entrez le taux de bruit Salt & Pepper: "))
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
variance_gaussian = 0.05 #float(input("Entrez la variance du bruit Gaussien: "))
# Ajout d'un bruit Gaussien
img_ballon_gaussian = ski.util.random_noise(img_ballon, mode='gaussian', var=variance_gaussian) 
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
plt.figure(5)
plt.subplot(3, 3, 1)
plt.imshow(img_ballon_salt_pepper)
plt.title("Ballon.jpg Salt & pepper")

kernel_size_range = [(3,3), (5,5), (7,7), (11,11)] # Plage de taille de kernel
for i,kernel_size in enumerate(kernel_size_range):
    print(f"Kernel_size : {kernel_size}")
    img_filtered = cv2.blur(img_ballon_salt_pepper, kernel_size) # Filtrage de l'image
    plt.subplot(3, 3, 2+i)
    plt.imshow(img_filtered)
    plt.title(f"{kernel_size}")

plt.tight_layout()
plt.show()