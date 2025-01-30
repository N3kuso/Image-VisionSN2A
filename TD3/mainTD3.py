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
file_ballon = "ballon.jpg" # variable contenant le nom du fichier
img_ballon = cv2.imread(file_ballon) # Lecture du fichier avec opencv, on obtient une image BGR
img_ballon = cv2.cvtColor(img_ballon, cv2.COLOR_BGR2RGB) # Conversion de l'image en RGB

# Affichage
plt.figure(2)
plt.imshow(img_ballon)
plt.title("Ballon.jpg")
plt.show()

###########################################################
# Q3 :
###########################################################
saltPepper_rate = float(input("Entrez le taux de bruit Salt & Pepper: "))
img_ballon_salt_pepper = ski.util.random_noise(img_ballon, mode='s&p', amount=saltPepper_rate )

# Affichage
plt.figure(3)
plt.imshow(img_ballon_salt_pepper)
plt.title("Ballon.jpg Salt & Pepper")
plt.show()

###########################################################
# Q4 :
###########################################################
variance_gaussian = float(input("Entrez la variance du bruit Gaussien: "))
img_ballon_gaussian = ski.util.random_noise(img_ballon, mode='gaussian', var=variance_gaussian)

# Affichage
plt.figure(4)
plt.imshow(img_ballon_gaussian)
plt.title("Ballon.jpg Gaussien")
plt.show()