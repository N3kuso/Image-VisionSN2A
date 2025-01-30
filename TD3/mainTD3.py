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
saltPepper_rate = 0.15
img_ballon_noised = ski.util.random_noise(img_ballon, mode='s&p', amount=saltPepper_rate )

# Affichage
plt.figure(3)
plt.imshow(img_ballon_noised)
plt.title("Ballon.jpg Salt & Pepper")
plt.show()