# -*- coding: utf-8 -*-
"""
Script pour le TD2 Image&Vision

@author: kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

###########################################################
# Q2 :
###########################################################
file_ballon = "ballon.jpg" # variable contenant le nom du fichier
img_ballon = cv2.imread(file_ballon) # Lecture du fichier avec opencv, on obtient une image BGR
img_ballon = cv2.cvtColor(img_ballon, cv2.COLOR_BGR2RGB) # Conversion de l'image en RGB

# Affichage
plt.imshow(img_ballon)
plt.title("Ballon.jpg")
plt.show()

###########################################################
# Q3 :
###########################################################
img_ballon_gray = cv2.cvtColor(img_ballon, cv2.COLOR_RGB2GRAY) # Conversion de l'image RGB en niveau de gris

# Affichage avec la palette de couleur grise
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Ballon.jpg en niveau de gris")
plt.show()

###########################################################
# Q4 :
###########################################################
from functionsMainTD2 import ImgNegative

img_ballon_negative = ImgNegative(img_ballon_gray)

#plt.hist(img_ballon_negative.reshape(-1,1), 255)

# # Affichage
plt.subplot(221)
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Ballon.jpg")

plt.subplot(222)
plt.hist(img_ballon_gray.reshape(-1,1), bins=255) # On vectorise la matrice sinon plt execute un histogramme de chaque ligne de notre matrice
plt.title("Histogramme")

plt.subplot(223)
plt.imshow(img_ballon_negative, cmap="grey")
plt.title("Ballon.jpg negatif")

plt.subplot(224)
plt.hist(img_ballon_negative.reshape(-1,1), bins=255)
plt.title("Histogramme Negatif")
plt.show()

###########################################################
# Q5 :
###########################################################
from functionsMainTD2 import ImgLogarithme

# Affichage
plt.subplot(4,2,1)
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Original")

plt.subplot(4,2,2)
plt.hist(img_ballon_gray.reshape(-1,1), bins=255)
plt.title("Histogramme")

c_range = [10,20,100]
k=3
for c in c_range:
    img_log = ImgLogarithme(img_ballon_gray, c)
    plt.subplot(4,2,k)
    plt.imshow(img_log, cmap="grey")
    plt.title(f"Log facteur c :{c}")
    
    plt.subplot(4,2, (k+1))
    plt.hist(img_log.reshape(-1,1), bins=255)
    #plt.title(f"Histogramme :{c}")
    
    k+=2
plt.show()

###########################################################
# Q6 :
###########################################################
from functionsMainTD2 import ImgPower

#Affichage
plt.subplot(4,2,1)
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Original")

plt.subplot(4,2,2)
plt.hist(img_ballon_gray.reshape(-1,1), bins=255)
plt.title("Histogramme")

p_range = [0.5,10,2]
k=3
for p in p_range:
    img_pow = ImgPower(img_ballon_gray, p)
    plt.subplot(4,2,k)
    plt.imshow(img_pow, cmap="grey")
    plt.title(f"Pow facteur p :{p}")
    
    plt.subplot(4,2, (k+1))
    plt.hist(img_pow.reshape(-1,1), bins=255)
    #plt.title(f"Histogramme :{c}")
    
    k+=2
plt.show()