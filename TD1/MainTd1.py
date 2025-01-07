"""
Script pour le TD1 Image&Vision

@author : kooky
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

###########################################################
# Q2 :
###########################################################

file_png='flowers.png'
image_png=cv2.imread(file_png)[:,:,::-1]


file_jpg='lena.jpg'
image_jpg=cv2.imread(file_jpg)[:,:,::-1]


file_bmp='hibiscus.bmp'
image_bmp=cv2.imread(file_bmp)[:,:,::-1]


# plot des images
plt.subplot(1,3,1)
plt.imshow(image_png)
plt.title(file_png)
plt.subplot(1,3,2)
plt.imshow(image_jpg)
plt.title(file_jpg)
plt.subplot(1,3,3)
plt.imshow(image_bmp)
plt.title(file_bmp)
plt.show()

###########################################################
# Q3 :
###########################################################
print(f"Image {file_png} -> Lignes = {len(image_png[:,0,0])} | Colonnes = {len(image_png[:,1,0])}")
print(f"Image {file_jpg} -> Lignes = {len(image_jpg[:,0,0])} | Colonnes = {len(image_jpg[:,1,0])}")
print(f"Image {file_bmp} -> Lignes = {len(image_bmp[:,0,0])} | Colonnes = {len(image_bmp[:,1,0])}")
# Pixels codés sur 8bits (uint8)


###########################################################
# Q4 :
###########################################################
list_cmap = ['viridis', 'plasma', 'inferno', 'magma', 'cividis']

for color in list_cmap:
    # affichage image_png
    plt.subplot(1,3,1)
    plt.imshow(image_png[:,:,0], cmap=color)
    plt.title(file_png)
    plt.subplot(1,3,2)
    plt.imshow(image_png[:,:,1], cmap=color)
    plt.title(file_png)
    plt.subplot(1,3,3)
    plt.imshow(image_png[:,:,2], cmap=color)
    plt.title(file_png)
    plt.show()
    
    # Affichage image_jpg
    plt.subplot(1,3,1)
    plt.imshow(image_jpg[:,:,0], cmap=color)
    plt.title(file_jpg)
    plt.subplot(1,3,2)
    plt.imshow(image_jpg[:,:,1], cmap=color)
    plt.title(file_jpg)
    plt.subplot(1,3,3)
    plt.imshow(image_jpg[:,:,2], cmap=color)
    plt.title(file_jpg)
    plt.show()
    
    # Affichage image_bmp
    plt.subplot(1,3,1)
    plt.imshow(image_bmp[:,:,0], cmap=color)
    plt.title(file_bmp)
    plt.subplot(1,3,2)
    plt.imshow(image_bmp[:,:,1], cmap=color)
    plt.title(file_bmp)
    plt.subplot(1,3,3)
    plt.imshow(image_bmp[:,:,2], cmap=color)
    plt.title(file_bmp)
    plt.show()
    

