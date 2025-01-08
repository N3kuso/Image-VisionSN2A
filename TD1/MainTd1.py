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
flag = 0
if flag ==1:
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

###########################################################
# Q5 :
###########################################################
image_jpg_resize=image_jpg[100:200,50:200]
plt.imshow(image_jpg_resize)
plt.title("Image redimensionnée")
plt.show()

###########################################################
# Q6 :
###########################################################
resolution_range=["256x256","128x128","64x64","32x32"]
range_n=[1,2,4,8]

for i in range(len(range_n)):
    plt.imshow(image_jpg[::range_n[i],::range_n[i]])
    plt.title(resolution_range[i])
    plt.show()
    
###########################################################
# Q7 :
###########################################################
img_gray=cv2.imread(file_jpg,0)
plt.imshow(img_gray, cmap="grey")
plt.title("Image en niveau de gris")
plt.show()

###########################################################
# Q8 :
###########################################################
from Functions_MainTd1 import NivGris

my_img_gray = NivGris(image_jpg)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Image en niveau de gris (Ma fonction Nivgris)")
plt.show()

###########################################################
# Q9 :
###########################################################
from Functions_MainTd1 import NivGrisM

my_img_gray2 = NivGrisM(image_jpg)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Image en niveau de gris (Ma fonction NivgrisM)")
plt.show()

###########################################################
# Q10 :
###########################################################
plt.subplot(1,3,1)
plt.imshow(img_gray, cmap="grey")
plt.title("niveau de gris")
plt.subplot(1,3,2)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Ma fonction Nivgris")
plt.subplot(1,3,3)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Ma fonction NivgrisM")
plt.show()