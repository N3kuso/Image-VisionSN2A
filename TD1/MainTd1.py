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
plt.subplot(3,1,1)
plt.imshow(img_gray, cmap="grey")
plt.title("niveau de gris")
plt.subplot(3,1,2)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Ma fonction Nivgris")
plt.subplot(3,1,3)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Ma fonction NivgrisM")
plt.show()

###########################################################
# Q11 :
###########################################################
plt.subplot(3,2,1)
plt.imshow(img_gray, cmap="grey")
plt.title("niveau de gris")
plt.subplot(3,2,3)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Ma fonction Nivgris")
plt.subplot(3,2,5)
plt.imshow(my_img_gray, cmap="grey")
plt.title("Ma fonction NivgrisM")

plt.subplot(3,2,2)
plt.imshow(img_gray[:8,:8], cmap="grey")
plt.title("niveau de gris")
plt.subplot(3,2,4)
plt.imshow(my_img_gray[:8,:8],cmap="grey")
plt.title("Ma fonction Nivgris")
plt.subplot(3,2,6)
plt.imshow(my_img_gray[:8,:8],cmap="grey")
plt.title("Ma fonction NivgrisM")
plt.show()

# Observation : de légères différences

###########################################################
# Q12 :
###########################################################
from Functions_MainTd1 import Morphing
image_hibiscus = image_bmp[:502,:502]
image_flowers = image_png

alpha_range = np.linspace(0,1,12)

for i in range (len(alpha_range)):
    img_temp= Morphing(image_hibiscus, image_flowers, alpha_range[i])
    plt.subplot(3,4,i+1)
    plt.imshow(img_temp)
    plt.axis("off")
    plt.title(f"Image {i+1}")

plt.show()

###########################################################
# Q13 :
###########################################################
# Lecture des images avec opencv dans le système BGR (par défaut)
img_flowers = cv2.imread(file_png)
img_lena = cv2.imread(file_jpg)
img_hibiscus = cv2.imread(file_bmp)

# Conversion des images dans le système HSV
img_flowers_hsv = cv2.cvtColor(img_flowers, cv2.COLOR_BGR2HSV)
img_lena_hsv = cv2.cvtColor(img_lena, cv2.COLOR_BGR2HSV)
img_hibiscus_hsv = cv2.cvtColor(img_hibiscus, cv2.COLOR_BGR2HSV)

# Plotting
plt.subplot(1,3,1)
plt.imshow(img_flowers_hsv)
plt.title(file_png)
plt.subplot(1,3,2)
plt.imshow(img_lena_hsv)
plt.title(file_jpg)
plt.subplot(1,3,3)
plt.imshow(img_hibiscus_hsv)
plt.title(file_bmp)
plt.show()


# Conversion des images dans le système XYZ
img_flowers_XYZ = cv2.cvtColor(img_flowers, cv2.COLOR_BGR2XYZ)
img_lena_XYZ = cv2.cvtColor(img_lena, cv2.COLOR_BGR2XYZ)
img_hibiscus_XYZ = cv2.cvtColor(img_hibiscus, cv2.COLOR_BGR2XYZ)

# Plotting
plt.subplot(1,3,1)
plt.imshow(img_flowers_XYZ)
plt.title(file_png)
plt.subplot(1,3,2)
plt.imshow(img_lena_XYZ)
plt.title(file_jpg)
plt.subplot(1,3,3)
plt.imshow(img_hibiscus_XYZ)
plt.title(file_bmp)
plt.show()


# Conversion des images dans le système YCrCb
img_flowers_YCrCb = cv2.cvtColor(img_flowers, cv2.COLOR_BGR2YCrCb)
img_lena_YCrCb = cv2.cvtColor(img_lena, cv2.COLOR_BGR2YCrCb)
img_hibiscus_YCrCb = cv2.cvtColor(img_hibiscus, cv2.COLOR_BGR2YCrCb)

# Plotting
plt.subplot(1,3,1)
plt.imshow(img_flowers_YCrCb)
plt.title(file_png)
plt.subplot(1,3,2)
plt.imshow(img_lena_YCrCb)
plt.title(file_jpg)
plt.subplot(1,3,3)
plt.imshow(img_hibiscus_YCrCb)
plt.title(file_bmp)
plt.show()

###########################################################
# Q14 :
###########################################################
from Functions_MainTd1 import Quantize
quantize_range = [2,4,8,16]

k=0
for n in quantize_range:
    img_quantized = Quantize(image_jpg,n)
    plt.subplot(221+k)
    plt.title(f"{n} colors")
    plt.imshow(img_quantized)
    k+=1

plt.show()
###########################################################
# Q15 :
###########################################################
img_lena_gray = cv2.cvtColor(img_lena, cv2.COLOR_BGR2GRAY)

plt.imshow(img_lena_gray, cmap="grey")
plt.title("Original en Gris")
plt.show()

plt.imshow(img_lena_gray.T, cmap="grey")
plt.title("Transposé")
plt.show()