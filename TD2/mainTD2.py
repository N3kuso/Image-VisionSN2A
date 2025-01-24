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

p_range = [0.01,5,0.5]
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

###########################################################
# Q7 :
###########################################################
from functionsMainTD2 import ImgCut

s1 = 192
s2 = 220
img_cut = ImgCut(img_ballon_gray, s1, s2)

# Affichage
plt.subplot(221)
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Ballon.jpg")

plt.subplot(222)
plt.hist(img_ballon_gray.reshape(-1,1), bins=255) # On vectorise la matrice sinon plt execute un histogramme de chaque ligne de notre matrice
plt.title("Histogramme")

plt.subplot(223)
plt.imshow(img_cut, cmap="grey")
plt.title(f"s1 : {s1} | s2 : {s2}")

plt.subplot(224)
plt.hist(img_cut.reshape(-1,1), bins=255)
plt.show()

###########################################################
# Q8 + Q9 :
###########################################################
from functionsMainTD2 import ImgSeuil

s = int(input("Inserez une valeur de seuil : "))
img_seuil = ImgSeuil(img_ballon_gray, s)

# Affichage
plt.subplot(221)
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Ballon.jpg")

plt.subplot(222)
plt.hist(img_ballon_gray.reshape(-1,1), bins=255) # On vectorise la matrice sinon plt execute un histogramme de chaque ligne de notre matrice
plt.title("Histogramme")

plt.subplot(223)
plt.imshow(img_seuil, cmap="grey")
plt.title(f"Binarisé s : {s}")

plt.subplot(224)
plt.hist(img_seuil.reshape(-1,1), bins=255)
plt.show()

###########################################################
# Q10 :
###########################################################
# Rechercher du meilleur seuil grâce à la méthode d'Otsu
ret, otsu = cv2.threshold(img_ballon_gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# ret -> Valeur de seuil trouvé | otsu -> Image binarisé

# Affichage
plt.subplot(221)
plt.imshow(img_ballon_gray, cmap="grey")
plt.title("Ballon.jpg")

plt.subplot(222)
plt.hist(img_ballon_gray.reshape(-1,1), bins=255) # On vectorise la matrice sinon plt execute un histogramme de chaque ligne de notre matrice
plt.title("Histogramme")

plt.subplot(223)
plt.imshow(otsu, cmap="grey")
plt.title(f"Otsu : {ret}")

plt.subplot(224)
plt.hist(otsu.reshape(-1,1), bins=255)
plt.show()

###########################################################
# Q11 :
###########################################################
from functionsMainTD2 import MyHistColor

# Calcul de l'histogramme de l'image ballon à l'aide de ma fonction
my_hist_color = MyHistColor(img_ballon)
x= np.arange(0,256,1)
color_range=["red", "green", "blue"]

for i in range(3):
    plt.bar(x, my_hist_color[:,i], color=color_range[i], label=color_range[i])

plt.legend()
plt.title("Mon Histogramme des couleurs")
plt.show()

###########################################################
# Q12 :
###########################################################
# Comparaison avec la fonction calcHist de opencv 
color = ('r','g','b')
for i,col in enumerate(color):
     histr = cv2.calcHist([img_ballon],[i],None,[256],[0,256])
     plt.bar(x, histr[:,0],color = col, label=col)
     plt.xlim([0,256])
plt.title("Histogramme opencv")
plt.legend()
plt.show()

###########################################################
# Q13 :
###########################################################
from functionsMainTD2 import MyHistGrey

# Calcul de l'histogramme en niveau de gris avec ma fonction
my_hist_grey = MyHistGrey(img_ballon_gray)

plt.bar(x, my_hist_grey)
plt.title("Mon Histogramme Niveau de gris")
plt.show()

# Comparaison avec la fonction histogramme de matplotlib
plt.hist(img_ballon_gray.reshape(-1,1), bins=255)
plt.title("Histogramme matplotlib Niveau de gris")
plt.show()

###########################################################
# Q14 :
###########################################################
from functionsMainTD2 import HistCumul

my_hist_cumul = HistCumul(img_ballon_gray)

plt.plot(my_hist_cumul, color="green", label = "cumulé")
plt.legend()
plt.title("Mon Histogramme cumulé")
plt.show()

### Comparaison avec les méthodes opencv et numpy
# Calcul de l'histogramme avec opencv
hist = cv2.calcHist([img_ballon_gray], [0], None, [256], [0, 256])

# Calcul de l'histogramme cumulé avec numpy
hist_cumul = np.cumsum(hist)

# Affichage
plt.plot(hist_cumul, color='green', label='Cumulé')
plt.legend()
plt.title("Histogramme cumulé OpenCV/Numpy")
plt.show()

###########################################################
# Q15 :
###########################################################
file_contraste = "contraste1.png" # variable contenant le nom du fichier
img_contraste = cv2.imread(file_contraste,0) # Lecture en niveau de gris

from functionsMainTD2 import ExpensionDyn

# Affichage
plt.subplot(221)
plt.imshow(img_contraste, cmap="grey")
plt.title("contraste1.png")

plt.subplot(222)
plt.hist(img_contraste.reshape(-1,1), bins=255) # On vectorise la matrice sinon plt execute un histogramme de chaque ligne de notre matrice
plt.title("Histogramme")
plt.show()

img_contraste_expanded = ExpensionDyn(img_contraste)

# Affichage
plt.subplot(221)
plt.imshow(img_contraste_expanded, cmap="grey")
plt.title("contraste1.jpg expanded")

plt.subplot(222)
plt.hist(img_contraste_expanded.reshape(-1,1), bins=255) # On vectorise la matrice sinon plt execute un histogramme de chaque ligne de notre matrice
plt.title("Histogramme")

###########################################################
# Q16 :
###########################################################