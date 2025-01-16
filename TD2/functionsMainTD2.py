# -*- coding: utf-8 -*-
"""
Fonctions pour le script MainTD2.py

@author: kooky
"""

import numpy as np

def ImgNegative(img):
    # On inverse l'image -> Valeur max - Intensité lumineuse du pixel
    return np.max(img) - img