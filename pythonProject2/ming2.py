# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 09:26:12 2023

@author: Ming Yao
"""
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def to_bin(dst):
    im = Image.open(dst, 'r')
    width, height = im.size
    pixel = list(im.getdata())
    arr = []
    for i in range(0, len(pixel)):
        count = 0
        for rgb in range(0, 3):
            count += pixel[i][rgb]
        if(count < 470):
            arr.append(int(1))
        else:
            arr.append(int(0))
    return arr


img = Image.new('RGB', (10, 10), 'white')
drawobj = ImageDraw.Draw(img)
five = to_bin("img/3/1.png")
print(five)
count = 0
for i in range(0, 10):
    for j in range(0, 10):
        if(five[count] == 1):
            drawobj.point([j, i], 'black')
        else:
            drawobj.point([j, i], 'white')
        count += 1
plt.imshow(img)