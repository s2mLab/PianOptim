
from ezc3d import c3d
import numpy as np
from matplotlib import pyplot as plt

c = c3d('BasPreStaA.c3d')

## Hand vectors far away from the origin
STYLrad = []
meta2 = []
meta5 = []
STYLulna = []

for i in range(3):
    meta2.append(c['data']['points'][i][41][200])
    meta5.append(c['data']['points'][i][47][200])
    STYLrad.append(c['data']['points'][i][30][200])
    STYLulna.append(c['data']['points'][i][33][200])

print("Hand vectors far away from the origin : ", "\n", STYLrad, "\n", meta2, "\n", meta5, "\n", STYLulna, "\n")

## Rapproché a l'origine avec STYLrad a l'origine

STYLrad_new = [0, 0, 0]
meta2_new = []
meta5_new = []
STYLula_new = []
for i in range(3):
    meta2_new.append(abs(meta2[i]-STYLrad[i]))
    meta5_new.append(abs(meta5[i]-STYLrad[i]))
    STYLula_new.append(abs(STYLulna[i] - STYLrad[i]))
print("Rapproché de l'origine avec STYLrad a l'origine : ", "\n", STYLrad_new, "\n", meta2_new, "\n", meta5_new, "\n", STYLula_new, "\n")

## Longueurs des vecteurs aux 2 positions
a = []
anew = []
b = []
bnew = []
c = []
cnew = []
d = []
dnew = []
for i in range(3):
    a.append(abs(meta2[i]-meta5[i]))
    anew.append(abs(meta2_new[i]-meta5_new[i]))
    b.append(abs(meta5[i]-STYLulna[i]))
    bnew.append(abs(meta5_new[i]-STYLula_new[i]))
    c.append(abs(STYLulna[i]-STYLrad[i]))
    cnew.append(abs(STYLula_new[i]-STYLrad_new[i]))
    d.append(abs(STYLrad[i]-meta2[i]))
    dnew.append(abs(STYLrad_new[i]-meta2_new[i]))
print("Longueurs des vecteurs aux 2 positions : ", "\n", a, "\n", anew, "\n", "\n", b, "\n", bnew, "\n", "\n", c, "\n", cnew, "\n", "\n", d, "\n", dnew)

