from ezc3d import c3d
import numpy
from numpy import linalg as la
from matplotlib import pyplot as plt

c3d = c3d("004_BasPreStaA.c3d")

## Hand vectors far away from the origin
STYLrad = []
meta2 = []
meta5 = []
STYLulna = []
STYLrad_up = []
STYLulna_up = []
for i in range(3):
    meta2.append(c3d["data"]["points"][i][41][200])
    meta5.append(c3d["data"]["points"][i][47][200])
    STYLrad.append(c3d["data"]["points"][i][30][200])
    STYLrad_up.append(c3d["data"]["points"][i][31][200])
    STYLulna.append(c3d["data"]["points"][i][33][200])
    STYLulna_up.append(c3d["data"]["points"][i][32][200])

print("Hand vectors far away from the origin : ", "\n", STYLrad, "\n", meta2, "\n", meta5, "\n", STYLulna, "\n")
####

milieuxxxx = STYLrad_up[0] - STYLrad[0]
milieuyyyy = STYLrad_up[1] - STYLrad[1]
milieuzzzz = STYLrad_up[2] - STYLrad[2]
milieu_rad_radup = (milieuxxxx**2 + milieuyyyy**2 + milieuzzzz**2) ** (1 / 2)
print("Le vecteur de Rad et Rad_up ramené a l'origine : ", milieuxxxx, milieuyyyy, milieuzzzz)
print("La moitié de la norme du vecteur : ", milieu_rad_radup / 2)

milieuxxx = STYLulna_up[0] - STYLulna[0]
milieuyyy = STYLulna_up[1] - STYLulna[1]
milieuzzz = STYLulna_up[2] - STYLulna[2]
milieu_ulna_ulnaup = (milieuxxx**2 + milieuyyy**2 + milieuzzz**2) ** (1 / 2)
print("Le vecteur de Ulna et Ulna_up ramené a l'origine : ", milieuxxx, milieuyyy, milieuzzz)
print("La moitié de la norme du vecteur : ", milieu_ulna_ulnaup / 2)

print(
    "Norme de la distance origin / centre de l'arriere de la main : ",
    ((milieu_rad_radup / 2 + milieu_ulna_ulnaup / 2) / 2),
)

####
## Rapproché a l'origine avec STYLrad a l'origine

STYLrad_new = [0, 0, 0]
meta2_new = []
meta5_new = []
STYLula_new = []
for i in range(3):
    meta2_new.append(abs(meta2[i] - STYLrad[i]))
    meta5_new.append(abs(meta5[i] - STYLrad[i]))
    STYLula_new.append(abs(STYLulna[i] - STYLrad[i]))
print(
    "Rapproché a l'origine avec STYLrad a l'origine : ",
    "\n",
    STYLrad_new,
    "\n",
    meta2_new,
    "\n",
    meta5_new,
    "\n",
    STYLula_new,
    "\n",
)

## Longueurs des vecteurs aux 2 positions
a = []
anew = []
b = []
bnew = []
c = []
cnew = []
d = []
dnew = []
e = []
for i in range(3):
    a.append(abs(meta2[i] - STYLrad[i]))
    anew.append(abs(meta2_new[i] - STYLrad_new[i]))
    b.append(abs(meta5[i] - STYLrad[i]))
    bnew.append(abs(meta5_new[i] - STYLrad_new[i]))
    c.append(abs(STYLulna[i] - STYLrad[i]))
    cnew.append(abs(STYLula_new[i] - STYLrad_new[i]))
    d.append(abs(STYLrad[i] - STYLrad[i]))
    dnew.append(abs(STYLrad_new[i] - STYLrad_new[i]))

# print("Longueurs des vecteurs aux 2 positions : ", "\n", a, "\n", anew, "\n", "\n", b, "\n", bnew, "\n", "\n", c, "\n", cnew, "\n", "\n", d, "\n", dnew)
# print(e)


# e.append(abs(meta2[i]-meta5[i]))
# norm = la.norm(e)
# print('The value of norm is:')
# print(norm)

## com for the hand
moyenne_x = (0.015898544311523433 + 0.08811069869995117 + 0.09361060333251953) / 4
moyenne_y = (-0.019521423339843746 - 0.022297187805175776 + 0.011576980590820317) / 4
moyenne_z = (0.01782612609863281 - 0.001428512573242188 - 0.009257064819335938) / 4

print("com for the hand: ", moyenne_x, moyenne_y, moyenne_z)

## Inertie matrice finger # r = 1
R = 1
m = 0.038125
h = 0.063
x = (m * R**2) / 2
y = m * (((R**2) / 4) + (h**2 / 12))
z = m * (((R**2) / 4) + (h**2 / 12))
print("Inertie matrice finger : ", x, y, z)

## Milieu du segment STYLrad et STYLulna
ar = -0.14411060333251952
br = 0.009923019409179681
cr = 0.047
au = 0.015898544311523433 + ar
bu = -0.019521423339843746 + br
cu = 0.01782612609863281 + cr
milieux = (ar + au) / 2
milieuy = (br + bu) / 2
milieuz = (cr + cu) / 2
print("Milieu du segment STYLrad et STYLulna", milieux, milieuy, milieuz)

print((-0.1361613311767578 - ar))
print((0.00016230773925780817 - br))
print((0.055913063049316404 - cr))

## Milieu du segment meta2 et meta5
a = -0.1361613311767578
b = 0.00016230773925780817
c = 0.055913063049316404

a2 = 0.08566133117675781 + a
b2 = 0.02133769226074219 + b
c2 = -0.018170127868652342 + c
a5 = 0.08016142654418945 + a
b5 = -0.012536476135253903 + b
c5 = -0.010341575622558592 + c
milieuxx = (a2 + a5) / 2
milieuyy = (b2 + b5) / 2
milieuzz = (c2 + c5) / 2
print("Milieu du segment meta2 et meta5", milieuxx, milieuyy, milieuzz)

print((a2 - a5) / 2)
print((b2 - b5) / 2)
print((c2 - c5) / 2)
