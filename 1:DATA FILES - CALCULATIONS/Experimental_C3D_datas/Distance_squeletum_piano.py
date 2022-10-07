
from ezc3d import c3d

c3d = c3d('BasPreStaA.c3d')
## distance between the left side of the squeltum and the LA note
# thl_thorax_side : index [16]
# middle_finger : index [46]
distance = []
for i in range(3):
    distance.append(c3d['data']['points'][i][16][2048] - c3d['data']['points'][i][46][2048])
print("La distance entre le côté gauche du squelette et la touche LA est :", distance)

## distance between the middle of the squeletum and left side of the squeletum
# thl_thorax_side : index [16]
# XIPH_thorax_front : index [11]
# XIPHback_thorax_back : index [15]

distance3 = []
for i in range(3):
    distance3.append(c3d['data']['points'][i][11][2048])
print("XIPH_thorax_front : ", distance3)

distance4 = []
for i in range(3):
    distance4.append(c3d['data']['points'][i][15][2048])
print("XIPHback_thorax_back : ", distance4)

thl_thorax_side_x = (c3d['data']['points'][0][16][2048])

XIPH_thorax_front_x = (c3d['data']['points'][0][11][2048])
XIPH_thorax_front_y = (c3d['data']['points'][1][11][2048])
XIPH_thorax_front_z = (c3d['data']['points'][2][11][2048])

XIPHback_thorax_back_x = (c3d['data']['points'][0][15][2048])
XIPHback_thorax_back_y = (c3d['data']['points'][1][15][2048])
XIPHback_thorax_back_z = (c3d['data']['points'][2][15][2048])

x = XIPH_thorax_front_x - thl_thorax_side_x
y = XIPHback_thorax_back_y - XIPH_thorax_front_y
z = XIPH_thorax_front_z
print("x : ", x, "y : ", y, "z : ", z)



