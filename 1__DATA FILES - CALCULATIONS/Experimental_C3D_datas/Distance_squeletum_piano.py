
from ezc3d import c3d

c3d = c3d('004_BasPreStaA.c3d')

# middle_finger : index [46]
# thl_thorax_side : index [16]
# XIPH_thorax_front : index [11]
# XIPHback_thorax_back : index [15]

thl_thorax_side_x = (c3d['data']['points'][0][16][2048])
thl_thorax_side_y = (c3d['data']['points'][1][16][2048])
thl_thorax_side_z = (c3d['data']['points'][2][16][2048])

XIPH_thorax_front_x = (c3d['data']['points'][0][11][2048])
XIPH_thorax_front_y = (c3d['data']['points'][1][11][2048])
XIPH_thorax_front_z = (c3d['data']['points'][2][11][2048])

XIPHback_thorax_back_x = (c3d['data']['points'][0][15][2048])
XIPHback_thorax_back_y = (c3d['data']['points'][1][15][2048])
XIPHback_thorax_back_z = (c3d['data']['points'][2][15][2048])

print("\n")
print("// On the experimentation frame //")
distance1 = []
for i in range(3):
    distance1.append(c3d['data']['points'][i][16][2048])
print("thl_thorax_side : ", distance1)

distance2 = []
for i in range(3):
    distance2.append(c3d['data']['points'][i][11][2048])
print("XIPH_thorax_front : ", distance2)

distance3 = []
for i in range(3):
    distance3.append(c3d['data']['points'][i][15][2048])
print("XIPHback_thorax_back : ", distance3)

distance = []
for i in range(3):
    distance.append(c3d['data']['points'][i][16][2048] - c3d['data']['points'][i][46][2048])
print("La distance entre le côté gauche du squelette et la touche LA est :", distance)

x = XIPHback_thorax_back_x - XIPH_thorax_front_x
y = XIPHback_thorax_back_y - XIPH_thorax_front_y
z = XIPHback_thorax_back_z - XIPH_thorax_front_z
print("Vector [XIPH_thorax_front et XIPHback_thorax_back] :", "x : ", x, "y : ", y, "z : ", z)
print("\n")
print("// On the model frame //")
print("Position of the thorax side marker :", z*0.001, (x*0.001-0.145), y*0.001)




