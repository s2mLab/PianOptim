
from ezc3d import c3d
import numpy as np

c = c3d('/home/mickaelbegon/Documents/Stage_Mathilde/programation/PianOptim/Fichier_C3D/BasPreStaA.c3d')

# General velocity vector
POS = c['data']['points'][2][46][:]
velocity = []
for i in range(2, len(POS)-1):
   vel = ((POS[i+1] - POS[i-1] * 1000) / 2/150)
   velocity.append(vel)

# Max velocity vector for each attack
DebDesc = [473,	879, 1286, 1676, 2048, 2411, 2777, 3179, 3551, 3927, 4305, 4667, 5042, 5434, 5832, 6178, 6549, 6918, 7270, 7592]
DebutFondTouche = [485,	890, 1296, 1687, 2062, 2423, 2790, 3188, 3560, 3936, 4319, 4678, 5052, 5445, 5846, 6195, 6563, 6932, 7281, 7607]
FinFondTouche = [492, 897, 1304, 1693, 2069, 2430, 2798, 3195, 3568, 3944, 4328, 4685, 5058, 5453, 5855, 6202, 6569, 6936, 7287, 7613]

velocity_max_for_20 = []

for i in range(19):
    arr = np.array(velocity[DebDesc[i]:DebutFondTouche[i]])
    velocity_max_for_20.append(min(arr))

# Average of max velocity vectors of each attack
average_velocity_max_20_notes = sum(velocity_max_for_20)/20
print("Average of max velocity vectors of each attack : " + str(- average_velocity_max_20_notes))


# Average of the time in s in Fondetouche
time_in_s_Fonddetouche = []
for i in range(19):
    time_in_s_Fonddetouche.append((FinFondTouche[i]-DebutFondTouche[i])/150)

average_time_in_s_Fonddetouche = sum(time_in_s_Fonddetouche)/20
print("Average of the time in s in Fondedetouche : " + str(average_time_in_s_Fonddetouche))






