
from ezc3d import c3d
import numpy as np
from matplotlib import pyplot as plt
from pyomeca import Analogs
from IPython import embed
from scipy.interpolate import interp1d
from scipy import interpolate
# Data Points
#     # So each frame of the animation is printed, each frame has 75 datapoints (3 xyz, residual value, cameras value)

c = c3d('004_BasPreStaA.c3d')

# General velocity vector
POS = c['data']['points'][2][46][:]
velocity = []
for i in range(2, len(POS)-1):
    vel = ((POS[i+1] - POS[i-1]) / (2/150))
    velocity.append(vel)
velocity = [ele for ele in velocity]
print(velocity)
# Explanations :
# 2/150 bc we want the velocity between i+1 et i-1 frame
# We take the absolute velocity values

# Plot the position depending on the time
T = (np.linspace(0, (len(POS)/150), num=(len(POS))))
plt.plot(T, POS, 'tab:red')
plt.title("Position for 20 attacks  \n")
plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.show()

# Plot the velocity vector depending on the time
T = (np.linspace(0, (len(POS)/150)-3, num=(len(POS)-3)))
plt.plot(T, velocity, 'tab:red')
plt.title("Velocity Vector for 20 attacks \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()

# Data's, number of frames for each attacks
DebDesc = [473, 879, 1286, 1676, 2048, 2411, 2777, 3179, 3551, 3927, 4305, 4667, 5042, 5434, 5832, 6178, 6549, 6918, 7270, 7592]
DebFondTouche = [485, 890, 1296, 1687, 2062, 2423, 2790, 3188, 3560, 3936, 4319, 4678, 5052, 5445, 5846, 6195, 6563, 6932, 7281, 7607]
FinFondTouche = [498, 897, 1308, 1693, 2069, 2430, 2800, 3195, 3568, 3944, 4328, 4685, 5058, 5453, 5855, 6202, 6569, 6936, 7287, 7613]

## Average of velocity of all atacks for each frame during the attack
# All attacks between 0 and 100 sec.
# And interpolation for 20 points

for i in range(20):
    T = (np.linspace(0, 100, num=len(velocity[DebDesc[i]: DebFondTouche[i]])))
    y = velocity[DebDesc[i]: DebFondTouche[i]]
    n = len(velocity[DebDesc[i]: DebFondTouche[i]])
    f = interpolate.interp1d(T, y, kind=5, fill_value='extrapolate')
    #interp_func = interp1d(T, y)
    #newarr = interp_func(np.arange(0, 100, 5))
    # df_interpol = df.groupby('house') \
    #     .resample('D') \
    #     .mean()
    # df_interpol['readvalue'] = df_interpol['readvalue'].interpolate()
    # df_interpol.head(4)

plt.plot(T, f(T), "r")
# plt.plot(T, y, 'ro')
# plt.plot(T, y, 'b')
plt.title("Velocity Vector for each attacks \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()


# Velocitiies of the 9th attack that we use for the pushed attack.
print('Velocity_attack_9 : ', velocity[3551:3560])

# With corrections : correct_1 = delete 5 frames to every DebDesc, correct_2 = only attacks that have less than 10 f.
DebDesc_correct_1 = []
for i in range(20):
    DebDesc_correct_1.append(DebDesc[i]+5)

DebDesc_correct_2 = []
DebFondTouche_correct_2 = []
FinFondTouche_correct_2 = []
for i in range(20):
    a = DebFondTouche[i] - DebDesc_correct_1[i]
    if a < 10:
        DebDesc_correct_2.append(DebDesc_correct_1[i])
        DebFondTouche_correct_2.append(DebFondTouche[i])
        FinFondTouche_correct_2.append(FinFondTouche[i])

#  FinRise_correct_2 for each attack
FinRise_correct_2 = []
for i in range(18):
    FinRise_correct_2.append(FinFondTouche_correct_2[i]+10)

# Max velocity vectors of each attack
velocity_max_for_20_attacks = []
for i in range(18):
    arr = np.array(velocity[DebDesc_correct_2[i]:DebFondTouche_correct_2[i]])
    velocity_max_for_20_attacks.append(max(arr))

# Average of max velocity vectors of each attack
average_velocity_max_20_attacks = sum(velocity_max_for_20_attacks)/18
print("Average of max velocity vectors of each attack : " + str(average_velocity_max_20_attacks) + " mm.s-1")

# Max velocity vectors of each rise
velocity_max_for_20_rises = []
for i in range(18):
    arr = np.array(velocity[FinFondTouche_correct_2[i]:FinRise_correct_2[i]])
    velocity_max_for_20_rises.append(max(arr))

# Average of max velocity vectors of each rise
average_velocity_max_20_rises = sum(velocity_max_for_20_rises)/18
print("Average of max velocity vectors of each rises : " + str(average_velocity_max_20_rises) + " mm.s-1")

# Average of the time in s during the attack
time_in_s_pushing = []
for i in range(18):
    time_in_s_pushing.append((DebFondTouche_correct_2[i]-DebDesc_correct_2[i])/150)

average_time_in_s_pushing = sum(time_in_s_pushing)/18
print("Average of the time during the attack : " + str(average_time_in_s_pushing*1000) + " ms")


# Average of the time in s in Fondetouche
time_in_s_Fonddetouche = []
for i in range(18):
    time_in_s_Fonddetouche.append((FinFondTouche_correct_2[i]-DebFondTouche_correct_2[i])/150)

average_time_in_s_Fonddetouche = sum(time_in_s_Fonddetouche)/18
print("Average of the time in Fondedetouche : " + str(average_time_in_s_Fonddetouche*1000) + " ms")
