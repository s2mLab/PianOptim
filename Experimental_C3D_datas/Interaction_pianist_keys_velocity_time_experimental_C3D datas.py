
from ezc3d import c3d
import numpy as np
from matplotlib import pyplot as plt

# Data Points
#     # So each frame of the animation is printed, each frame has 75 datapoints (3 xyz, residual value, cameras value)

c = c3d('BasPreStaA.c3d')

# General velocity vector
POS = c['data']['points'][2][46][:]
velocity = []
for i in range(2, len(POS)-1):
   vel = ((POS[i+1] - POS[i-1]) / (2/150))
   velocity.append(vel)
velocity = [abs(ele) for ele in velocity]
# Explenations :
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

# Max velocity vector for each attack (min bc the velocity is negative)
DebDesc = [473, 879, 1286, 1676, 2048, 2411, 2777, 3179, 3551, 3927, 4305, 4667, 5042, 5434, 5832, 6178, 6549, 6918, 7270, 7592]
DebFondTouche = [485,	890, 1296, 1687, 2062, 2423, 2790, 3188, 3560, 3936, 4319, 4678, 5052, 5445, 5846, 6195, 6563, 6932, 7281, 7607]
FinFondTouche = [498, 897, 1308, 1693, 2069, 2430, 2800, 3195, 3568, 3944, 4328, 4685, 5058, 5453, 5855, 6202, 6569, 6936, 7287, 7613]
# With corrections
# DebDesc = [879, 1676, 2048, 2411, 4305, 4667, 5042, 5832, 6178, 6549, 6918, 7270, 7592]
# DebFondTouche = [890, 1687, 2062, 2423, 4319, 4678, 5052, 5846, 6195, 6563, 6932, 7281, 7607]
# FinFondTouche = [897, 1693, 2069, 2430, 4328, 4685, 5058, 5855, 6202, 6569, 6936, 7287, 7613]

# With corrections : correct_1 = delete 5 frames to every DebDesc, correct_2 = only attacks that have less than 10 f.
DebDesc_correct_1 = []
for i in range(20):
    DebDesc_correct_1.append(DebDesc[i]+5)
print(DebDesc_correct_1)

DebDesc_correct_2 = []
DebFondTouche_correct_2 = []
FinFondTouche_correct_2 = []
for i in range(20):
    a = DebFondTouche[i] - DebDesc_correct_1[i]
    if a < 10:
        DebDesc_correct_2.append(DebDesc_correct_1[i])
        DebFondTouche_correct_2.append(DebFondTouche[i])
        FinFondTouche_correct_2.append(FinFondTouche[i])

# max velocity vectors of each attack
velocity_max_for_20 = []
for i in range(18):
    arr = np.array(velocity[DebDesc_correct_2[i]:DebFondTouche_correct_2[i]])
    velocity_max_for_20.append(max(arr))

# Average of max velocity vectors of each attack
average_velocity_max_20_notes = sum(velocity_max_for_20)/18
print("Average of max velocity vectors of each attack : " + str(average_velocity_max_20_notes) + " mm.s-1")


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






