
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

# Plot the position depending on the time
Tsec = (np.linspace(0, (len(POS)/150), num=(len(POS))))
Tframe = (np.linspace(0, (len(POS)), num=(len(POS))))
T = Tframe

plt.plot(T, POS, 'tab:red')
plt.title("Positions of the middle finger marker for 20 attacks  \n")
plt.xlabel('Frames from 0 to 10 000')
plt.ylabel('Position (mm)')
plt.grid(which='major', axis='x', color='black', linewidth=0.1)
plt.xticks(np.arange(0, 10000, step=50))

plt.axhline(y=0, color='black', linestyle='-')

plt.axvline(x=2048, color='b', linestyle='-')
plt.axvline(x=2062, color='b', linestyle='-')
plt.axvline(x=2069, color='b', linestyle='-')

plt.axvline(x=2411, color='b', linestyle='-')
plt.axvline(x=2423, color='b', linestyle='-')
plt.axvline(x=2430, color='b', linestyle='-')
# +5, -2
plt.show()

# Plot the velocity vector depending on the time
Tsec = (np.linspace(0, (len(POS)/150)-3, num=(len(POS)-3)))
Tframe = (np.linspace(0, (len(POS))-3, num=(len(POS)-3)))
T = Tframe

plt.plot(T, velocity, 'tab:red')
plt.title("Velocity of the middle finger marker for 20 attacks \n")
plt.xlabel('Frames from 0 to 10 000')
plt.ylabel('Velocity (mm.s-1)')
plt.grid(which='major', axis='x', color='black', linewidth=0.1)
plt.xticks(np.arange(0, 10000, step=50))

plt.axhline(y=0, color='black', linestyle='-')

plt.axvline(x=2048, color='b', linestyle='-')
plt.axvline(x=2062, color='b', linestyle='-')
plt.axvline(x=2069, color='b', linestyle='-')

plt.axvline(x=2411, color='b', linestyle='-')
plt.axvline(x=2423, color='b', linestyle='-')
plt.axvline(x=2430, color='b', linestyle='-')

plt.show()

T = Tsec

# Data's, number of frames for each attacks
DebDesc = [473, 879, 1286, 1676, 2048, 2411, 2777, 3179, 3551, 3927, 4305, 4667, 5042, 5434, 5832, 6178, 6549, 6918, 7270, 7592]
DebFondTouche = [485, 890, 1296, 1687, 2062, 2423, 2790, 3188, 3560, 3936, 4319, 4678, 5052, 5445, 5846, 6195, 6563, 6932, 7281, 7607]
FinFondTouche = [498, 897, 1308, 1693, 2069, 2430, 2800, 3195, 3568, 3944, 4328, 4685, 5058, 5453, 5855, 6202, 6569, 6936, 7287, 7613]

# # # # Average of velocity of all attacks for each frame during the attack
# # All attacks between 0 and 100 sec
# # Interpolation done for 20 point
# # velocity_each_attack_arrays = []
# velocity_each_attack_arrays = np.zeros((20, 8))
# for i in range(20):
#     T = (np.linspace(0, 100, num=len(velocity[DebDesc[i]: DebFondTouche[i]])))
#     y = velocity[DebDesc[i]: DebFondTouche[i]]
#     n = len(velocity[DebDesc[i]: DebFondTouche[i]])
#     f = interpolate.interp1d(T, y)
#     Tnew = np.linspace(0, 100, 8)
#     interpolated = f(Tnew)
#     plt.plot(Tnew, interpolated, '.')
#     velocity_each_attack_arrays[i, :] = interpolated
#
#
# plt.title("Velocity profile of the middle finger marker for each attack \n")
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (mm.s-1)')
# plt.show()
# print("20 velocities of 20 attacks : ", velocity_each_attack_arrays)
#
# # Average of all attacks at each time step
# average_velocities_profile = np.mean(velocity_each_attack_arrays, axis=0)
# average_velocities_profile = average_velocities_profile
# Tnew = np.linspace(0, 100, 8)
#
# print("average_velocities_profile : ", average_velocities_profile)
#
# plt.plot(Tnew, average_velocities_profile, ".")
# plt.title("Average of velocity profiles of the middle finger marker of 20 attacks \n")
# plt.xlabel('Time (s)')
# plt.ylabel('Velocity (mm.s-1)')
# plt.show()

##### FELIPE VALUES #####
# # # For jeu_presse during the attack inside the key
average_velocities_profile2 = [0, -0.113772161006927, -0.180575996580578, -0.270097219830468, -0.347421549388341, -0.290588704744975, -0.0996376128423782, 0]
Tnew2 = np.linspace(0, 100, 8)

print("average_velocities_profile_8_nodes : ", average_velocities_profile2)

plt.plot(Tnew2, average_velocities_profile2, ".")
plt.title("Jeu presse \n \n Average of velocity profiles between the key and the fond_de_touche \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m.s-1)')
plt.show()

# # # To find the value at 75% of the attack inside the key
T = (np.linspace(0, 100, num=8))
average_velocities_profile2 = [0, -0.113772161006927, -0.180575996580578, -0.270097219830468, -0.347421549388341, -0.290588704744975, -0.0996376128423782, 0]
n = len(average_velocities_profile2)
f = interpolate.interp1d(T, average_velocities_profile2)
Tnew3 = np.linspace(0, 100, 5)
interpolated = f(Tnew3)
plt.plot(Tnew3, interpolated, '.')

print("average_velocities_profile_5_nodes_75% : ", interpolated)

plt.axvline(x=75, color='black', linestyle='-')
plt.plot(Tnew3, interpolated, ".")
plt.title("Jeu presse \n \n Average of velocity profiles to find the velocity at 75% \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m.s-1)')
plt.show()

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

##### FELIPE VALUES #####
# For jeu_presse during the attack inside the key
# 0
# -113.772161006927
# -180.575996580578
# -270.097219830468
# -347.421549388341
# -290.588704744975
# -99.6376128423782
# 0