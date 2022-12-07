
from ezc3d import c3d
import numpy as np
from matplotlib import pyplot as plt
from pyomeca import Analogs
from IPython import embed
from scipy.interpolate import interp1d
from scipy import interpolate
# Data Points
#     # So each frame of the animation is printed, each frame has 75 datapoints (3 xyz, residual value, cameras value)

c = c3d('012_BasFraStaB.c3d')

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
plt.title("Positions of the middle finger marker for 18 attacks  \n")
plt.xlabel('Frames from 0 to 10 000')
plt.ylabel('Position (mm)')
plt.grid(which='major', axis='x', color='black', linewidth=0.1)
plt.xticks(np.arange(0, 10000, step=50))

plt.axhline(y=0, color='black', linestyle='-')

plt.axvline(x=898, color='b', linestyle='-')
plt.axvline(x=956, color='b', linestyle='-')
plt.axvline(x=960, color='b', linestyle='-')
plt.axvline(x=967, color='b', linestyle='-')

plt.axvline(x=1242, color='b', linestyle='-')
plt.axvline(x=1289, color='b', linestyle='-')
plt.axvline(x=1293, color='b', linestyle='-')
plt.axvline(x=1302, color='b', linestyle='-')

plt.show()

# Plot the velocity vector depending on the time
Tsec = (np.linspace(0, (len(POS)/150)-3, num=(len(POS)-3)))
Tframe = (np.linspace(0, (len(POS))-3, num=(len(POS)-3)))
T = Tframe

plt.plot(T, velocity, 'tab:red')
plt.title("Velocity of the middle finger marker for 18 attacks \n")
plt.xlabel('Frame from 0 to 10 000')
plt.ylabel('Velocity (mm.s-1)')
plt.grid(which='major', axis='x', color='black', linewidth=0.1)
plt.xticks(np.arange(0, 10000, step=50))

plt.axhline(y=0, color='black', linestyle='-')

plt.axvline(x=898, color='b', linestyle='-')
plt.axvline(x=956, color='b', linestyle='-')
plt.axvline(x=960, color='b', linestyle='-')
plt.axvline(x=967, color='b', linestyle='-')

plt.axvline(x=1242, color='b', linestyle='-')
plt.axvline(x=1289, color='b', linestyle='-')
plt.axvline(x=1293, color='b', linestyle='-')
plt.axvline(x=1302, color='b', linestyle='-')
plt.show()

T = Tsec

# Data's, number of frames for each attacks
DebDesc = [898,	  1242,    1578,	1916,	2246,	2568,	2903,	3225,	3572,	3912,	4254,	4570,	4908,	5235,	5582,	5908,	6243,	6582]
Touched_key = [956, 1289, 1623, 1958, 2295, 2616, 2950, 3277, 3612, 3957, 4295, 4622, 4957, 5276, 5616, 5942, 6285, 6625]
DebFondTouche = [960,   1293,	1626,	1962,	2298,	2619,	2954,	3281,	3616,	3961,	4299,	4627,	4961,	5280,	5620,	5946,	6289,	6629]
FinFondTouche = [967,   1302,	1635,	1970,	2309,	2628,	2962,	3290,	3624,	3971,	4307,	4636,	4969,	5288,	5629,	5956,	6298,	6637]

# # Average of velocity of all attacks for each frame between the beginning and the touched_key
# All attacks between 0 and 100 sec
#
velocity_each_attack_arrays = np.zeros((20, 46))
for i in range(18):
    T = (np.linspace(0, 100, num=len(velocity[DebDesc[i]: Touched_key[i]])))
    y = velocity[DebDesc[i]: Touched_key[i]]
    n = len(velocity[DebDesc[i]: Touched_key[i]])
    f = interpolate.interp1d(T, y)
    Tnew = np.linspace(0, 100, 46)
    interpolated = f(Tnew)
    plt.plot(Tnew, interpolated, '.')
    velocity_each_attack_arrays[i, :] = interpolated


plt.title("Velocity profile for each attack between the beginning and the key \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()
print("46 velocities of 18 attacks : ", velocity_each_attack_arrays)

# Average of all attacks at each time step
average_velocities_profile = np.mean(velocity_each_attack_arrays, axis=0)
average_velocities_profile = average_velocities_profile
Tnew = np.linspace(0, 100, 46)

print("average_velocities_profile : ", average_velocities_profile)

plt.plot(Tnew, average_velocities_profile, ".")
plt.title("Average of velocity profiles between the beginning and the key \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()

# # # Average of velocity of all attacks for each frame between the touched_key and the fond_de_touche
# All attacks between 0 and 100 sec
# Interpolation done for 20 point
velocity_each_attack_arrays2 = np.zeros((20, 5))
for i in range(18):
    T2 = (np.linspace(0, 100, num=len(velocity[Touched_key[i]: DebFondTouche[i]])))
    y2 = velocity[Touched_key[i]: DebFondTouche[i]]
    n2 = len(velocity[Touched_key[i]: DebFondTouche[i]])
    f2 = interpolate.interp1d(T2, y2)
    Tnew2 = np.linspace(0, 100, 5)
    interpolated = f2(Tnew2)
    plt.plot(Tnew2, interpolated, '.')
    velocity_each_attack_arrays2[i, :] = interpolated


plt.title("Velocity profile for each attack between the key and the fond_de_touche \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()
print("5 velocities of 18 attacks : ", velocity_each_attack_arrays2)

# Average of all attacks at each time step
average_velocities_profile2 = np.mean(velocity_each_attack_arrays2, axis=0)
Tnew2 = np.linspace(0, 100, 5)

print("average_velocities_profile2 : ", average_velocities_profile2)

plt.plot(Tnew2, average_velocities_profile2, ".")
plt.title("Average of velocity profiles between the key and the fond_de_touche \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()

##### FELIPE VALUES #####
# # # For jeu_frappe during the attack inside the key

average_velocities_profile2 = [-0.698417100906372, -0.474601301515033, -0.368024758139809, -0.357349785081633, -0.367995643393795, -0.277969583506421, 0]
Tnew2 = np.linspace(0, 100, 7)

print("average_velocities_profile_7_nodes : ", average_velocities_profile2)

plt.plot(Tnew2, average_velocities_profile2, ".")
plt.title("Jeu 0_frappe \n \n Average of velocity profiles between the key and the fond_de_touche \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m.s-1)')
plt.show()

# # # To find the value at 75% of the attack inside the key
T = (np.linspace(0, 100, num=7))
average_velocities_profile2 = [-0.698417100906372, -0.474601301515033, -0.368024758139809, -0.357349785081633, -0.367995643393795, -0.277969583506421, 0]
n = len(average_velocities_profile2)
f = interpolate.interp1d(T, average_velocities_profile2)
Tnew3 = np.linspace(0, 100, 5)
interpolated = f(Tnew3)
plt.plot(Tnew3, interpolated, '.')

print("average_velocities_profile_5_nodes_75% : ", interpolated)

plt.axvline(x=75, color='black', linestyle='-')
plt.plot(Tnew3, interpolated, ".")
plt.title("Jeu 0_frappe \n \n Average of velocity profiles to find the velocity at 75% \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m.s-1)')
plt.show()

# # # FinRise for each attack
FinRise = []
for i in range(18):
    FinRise .append(FinFondTouche[i]+10)

# Max velocity vectors of each attack
velocity_max_for_20_attacks = []
for i in range(18):
    arr = np.array(velocity[DebDesc[i]:DebFondTouche[i]])
    velocity_max_for_20_attacks.append(max(arr))

# Average of max velocity vectors of each attack
average_velocity_max_20_attacks = sum(velocity_max_for_20_attacks)/18
print("Average of max velocity vectors of each attack : " + str(average_velocity_max_20_attacks) + " mm.s-1")

# Max velocity vectors of each rise
velocity_max_for_20_rises = []
for i in range(18):
    arr = np.array(velocity[FinFondTouche[i]:FinRise[i]])
    velocity_max_for_20_rises.append(max(arr))

# Average of max velocity vectors of each rise
average_velocity_max_20_rises = sum(velocity_max_for_20_rises)/18
print("Average of max velocity vectors of each rises : " + str(average_velocity_max_20_rises) + " mm.s-1")

# Average of the time in s during the attack
time_in_s_pushing = []
for i in range(18):
    time_in_s_pushing.append((DebFondTouche[i]-DebDesc[i])/150)

average_time_in_s_pushing = sum(time_in_s_pushing)/18
print("Average of the time during the attack : " + str(average_time_in_s_pushing*1000) + " ms")


# Average of the time in s in Fondetouche
time_in_s_Fonddetouche = []
for i in range(18):
    time_in_s_Fonddetouche.append((FinFondTouche[i]-DebFondTouche[i])/150)

average_time_in_s_Fonddetouche = sum(time_in_s_Fonddetouche)/18
print("Average of the time in Fondedetouche : " + str(average_time_in_s_Fonddetouche*1000) + " ms")

