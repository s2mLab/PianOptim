
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
# We take the absolute velocity values

# Plot the position depending on the time
T = (np.linspace(0, (len(POS)/150), num=(len(POS))))
plt.plot(T, POS, 'tab:red')
plt.title("Positions of the middle finger marker for 18 attacks  \n")
plt.xlabel('Time (s)')
plt.ylabel('Position (mm)')
plt.show()

# Plot the velocity vector depending on the time
T = (np.linspace(0, (len(POS)/150)-3, num=(len(POS)-3)))
plt.plot(T, velocity, 'tab:red')
plt.title("Velocity of the middle finger marker for 18 attacks \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()

# Data's, number of frames for each attacks
DebDesc = [898,	  1242,    1578,	1916,	2246,	2568,	2903,	3225,	3572,	3912,	4254,	4570,	4908,	5235,	5582,	5908,	6243,	6582]
DebFondTouche = [960,   1293,	1626,	1962,	2298,	2619,	2954,	3281,	3616,	3961,	4299,	4627,	4961,	5280,	5620,	5946,	6289,	6629]
FinFondTouche = [967,   1302,	1635,	1970,	2309,	2628,	2962,	3290,	3624,	3971,	4307,	4636,	4969,	5288,	5629,	5956,	6298,	6637]

# # # Average of velocity of all attacks for each frame during the attack
# All attacks between 0 and 100 sec
# Interpolation done for 20 point
velocity_each_attack_arrays = []
for i in range(18):
    T = (np.linspace(0, 100, num=len(velocity[DebDesc[i]: DebFondTouche[i]])))
    y = velocity[DebDesc[i]: DebFondTouche[i]]
    n = len(velocity[DebDesc[i]: DebFondTouche[i]])
    f = interpolate.interp1d(T, y)
    Tnew = np.linspace(0, 100, 50)
    interpolated = f(Tnew)
    plt.plot(Tnew, interpolated, '.')
    velocity_each_attack_arrays.append(interpolated)


plt.title("Velocity profile of the middle finger marker for each attack \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()
print("20 velocities of 18 attacks : ", velocity_each_attack_arrays)
# Average of all attacks at each time step
average_velocities_profile = []
a = []
for i in range(50):
    for j in range(18):
        a.append(velocity_each_attack_arrays[j][i])
    average_velocities_profile.append(sum(a) / 50)
    a.clear()
print("average_velocities_profile", average_velocities_profile)

Tnew = np.linspace(0, 100, 50)
plt.plot(Tnew, average_velocities_profile, ".")
plt.title("Average of velocity profiles of the middle finger marker of 18 attacks \n")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (mm.s-1)')
plt.show()

#  FinRise for each attack
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

