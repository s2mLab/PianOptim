
from matplotlib import pyplot as plt
import pickle
import numpy as np

###########################################################
# Import results with pelvis rotZ
with open("Piano_results_3_phases.pckl",'rb') as file:
    new_dict = pickle.load(file)
###########################################################
# print(new_dict)
# print(new_dict == "Piano_results_3_phases.pckl")
# print(type(new_dict))
###########################################################
fig, ax = plt.subplots()
U = np.hstack((new_dict["controls"][i]["tau"] for i in range(3)))
T = np.hstack(np.linspace(0, 0.36574653, num=16), np.linspace(0.36574654, 0.36574654+0.10555556, num=16 ), np.linspace(0.36574654+0.10555557, 0.36574654+0.10555557+0.40625, num=16))
ax.plot(T, U)
###########################################################
# tau_thorax_rotX #
# plt.figure()
# time_x = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35]
# tau_thorax_rotX_with_pelvis_rotZ_y = [38496, 42000, 46752, 49320, 53200, 56000, 62316, 64928, 67317, 68748, 73752]
# plt.plot(time_x, tau_thorax_rotX_with_pelvis_rotZ_y)
#
# tau_thorax_rotX_without_pelvis_rotZ_y = [20046, 17100, 20000, 24744, 30500, 37732, 41247, 45372, 48876, 53850, 57287]
# plt.plot(time_x, tau_thorax_rotX_without_pelvis_rotZ_y)
#
# plt.xlabel('time')
# plt.ylabel('tau')
# plt.title('tau_thorax_rotX')
#
# plt.legend(['with Pelvis rotZ', 'without Pelvis rotZ'])
# plt.show()
#
# ####### Qdot_thorax_rotY #######
