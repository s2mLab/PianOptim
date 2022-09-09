
from matplotlib import pyplot as plt
import pickle
import matplotlib.patches as mpatches
import numpy as np

# Import results with pelvis rotZ
with open("Piano_results_3_phases.pckl",'rb') as file:
    new_dict = pickle.load(file)

with open("Piano_results_3_phases_without_pelvis_rotZ.pckl",'rb') as file:
    new_dict2 = pickle.load(file)

# Print the dic ###########################################
# print(new_dict)
# print(new_dict == "Piano_results_3_phases.pckl")
# print(type(new_dict))
###########################################################
U = np.hstack((new_dict["controls"][i]["tau"][:,:-1] if i < 2 else new_dict["controls"][i]["tau"] for i in range(3)))
U2 = np.hstack((new_dict2["controls"][i]["tau"][:,:-1] if i < 2 else new_dict2["controls"][i]["tau"] for i in range(3)))
T = np.hstack((np.linspace(0, 0.36574653, num=15), np.linspace(0.36574653, 0.36574653+0.10555556, num=15), np.linspace(0.36574653+0.10555556, 0.36574653+0.10555556+0.40625, num=16)))

fig, axs = plt.subplots(3, 4)

axs[0, 0].plot(T, U[0, :], 'tab:red')
axs[0, 0].set_title("Tau_pelvis_rotZ_with_pelvis_rotZ")
# il n y a plus de DoF pour le pelvis

axs[0, 1].plot(T, U[1, :], 'tab:red')
axs[0, 1].set_title("Tau_thorax_rotX_with_pelvis_rotZ")
axs[0, 1].plot(T, U2[0, :], 'tab:blue')

axs[0, 2].plot(T, U[2, :], 'tab:red')
axs[0, 2].set_title("Tau_thorax_rotY_with_pelvis_rotZ")
axs[0, 2].plot(T, U2[1, :], 'tab:blue')

axs[0, 3].plot(T, U[3, :], 'tab:red')
axs[0, 3].set_title("Tau_thorax_rotZ_with_pelvis_rotZ")
axs[0, 3].plot(T, U2[2, :], 'tab:blue')

axs[1, 0].plot(T, U[4, :], 'tab:red')
axs[1, 0].set_title("Tau_humerus_right_rotX_with_pelvis_rotZ")
axs[1, 0].plot(T, U2[3, :], 'tab:blue')

axs[1, 1].plot(T, U[5, :], 'tab:red')
axs[1, 1].set_title("Tau_humerus_right_rotY_with_pelvis_rotZ")
axs[1, 1].plot(T, U2[4, :], 'tab:blue')

axs[1, 2].plot(T, U[6, :], 'tab:red')
axs[1, 2].set_title("Tau_humerus_right_rotZ_with_pelvis_rotZ")
axs[1, 2].plot(T, U2[5, :], 'tab:blue')

axs[1, 3].plot(T, U[7, :], 'tab:red')
axs[1, 3].set_title("Tau_ulna_effector_right_rotZ_with_pelvis_rotZ")
axs[1, 3].plot(T, U2[6, :], 'tab:blue')

axs[2, 0].plot(T, U[8, :], 'tab:red')
axs[2, 0].set_title("Tau_radius_effector_right_rotY_with_pelvis_rotZ")
axs[2, 0].plot(T, U2[7, :], 'tab:blue')

axs[2, 1].plot(T, U[9, :], 'tab:red')
axs[2, 1].set_title("Tau_hand_right_rotX_with_pelvis_rotZ")
axs[2, 1].plot(T, U2[8, :], 'tab:blue')

axs[2, 2].plot(T, U[10, :], 'tab:red')
axs[2, 2].set_title("Tau_hand_right_rotZ_with_pelvis_rotZ")
axs[2, 2].plot(T, U2[9, :], 'tab:blue')

axs[2, 3].set_title("NOBODY")

# principal and axe titles
for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Tau')
fig.suptitle('Torque of each DoF depending of the time for multiple contexts', fontname="Times New Roman", fontweight="bold")

# legends
line_labels = ["With every DoF", "Without pelvis_rotZ DoF"]
l1 = axs[0, 0].plot(T, U[0, :], color="red")
l2 = axs[0, 1].plot(T, U2[0, :], color="green")
fig.legend([l1, l2],     # The line objects
           labels=line_labels,   # The labels for each line
           loc="upper right",   # Position of legend
           borderaxespad=0.8,    # Small spacing around legend box
           title="Legend Title")  # Title for the legend

# show the graph
fig.tight_layout()
plt.show()

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
