
from matplotlib import pyplot as plt
import pickle
import numpy as np

# Import results with pelvis rotZ
with open("Piano_results_3_phases.pckl",'rb') as file:
    new_dict = pickle.load(file)
with open("Piano_results_3_phases_without_pelvis_rotZ.pckl",'rb') as file:
    new_dict2 = pickle.load(file)
with open("Piano_results_3_phases_without_pelvis_rotZ_and_thorax.pckl", 'rb') as file:
    new_dict3 = pickle.load(file)

# Print the dic ###########################################
# print(new_dict)
# print(new_dict == "Piano_results_3_phases.pckl")
# print(type(new_dict))

###########################################################

# COMMUN ##################################################
T = np.hstack((np.linspace(0, 0.36574653, num=15), np.linspace(0.36574653, 0.36574653+0.10555556, num=15), np.linspace(0.36574653+0.10555556, 0.36574653+0.10555556+0.40625, num=16)))

# Q ######################################################
figQ, axs = plt.subplots(3, 4)

q = np.hstack(new_dict["states"][i]["q"][:, :-1] if i < 2 else new_dict["states"][i]["q"] for i in range(3))
q2 = np.hstack(new_dict2["states"][i]["q"][:, :-1] if i < 2 else new_dict2["states"][i]["q"] for i in range(3))
q3 = np.hstack(new_dict3["states"][i]["q"][:, :-1] if i < 2 else new_dict3["states"][i]["q"] for i in range(3))

axs[0, 0].plot(T, q[0, :], 'tab:red')
axs[0, 0].set_title("pelvis_rotZ_anteversion(-)/retroversion(+) \n")
# il n y a plus de DoF pour le pelvis

axs[0, 1].plot(T, q[1, :], 'tab:red')
axs[0, 1].set_title("thorax_rotX_inclination_right(-)/left(+) \n")
axs[0, 1].plot(T, q2[0, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[0, 2].plot(T, q[2, :], 'tab:red')
axs[0, 2].set_title("thorax_rotY_rotation_right(-)/left(+) \n")
axs[0, 2].plot(T, q2[1, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[0, 3].plot(T, q[3, :], 'tab:red')
axs[0, 3].set_title("thorax_rotZ_extension(-)/flexion(+) \n")
axs[0, 3].plot(T, q2[2, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[1, 0].plot(T, q[4, :], 'tab:red')
axs[1, 0].set_title("humerus_right_rotX_abduction(-)/adduction(+) \n")
axs[1, 0].plot(T, q2[3, :], 'tab:blue')
axs[1, 0].plot(T, q3[0, :], 'tab:orange')

axs[1, 1].plot(T, q[5, :], 'tab:red')
axs[1, 1].set_title("humerus_right_rotY_rotation_extern(-)/intern(+) \n")
axs[1, 1].plot(T, q2[4, :], 'tab:blue')
axs[1, 1].plot(T, q3[1, :], 'tab:orange')

axs[1, 2].plot(T, q[6, :], 'tab:red')
axs[1, 2].set_title("humerus_right_rotZ_extension(-)/flexion(+) \n")
axs[1, 2].plot(T, q2[5, :], 'tab:blue')
axs[1, 2].plot(T, q3[2, :], 'tab:orange')

axs[1, 3].plot(T, q[7, :], 'tab:red')
axs[1, 3].set_title("ulna_effector_right_rotZ_extension(-)/flexion(+) \n")
axs[1, 3].plot(T, q2[6, :], 'tab:blue')
axs[1, 3].plot(T, q3[3, :], 'tab:orange')

axs[2, 0].plot(T, q[8, :], 'tab:red')
axs[2, 0].set_title("radius_effector_right_rotY_rotation_extern(-)/intern(+) \n")
axs[2, 0].plot(T, q2[7, :], 'tab:blue')
axs[2, 0].plot(T, q3[4, :], 'tab:orange')

axs[2, 1].plot(T, q[9, :], 'tab:red')
axs[2, 1].set_title("hand_right_rotX_extension(-)/flexion(+) \n")
axs[2, 1].plot(T, q2[8, :], 'tab:blue')
axs[2, 1].plot(T, q3[5, :], 'tab:orange')

axs[2, 2].plot(T, q[10, :], 'tab:red')
axs[2, 2].set_title("hand_right_rotZ_deviation_radial(-)/ulnar(+) \n")
axs[2, 2].plot(T, q2[9, :], 'tab:blue')
axs[2, 2].plot(T, q3[6, :], 'tab:orange')

axs[2, 3].set_title("NOBODY")

# phase lines and axe titles
for i in range(0, 3):
    for j in range(0, 4):
        axs[i, j].axvline(x=0.36574653, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.36574653+0.10555556, color='gray', linestyle='--')
for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='q')
figQ.suptitle('State q of each DoF depending of the time for multiple contexts', fontname="Times New Roman", fontweight="bold")

# legends
line_labels = ["With every DoF", "Without pelvis_rotZ DoF", "Without pelvis_rotZ and thorax DoF"]

figQ.legend("r", "b", "g",               # The line objects
           labels=line_labels,       # The labels for each line
           loc="upper right",        # Position of legend
           borderaxespad=0.8,        # Small spacing around legend box
           title="Legend Titles")     # Title for the legend

figQ.tight_layout()

# Qdot ######################################################
figQdot, axs = plt.subplots(3, 4)

qdot = np.hstack(new_dict["states"][i]["qdot"][:, :-1] if i < 2 else new_dict["states"][i]["qdot"] for i in range(3))
qdot2 = np.hstack(new_dict2["states"][i]["qdot"][:, :-1] if i < 2 else new_dict2["states"][i]["qdot"] for i in range(3))
qdot3 = np.hstack(new_dict3["states"][i]["qdot"][:, :-1] if i < 2 else new_dict3["states"][i]["qdot"] for i in range(3))

axs[0, 0].plot(T, q[0, :], 'tab:red')
axs[0, 0].set_title("pelvis_rotZ_anteversion(-)/retroversion(+) \n")
# il n y a plus de DoF pour le pelvis

axs[0, 1].plot(T, qdot[1, :], 'tab:red')
axs[0, 1].set_title("thorax_rotX_inclination_right(-)/left(+) \n")
axs[0, 1].plot(T, qdot2[0, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[0, 2].plot(T, qdot[2, :], 'tab:red')
axs[0, 2].set_title("thorax_rotY_rotation_right(-)/left(+) \n")
axs[0, 2].plot(T, qdot2[1, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[0, 3].plot(T, qdot[3, :], 'tab:red')
axs[0, 3].set_title("thorax_rotZ_extension(-)/flexion(+) \n")
axs[0, 3].plot(T, qdot2[2, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[1, 0].plot(T, qdot[4, :], 'tab:red')
axs[1, 0].set_title("humerus_right_rotX_abduction(-)/adduction(+) \n")
axs[1, 0].plot(T, qdot2[3, :], 'tab:blue')
axs[1, 0].plot(T, qdot3[0, :], 'tab:orange')

axs[1, 1].plot(T, qdot[5, :], 'tab:red')
axs[1, 1].set_title("humerus_right_rotY_rotation_extern(-)/intern(+) \n")
axs[1, 1].plot(T, qdot2[4, :], 'tab:blue')
axs[1, 1].plot(T, qdot3[1, :], 'tab:orange')

axs[1, 2].plot(T, qdot[6, :], 'tab:red')
axs[1, 2].set_title("humerus_right_rotZ_extension(-)/flexion(+) \n")
axs[1, 2].plot(T, qdot2[5, :], 'tab:blue')
axs[1, 2].plot(T, qdot3[2, :], 'tab:orange')

axs[1, 3].plot(T, qdot[7, :], 'tab:red')
axs[1, 3].set_title("ulna_effector_right_rotZ_extension(-)/flexion(+) \n")
axs[1, 3].plot(T, qdot2[6, :], 'tab:blue')
axs[1, 3].plot(T, qdot3[3, :], 'tab:orange')

axs[2, 0].plot(T, qdot[8, :], 'tab:red')
axs[2, 0].set_title("radius_effector_right_rotY_rotation_extern(-)/intern(+) \n")
axs[2, 0].plot(T, qdot2[7, :], 'tab:blue')
axs[2, 0].plot(T, qdot3[4, :], 'tab:orange')

axs[2, 1].plot(T, qdot[9, :], 'tab:red')
axs[2, 1].set_title("hand_right_rotX_extension(-)/flexion(+) \n")
axs[2, 1].plot(T, qdot2[8, :], 'tab:blue')
axs[2, 1].plot(T, qdot3[5, :], 'tab:orange')

axs[2, 2].plot(T, qdot[10, :], 'tab:red')
axs[2, 2].set_title("hand_right_rotZ_deviation_radial(-)/ulnar(+) \n")
axs[2, 2].plot(T, qdot2[9, :], 'tab:blue')
axs[2, 2].plot(T, qdot3[6, :], 'tab:orange')

axs[2, 3].set_title("NOBODY")

# phase lines and axe titles
for i in range(0, 3):
    for j in range(0, 4):
        axs[i, j].axvline(x=0.36574653, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.36574653+0.10555556, color='gray', linestyle='--')
for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Qdot')
figQdot.suptitle('Velocity Qdot of each DoF depending of the time for multiple contexts', fontname="Times New Roman", fontweight="bold")

# legends
line_labels = ["With every DoF", "Without pelvis_rotZ DoF", "Without pelvis_rotZ and thorax DoF"]
Qdot1 = axs[0, 0].plot(T, qdot[0, :], 'tab:red')
Qdot2 = axs[0, 1].plot(T, qdot2[0, :], 'tab:blue')
Qdot3 = axs[2, 2].plot(T, qdot3[6, :], 'tab:orange')

figQdot.legend([Qdot1, Qdot2, Qdot3],                 # The line objects
           labels=line_labels,       # The labels for each line
           loc="upper right",        # Position of legend
           borderaxespad=0.8,        # Small spacing around legend box
           title="Legend Titles")     # Title for the legend

figQdot.tight_layout()

# TAU ######################################################
figU, axs = plt.subplots(3, 4)

U = np.hstack((new_dict["controls"][i]["tau"][:,:-1] if i < 2 else new_dict["controls"][i]["tau"] for i in range(3)))
U2 = np.hstack((new_dict2["controls"][i]["tau"][:,:-1] if i < 2 else new_dict2["controls"][i]["tau"] for i in range(3)))
U3 = np.hstack((new_dict3["controls"][i]["tau"][:,:-1] if i < 2 else new_dict3["controls"][i]["tau"] for i in range(3)))


axs[0, 0].plot(T, U[0, :], 'tab:red')
axs[0, 0].set_title("Tau_pelvis_rotZ_anteversion(-)/retroversion(+) \n")

# il n y a plus de DoF pour le pelvis

axs[0, 1].plot(T, U[1, :], 'tab:red')
axs[0, 1].set_title("Tau_thorax_rotX_inclination_right(-)/left(+) \n")
axs[0, 1].plot(T, U2[0, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[0, 2].plot(T, U[2, :], 'tab:red')
axs[0, 2].set_title("Tau_thorax_rotY_rotation_right(-)/left(+) \n")
axs[0, 2].plot(T, U2[1, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[0, 3].plot(T, U[3, :], 'tab:red')
axs[0, 3].set_title("Tau_thorax_rotZ_extension(-)/flexion(+) \n")
axs[0, 3].plot(T, U2[2, :], 'tab:blue')
# il n y a plus de DoF pour le thorax

axs[1, 0].plot(T, U[4, :], 'tab:red')
axs[1, 0].set_title("Tau_humerus_right_rotX_abduction(-)/adduction(+) \n")
axs[1, 0].plot(T, U2[3, :], 'tab:blue')
axs[1, 0].plot(T, U3[0, :], 'tab:orange')

axs[1, 1].plot(T, U[5, :], 'tab:red')
axs[1, 1].set_title("Tau_humerus_right_rotY_rotation_extern(-)/intern(+) \n")
axs[1, 1].plot(T, U2[4, :], 'tab:blue')
axs[1, 1].plot(T, U3[1, :], 'tab:orange')

axs[1, 2].plot(T, U[6, :], 'tab:red')
axs[1, 2].set_title("Tau_humerus_right_rotZ_extension(-)/flexion(+) \n")
axs[1, 2].plot(T, U2[5, :], 'tab:blue')
axs[1, 2].plot(T, U3[2, :], 'tab:orange')

axs[1, 3].plot(T, U[7, :], 'tab:red')
axs[1, 3].set_title("Tau_ulna_effector_right_rotZ_extension(-)/flexion(+) \n")
axs[1, 3].plot(T, U2[6, :], 'tab:blue')
axs[1, 3].plot(T, U3[3, :], 'tab:orange')

axs[2, 0].plot(T, U[8, :], 'tab:red')
axs[2, 0].set_title("Tau_radius_effector_right_rotY_rotation_extern(-)/intern(+) \n")
axs[2, 0].plot(T, U2[7, :], 'tab:blue')
axs[2, 0].plot(T, U3[4, :], 'tab:orange')

axs[2, 1].set_title("Tau_hand_right_rotX_extension(-)/flexion(+) \n")
axs[2, 1].plot(T, U[9, :], 'tab:red')
axs[2, 1].plot(T, U2[8, :], 'tab:blue')
axs[2, 1].plot(T, U3[5, :], 'tab:orange')

axs[2, 2].set_title("Tau_hand_right_rotZ_deviation_radial(-)/ulnar(+) \n")
axs[2, 2].plot(T, U[10, :], 'tab:red')
axs[2, 2].plot(T, U2[9, :], 'tab:blue')
axs[2, 2].plot(T, U3[6, :], 'tab:orange')

axs[2, 3].set_title("NOBODY")

# phase lines and axe titles
for i in range(0, 3):
    for j in range(0, 4):
        axs[i, j].axvline(x=0.36574653, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.36574653+0.10555556, color='gray', linestyle='--')
for ax in axs.flat:
    ax.set(xlabel='Time', ylabel='Tau')
figU.suptitle('Torque of each DoF depending of the time for multiple contexts', fontname="Times New Roman", fontweight="bold")

# legends
line_labels = ["With every DoF", "Without pelvis_rotZ DoF", "Without pelvis_rotZ and thorax DoF"]
U1 = axs[0, 0].plot(T, U[0, :], 'tab:red')
U2 = axs[0, 1].plot(T, U2[0, :], 'tab:blue')
U3 = axs[2, 2].plot(T, U3[6, :], 'tab:orange')

figU.legend([U1, U2, U3],                 # The line objects
           labels=line_labels,       # The labels for each line
           loc="upper right",        # Position of legend
           borderaxespad=0.8,        # Small spacing around legend box
           title="Legend Titles")     # Title for the legend

figU.tight_layout()

# show the graph
plt.show()

# Save images
# figU.savefig('State q of each DoF depending of the time for multiple contexts.png')
# figQdot.savefig('Velocity Qdot of each DoF depending of the time for multiple contexts.png')
# figQ.savefig('Torque of each DoF depending of the time for multiple contexts.png')
