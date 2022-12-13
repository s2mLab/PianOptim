
from matplotlib import pyplot as plt
import pickle
import numpy as np

# Import results with pelvis rotZ
with open(
        "/Mathilde_2022/2__FINAL_MODELS_OSCAR/5___final___squeletum_hand_finger_1_key_4_phases/pressed/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl", 'rb') as file:new_dict = pickle.load(file)
with open(
        "/Mathilde_2022/2__FINAL_MODELS_OSCAR/5___final___squeletum_hand_finger_1_key_4_phases/pressed/2_finger_hand_ulna_radius_minimize_at_10_000/2_finger_hand_ulna_radius_minimize_at_10_000.pckl", 'rb') as file: new_dict2 = pickle.load(file)

# Print the dic ###########################################
# print(new_dict)
# print(new_dict == "Piano_results.pckl")
# print(type(new_dict))
###########################################################

# COMMUN ##################################################
T = np.hstack((np.linspace(0, 0.3, num=99), np.linspace(0.3, 0.3+0.044, num=99), np.linspace(0.3+0.044, 0.3+0.044+0.051, num=99), np.linspace(0.3+0.044+0.051, 0.3+0.044+0.051+0.35, num=99)))

# Q ######################################################
figQ, axs = plt.subplots(4, 3)
figQ.delaxes(axs[3][1])
figQ.delaxes(axs[3][2])
plt.subplots_adjust(top=0.895,
                    bottom=0.045,
                    left=0.042,
                    right=0.986,
                    hspace=0.514,
                    wspace=0.15)

q = np.hstack(new_dict["states"][i]["q"][:, :-1] if i < 3 else new_dict["states"][i]["q"] for i in range(4))
q2 = np.hstack(new_dict2["states"][i]["q"][:, :-1] if i < 3 else new_dict2["states"][i]["q"] for i in range(4))

axs[0, 0].plot(T, q[0, :], 'tab:orange', label="Index, hand, radius and ulna at 100. Objectif * 1.")
axs[0, 0].set_title("pelvis_rotZ_anteversion(-)/retroversion(+)", fontsize=14)
axs[0, 0].plot(T, q2[0, :], 'tab:blue', label="Index, hand, radius and ulna at 10 000. Others at 100. Objectif * 50.", linestyle='--')

axs[0, 1].plot(T, q[1, :], 'tab:orange')
axs[0, 1].set_title("thorax_rotY_rotation_right(-)/left(+)", fontsize=14)
axs[0, 1].plot(T, q2[1, :], 'tab:blue', linestyle='--')

axs[0, 2].plot(T, q[2, :], 'tab:orange')
axs[0, 2].set_title("thorax_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[0, 2].plot(T, q2[2, :], 'tab:blue', linestyle='--')

axs[1, 0].plot(T, q[3, :], 'tab:orange')
axs[1, 0].set_title("humerus_rotX_abduction(-)/adduction(+)", fontsize=14)
axs[1, 0].plot(T, q2[3, :], 'tab:blue', linestyle='--')

axs[1, 1].plot(T, q[4, :], 'tab:orange')
axs[1, 1].set_title("humerus_rotY_rotation_extern(-)/intern(+)", fontsize=14)
axs[1, 1].plot(T, q2[4, :], 'tab:blue', linestyle='--')

axs[1, 2].plot(T, q[5, :], 'tab:orange')
axs[1, 2].set_title("humerus_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[1, 2].plot(T, q2[5, :], 'tab:blue', linestyle='--')

axs[2, 0].plot(T, q[6, :], 'tab:orange')
axs[2, 0].set_title("ulna_effector_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[2, 0].plot(T, q2[6, :], 'tab:blue', linestyle='--')

axs[2, 1].plot(T, q[7, :], 'tab:orange')
axs[2, 1].set_title("radius_effector_rotY_rotation_extern(-)/intern(+)", fontsize=14)
axs[2, 1].plot(T, q2[7, :], 'tab:blue', linestyle='--')

axs[2, 2].plot(T, q[8, :], 'tab:orange')
axs[2, 2].set_title("hand_rotX_extension(-)/flexion(+)", fontsize=14)
axs[2, 2].plot(T, q2[8, :], 'tab:blue', linestyle='--')

axs[3, 0].plot(T, q[9, :], 'tab:orange')
axs[3, 0].set_title("index_rotX_extension(-)/flexion(+)", fontsize=14)
axs[3, 0].plot(T, q2[9, :], 'tab:blue', linestyle='--')

# phase lines and axe titles
for i in range(0, 4):
    for j in range(0, 3):
        axs[i, j].axvline(x=0.3, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3+0.044, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3+0.044+0.051, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3+0.044+0.051+0.35, color='gray', linestyle='--')
        axs[i, j].grid()

for ax in axs.flat:
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('q (rad)', fontsize=12)
figQ.suptitle('States (q) of limbs by minimizing more the finger, hand, radius & ulna for a staccato pressed attack of one key.', fontweight="bold", size=18)
figQ.legend(loc="lower right",        # Position of legend
           borderaxespad=0,        # Small spacing around legend box
           title="Weight of minimisation controls for :", prop={"size": 14}, title_fontsize=18)  # Title for the legend

# Qdot ######################################################
figQdot, axs = plt.subplots(4, 3)
figQdot.delaxes(axs[3][1])
figQdot.delaxes(axs[3][2])
plt.subplots_adjust(top=0.895,
                    bottom=0.045,
                    left=0.042,
                    right=0.986,
                    hspace=0.514,
                    wspace=0.15)

qdot = np.hstack(new_dict["states"][i]["qdot"][:, :-1] if i < 3 else new_dict["states"][i]["qdot"] for i in range(4))
qdot2 = np.hstack(new_dict2["states"][i]["qdot"][:, :-1] if i < 3 else new_dict2["states"][i]["qdot"] for i in range(4))


axs[0, 0].plot(T, qdot[0, :], 'tab:orange', label="Index, hand, radius and ulna at 100. Objectif * 1.")
axs[0, 0].set_title("pelvis_rotZ_anteversion(-)/retroversion(+)", fontsize=14)
axs[0, 0].plot(T, qdot2[0, :], 'tab:blue', label="Index, hand, radius and ulna at 10 000. Others at 100. Objectif * 50.", linestyle='--')

axs[0, 1].plot(T, qdot[1, :], 'tab:orange')
axs[0, 1].set_title("thorax_rotY_rotation_right(-)/left(+)", fontsize=14)
axs[0, 1].plot(T, qdot2[1, :], 'tab:blue', linestyle='--')

axs[0, 2].plot(T, qdot[2, :], 'tab:orange')
axs[0, 2].set_title("thorax_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[0, 2].plot(T, qdot2[2, :], 'tab:blue', linestyle='--')

axs[1, 0].plot(T, qdot[3, :], 'tab:orange')
axs[1, 0].set_title("humerus_rotX_abduction(-)/adduction(+)", fontsize=14)
axs[1, 0].plot(T, qdot2[3, :], 'tab:blue', linestyle='--')

axs[1, 1].plot(T, qdot[4, :], 'tab:orange')
axs[1, 1].set_title("humerus_rotY_rotation_extern(-)/intern(+)", fontsize=14)
axs[1, 1].plot(T, qdot2[4, :], 'tab:blue', linestyle='--')

axs[1, 2].plot(T, qdot[5, :], 'tab:orange')
axs[1, 2].set_title("humerus_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[1, 2].plot(T, qdot2[5, :], 'tab:blue', linestyle='--')

axs[2, 0].plot(T, qdot[6, :], 'tab:orange')
axs[2, 0].set_title("ulna_effector_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[2, 0].plot(T, qdot2[6, :], 'tab:blue', linestyle='--')

axs[2, 1].plot(T, qdot[7, :], 'tab:orange')
axs[2, 1].set_title("radius_effector_rotY_rotation_extern(-)/intern(+)", fontsize=14)
axs[2, 1].plot(T, qdot2[7, :], 'tab:blue', linestyle='--')

axs[2, 2].plot(T, qdot[8, :], 'tab:orange')
axs[2, 2].set_title("hand_rotX_extension(-)/flexion(+)", fontsize=14)
axs[2, 2].plot(T, qdot2[8, :], 'tab:blue', linestyle='--')

axs[3, 0].plot(T, qdot[9, :], 'tab:orange')
axs[3, 0].set_title("index_rotX_extension(-)/flexion(+)", fontsize=14)
axs[3, 0].plot(T, qdot2[9, :], 'tab:blue', linestyle='--')

# phase lines and axe titles
for i in range(0, 4):
    for j in range(0, 3):
        axs[i, j].axvline(x=0.3, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3 + 0.044, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3 + 0.044 + 0.051, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3 + 0.044 + 0.051 + 0.35, color='gray', linestyle='--')
        axs[i, j].grid()

for ax in axs.flat:
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Qdot (rad.s⁻1)', fontsize=12)

figQdot.suptitle('States (qdot) of limbs by minimizing more the finger, hand, radius & ulna for a staccato pressed attack of one key.', fontweight="bold", size=18)

# legends
figQdot.legend(loc="lower right", borderaxespad=0, title="Weight of minimisation controls for :", prop={"size": 14}, title_fontsize=18)

# TAU ######################################################
figU, axs = plt.subplots(4, 3)
figU.delaxes(axs[3][1])
figU.delaxes(axs[3][2])
plt.subplots_adjust(top=0.895,
                    bottom=0.045,
                    left=0.042,
                    right=0.986,
                    hspace=0.514,
                    wspace=0.15)

T2 = np.hstack((np.linspace(0, 0.3, num=20), np.linspace(0.3, 0.3+0.044, num=20), np.linspace(0.3+0.044, 0.3+0.044+0.051, num=20), np.linspace(0.3+0.044+0.051, 0.3+0.044+0.051+0.35, num=20)))

U = np.hstack((new_dict["controls"][i]["tau"][:, :-1] if i < 3 else new_dict["controls"][i]["tau"] for i in range(4)))
U2 = np.hstack((new_dict2["controls"][i]["tau"][:, :-1] if i < 3 else new_dict2["controls"][i]["tau"] for i in range(4)))


axs[0, 0].plot(T2, U[0, :], 'tab:orange', label="Index, hand, radius and ulna at 100. Objectif * 1.")
axs[0, 0].set_title("Tau_pelvis_rotZ_anteversion(-)/retroversion(+)", fontsize=14)
axs[0, 0].plot(T2, U2[0, :], 'tab:blue', label="Index, hand, radius and ulna at 10 000. Others at 100. Objectif * 50.", linestyle='--')

axs[0, 1].plot(T2, U[1, :], 'tab:orange')
axs[0, 1].set_title("Tau_thorax_rotY_rotation_right(-)/left(+)", fontsize=14)
axs[0, 1].plot(T2, U2[1, :], 'tab:blue', linestyle='--')

axs[0, 2].plot(T2, U[2, :], 'tab:orange')
axs[0, 2].set_title("Tau_thorax_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[0, 2].plot(T2, U2[2, :], 'tab:blue', linestyle='--')

axs[1, 0].plot(T2, U[3, :], 'tab:orange')
axs[1, 0].set_title("Tau_humerus_rotX_abduction(-)/adduction(+)", fontsize=14)
axs[1, 0].plot(T2, U2[3, :], 'tab:blue', linestyle='--')

axs[1, 1].plot(T2, U[4, :], 'tab:orange')
axs[1, 1].set_title("Tau_humerus_rotY_rotation_extern(-)/intern(+)", fontsize=14)
axs[1, 1].plot(T2, U2[4, :], 'tab:blue', linestyle='--')


axs[1, 2].plot(T2, U[5, :], 'tab:orange')
axs[1, 2].set_title("Tau_humerus_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[1, 2].plot(T2, U2[5, :], 'tab:blue', linestyle='--')

axs[2, 0].plot(T2, U[6, :], 'tab:orange')
axs[2, 0].set_title("Tau_ulna_effector_rotZ_extension(-)/flexion(+)", fontsize=14)
axs[2, 0].plot(T2, U2[6, :], 'tab:blue', linestyle='--')

axs[2, 1].plot(T2, U[7, :], 'tab:orange')
axs[2, 1].set_title("Tau_radius_effector_rotY_rotation_extern(-)/intern(+)", fontsize=14)
axs[2, 1].plot(T2, U2[7, :], 'tab:blue', linestyle='--')

axs[2, 2].set_title("Tau_hand_rotX_extension(-)/flexion(+)", fontsize=14)
axs[2, 2].plot(T2, U[8, :], 'tab:orange')
axs[2, 2].plot(T2, U2[8, :], 'tab:blue', linestyle='--')

axs[3, 0].set_title("index_rotX_extension(-)/flexion(+)", fontsize=14)
axs[3, 0].plot(T2, U[9, :], 'tab:orange')
axs[3, 0].plot(T2, U2[9, :], 'tab:blue', linestyle='--')


# phase lines, legends and axe titles
for i in range(0, 4):
    for j in range(0, 3):
        axs[i, j].axvline(x=0.3, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3+0.044, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3+0.044+0.051, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.3+0.044+0.051+0.35, color='gray', linestyle='--')
        axs[i, j].grid()

for ax in axs.flat:
    ax.set_xlabel('Time (s)', fontsize=12)
    ax.set_ylabel('Tau (N.m)', fontsize=12)

figU.suptitle('Torque (tau) of limbs by minimizing more the finger, hand, radius & ulna for a staccato pressed attack of one key.', fontweight="bold", size=18)
figU.legend(loc="lower right", borderaxespad=0, title="Weight of minimisation controls for :", prop={"size": 14}, title_fontsize=18)

# show the graph
plt.show()

# Save images
# figU.savefig('State q of each DoF depending of the time for multiple contexts.png')
# figQdot.savefig('Velocity Qdot of each DoF depending of the time for multiple contexts.png')
# figQ.savefig('Torque of each DoF depending of the time for multiple contexts.png')
