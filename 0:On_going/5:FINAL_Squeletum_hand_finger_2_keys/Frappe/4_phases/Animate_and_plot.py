import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle
from bioptim import (
    BiorbdInterface,
    )
# w hand 10
# with open(
#         "/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/results_download/4_pi_on_two_finger_ph0&3_10_ph1&2_100000_hand_10_100000_with_5_cm/4_pi_on_two_finger_ph0&3_10_ph1&2_100000_hand_10_100000_with_5_cm.pckl",
#           'rb') as file:new_dict = pickle.load(file)

# w hand 100
with open(
        "/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/results_download/5_pi_on_two_finger_0&3_100_1&2_100000_hand_100_100000_with_5_cm/5_pi_on_two_finger_0&3_100_1&2_100000_hand_100_100000_with_5_cm.pckl",
          'rb') as file:new_dict = pickle.load(file)

biorbd_model_path: str = "/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/Squeletum_hand_finger_3D_2_keys_octave_LA_frappe.bioMod"

# # --- Animate --- # #

b = bioviz.Viz(biorbd_model_path, markers_size=0.0010, contacts_size=0.0010, show_floor=False,
               show_segments_center_of_mass=True, show_global_ref_frame=True, show_local_ref_frame=False, )

all_q = np.hstack((new_dict["states"][0]["q"], new_dict["states"][1]["q"],
                   new_dict["states"][2]["q"], new_dict["states"][3]["q"]))

# # --- Plot marker positions curves --- # #

t = np.linspace(0, sum(new_dict["phase_time"]), sum(new_dict["phase_shape"]))
t_tau = np.linspace(0, sum(new_dict["phase_time_tau"]), sum(new_dict["phase_shape_tau"]))
figQ, axs = plt.subplots(2, 3)

axs[0, 0].set_title("Trans X\n")
axs[0, 0].set(ylabel="Position (m)")
axs[0, 0].plot(t, new_dict["q_finger_marker_5_idx_1"][:, 0], color='green',
               label="C1 : Finger 5 at the top right of the principal finger\n"
                     "- in y : Finger 5 < Principal finger")
axs[0, 0].plot(t, new_dict["q_finger_marker_idx_4"][:, 0], color='red')

axs[0, 1].set_title("Trans Y\n")
axs[0, 1].plot(t, new_dict["q_finger_marker_5_idx_1"][:, 1], color='green')
axs[0, 1].plot(t, new_dict["q_finger_marker_idx_4"][:, 1], color='r')  # "C1 : Finger 5 at the right of the principal finger"

axs[0, 2].set_title("Trans Z\n")
axs[0, 2].plot(t, new_dict["q_finger_marker_5_idx_1"][:, 2], color='green')
axs[0, 2].plot(t, new_dict["q_finger_marker_idx_4"][:, 2], color='r')  # "C1 : Finger 5 at the right of the principal finger"
axs[0, 2].axhline(y=0.07808863830566405 - 0.01, color='b', linestyle='--',
                  label="C2 : Finger 5 and principal finger above the bed key\n"
                        "- bed Key")

axs[1, 0].set_title("Rot X\n")
axs[1, 0].set(ylabel='Position (m)')
axs[1, 0].plot(t_tau, new_dict["tau_finger_marker_idx_4"][:, 0], color='red')
axs[1, 0].axhline(y=0, color='m', linestyle='--',
                  label="Obj2 : Principal Finger just in rotation around y \n"
                        "- no rotation in x"
                        "    - no rotation in z")
axs[1, 1].set_title("Rot Y\n")
axs[1, 1].set(xlabel='Time (s) \n')
axs[1, 1].plot(t_tau, new_dict["tau_finger_marker_idx_4"][:, 1], color='red')

axs[1, 2].set_title("Rot Z\n")
axs[1, 2].plot(t_tau, new_dict["tau_finger_marker_idx_4"][:, 2], color='red')
axs[1, 2].axhline(y=0, color='m', linestyle='--')

figQ.suptitle(
    'States q and controls u for important markers\nSMALL ONE Finger_Marker_5 (green) & PRINCIPAL Finger_Marker (red)',
    fontsize=16, fontweight='bold')
figQ.legend(loc="upper right", borderaxespad=0, prop={"size": 8},
            title="Spacial constraints and objectives for Markers :")
for i in range(0, 2):
    for j in range(0, 3):
        axs[i, j].axvline(x=0.30, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.30 + 0.027, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.30 + 0.027 + 0.058, color='gray', linestyle='--')
        axs[i, j].axvline(x=0.30 + 0.027 + 0.058 + 0.3, color='gray', linestyle='--')

b.load_movement(all_q)
b.exec()
plt.show()


