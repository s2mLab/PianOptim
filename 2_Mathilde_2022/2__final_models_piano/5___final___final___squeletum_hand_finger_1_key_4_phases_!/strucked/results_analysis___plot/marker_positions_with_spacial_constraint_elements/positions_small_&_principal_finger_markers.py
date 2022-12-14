import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle
from bioptim import (
    BiorbdInterface,
)

with open(
    "/home/lim/Documents/Stage Mathilde/PianOptim/Mathilde_2022/2__final_models_piano/5___final___final___squeletum_hand_finger_1_key_4_phases_!/strucked/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl",
    "rb",
) as file:
    new_dict = pickle.load(file)


biorbd_model_path: str = "/2__FINAL_MODELES_OSCAR/5___FINAL_Squeletum_hand_finger_1_key_4_phases/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"

# # --- Plot marker positions curves --- # #

all_q = np.hstack(
    (new_dict["states"][0]["q"], new_dict["states"][1]["q"], new_dict["states"][2]["q"], new_dict["states"][3]["q"])
)

t = np.linspace(0, sum(new_dict["phase_time"]), sum(new_dict["phase_shape"]))
t_tau = 0
figQ, axs = plt.subplots(3)

axs[0].set_title("Trans X")
axs[0].set(ylabel="Position (m)")
axs[0].plot(
    t,
    new_dict["q_finger_marker_5_idx_1"][:, 0],
    color="green",
    label="C1 : Finger 5 at the top right of the principal finger\n" "- in y : Finger 5 < Principal finger",
)
axs[0].plot(t, new_dict["q_finger_marker_idx_4"][:, 0], color="red")

axs[1].set_title("Trans Y")
axs[1].plot(t, new_dict["q_finger_marker_5_idx_1"][:, 1], color="green")
axs[1].plot(t, new_dict["q_finger_marker_idx_4"][:, 1], color="r")

axs[2].set_title("Trans Z")
axs[2].plot(t, new_dict["q_finger_marker_5_idx_1"][:, 2], color="green")
axs[2].plot(t, new_dict["q_finger_marker_idx_4"][:, 2], color="r")
axs[2].axhline(
    y=0.07808863830566405 - 0.01 - 0.01,
    color="b",
    linestyle="--",
    label="C2 : Finger 5 and principal finger above the bed key\n" "- bed Key",
)

figQ.suptitle(
    "States q and controls u for important markers\nSMALL ONE Finger_Marker_5 (green) & PRINCIPAL Finger_Marker (red)",
    fontsize=16,
    fontweight="bold",
)
figQ.legend(
    loc="upper right", borderaxespad=0, prop={"size": 8}, title="Spacial constraints and objectives for Markers :"
)
for j in range(3):
    axs[j].axvline(x=0.30, color="gray", linestyle="--")
    axs[j].axvline(x=0.30 + 0.027, color="gray", linestyle="--")
    axs[j].axvline(x=0.30 + 0.027 + 0.058, color="gray", linestyle="--")
    axs[j].axvline(x=0.30 + 0.027 + 0.058 + 0.3, color="gray", linestyle="--")

plt.show()
