import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle
from bioptim import (
    BiorbdInterface,
    )

with open(
        "/a_Mathilde_2022/1__EXPERIMENTAL_DATAS_+_CALCULATIONS/results_dof_with_keys/Piano_results.pckl",
          'rb') as file:new_dict = pickle.load(file)



biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/a_Mathilde_2022/2__FINAL_MODELS_OSCAR/5___final___squeletum_hand_finger_1_key_4_phases/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"

# # --- Animate --- # #

b = bioviz.Viz(biorbd_model_path, markers_size=0.0010, contacts_size=0.0010, show_floor=False,
               show_segments_center_of_mass=True, show_global_ref_frame=True, show_local_ref_frame=False, )

all_q = np.hstack((new_dict["states"][0]["q"], new_dict["states"][1]["q"],
                   new_dict["states"][2]["q"], new_dict["states"][3]["q"]))

b.load_movement(all_q)
b.exec()
plt.show()


