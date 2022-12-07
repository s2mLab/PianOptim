import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle
from bioptim import (
    BiorbdInterface,
    )

with open(
        "/2__FINAL_MODELES_OSCAR/5___FINAL_Squeletum_hand_finger_1_key_4_phases/pressed/1_every_dof_minimized_at_100/test_egality/1_every_dof_minimized_at_100.pckl",
          'rb') as file:new_dict = pickle.load(file)



biorbd_model_path: str = "/2__FINAL_MODELES_OSCAR/5___FINAL_Squeletum_hand_finger_1_key_4_phases/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"

# # --- Animate --- # #

b = bioviz.Viz(biorbd_model_path, markers_size=0.0010, contacts_size=0.0010, show_floor=False,
               show_segments_center_of_mass=True, show_global_ref_frame=True, show_local_ref_frame=False, )

all_q = np.hstack((new_dict["states"][0]["q"], new_dict["states"][1]["q"],
                   new_dict["states"][2]["q"], new_dict["states"][3]["q"]))

b.load_movement(all_q)
b.exec()
plt.show()


