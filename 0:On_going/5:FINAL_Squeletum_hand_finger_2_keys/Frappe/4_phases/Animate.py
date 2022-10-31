import pickle
import bioviz
import numpy as np
import numpy

with open(
        "/home/lim/Documents/Stage Mathilde/PianOptim/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/results_download/Piano_results_4_phases_7_1obj_rot_y.pckl",
          'rb') as file:new_dict = pickle.load(file)
biorbd_model_path: str = "/home/lim/Documents/Stage Mathilde/PianOptim/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/Squeletum_hand_finger_3D_2_keys_octave_LA_frappe.bioMod"


b = bioviz.Viz(biorbd_model_path, markers_size=0.0010, contacts_size=0.0010, show_floor=False,
               show_segments_center_of_mass=True, show_global_ref_frame=True, show_local_ref_frame=False, )

all_q = np.hstack((new_dict["states"][0]["q"], new_dict["states"][1]["q"],
                   new_dict["states"][2]["q"], new_dict["states"][3]["q"]))
b.load_movement(all_q)
b.exec()

