
import bioviz

model_path = "/home/lim/Documents/Stage Mathilde/PianOptim/0:On_going/5:FINAL_Squeletum_hand_finger_2_keys/Frappe/4_phases/Squeletum_hand_finger_3D_2_keys_octave_LA_frappe.bioMod"

b = bioviz.Viz(model_path, markers_size=0.00150, contacts_size=0.00150, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=True,
                show_local_ref_frame=True,)
b.exec()







