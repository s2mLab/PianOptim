
import bioviz

model_path = "/home/lim/Documents/Stage Mathilde/PianOptim/2: FINAL_MODELES/2: FINAL_Finger_2_keys_simulation/FINAL_Finger_2_keys_simulation.bioMod"

b = bioviz.Viz(model_path, markers_size=0.0020, contacts_size=0.0020, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=False,
                show_local_ref_frame=False,)
b.exec()







