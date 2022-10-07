
import bioviz

model_path = "/home/lim/Documents/Stage Mathilde/PianOptim/0:On going/Piano_with_hand_and_key/Piano_with_hand_and_keys.bioMod"

b = bioviz.Viz(model_path, markers_size=0.0020, contacts_size=0.0020, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=False,
                show_local_ref_frame=False,)
b.exec()







