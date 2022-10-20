
import bioviz

model_path = "/0:On going/5:FINAL_Piano_with_hand_and_key_frappe_/Piano_with_hand_and_keys.bioMod"

b = bioviz.Viz(model_path, markers_size=0.0020, contacts_size=0.0020, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=True,
                show_local_ref_frame=False,)
b.exec()







