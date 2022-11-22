
import bioviz

model_path = "/home/lim/Documents/Stage Mathilde/PianOptim/5:Files/2_models_of_squeletum/Stanford Model/Stanford_VA_upper_limb_model_0_40.bioMod"

b = bioviz.Viz(model_path, markers_size=0.00150, contacts_size=0.00150, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=True,
                show_local_ref_frame=False,)
b.exec()







