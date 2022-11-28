
import bioviz

model_path = "/home/lim/Documents/Stage Mathilde/PianOptim/0__On_going/limbs_size/vpt_files/Squeletum_colored.bioMod"

b = bioviz.Viz(model_path, markers_size=0.00150, contacts_size=0.00150, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=True,
                show_local_ref_frame=False,)
b.exec()







