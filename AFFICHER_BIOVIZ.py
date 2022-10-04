
import bioviz

model_path = "/home/mickaelbegon/Documents/Stage_Mathilde/programation/PianOptim/0: On going/2D_Simulation/2_keys_Simulation_hand_with_impact.bioMod"

b = bioviz.Viz(model_path, markers_size=0.0010, contacts_size=0.0010, show_floor=False,
                show_segments_center_of_mass=False, show_global_ref_frame=True,
                show_local_ref_frame=False,)
b.exec()







