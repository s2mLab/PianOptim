
import bioviz

model_path = "/home/mickaelbegon/Documents/Stage_Mathilde/programation/PianOptim/0: On going/1D_Simulation/1_key_Simulation_hand_with_impact.bioMod"

b = bioviz.Viz(model_path, markers_size=0.0010, contacts_size=0.0010, show_floor=False,
                show_segments_center_of_mass=True, show_global_ref_frame=True,
                show_local_ref_frame=True,)
b.exec()







