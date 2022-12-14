import bioviz

model_path = "/2__FINAL_MODELES_OSCAR/5___FINAL_Squeletum_hand_finger_1_key_4_phases/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"

b = bioviz.Viz(
    model_path,
    markers_size=0.00150,
    contacts_size=0.00150,
    show_floor=False,
    show_segments_center_of_mass=False,
    show_global_ref_frame=True,
    show_local_ref_frame=False,
)
b.exec()
