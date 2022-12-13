import bioviz
import numpy as np
from matplotlib import pyplot as plt
import pickle
from bioptim import (
    BiorbdInterface,
    )

with open(
        "/Mathilde_2022/2__FINAL_MODELS_OSCAR/5___final___squeletum_hand_finger_1_key_4_phases/pressed/3_multistart_different_minimisations/results/2_results_analysis/pareto_front_curve_of_one_proximal_limb_torques_d._on_distal_limbs_torques/1_tab_pelvis_pareto_front.pckl",
          'rb') as file:new_dict = pickle.load(file)



print(new_dict["tab"])


