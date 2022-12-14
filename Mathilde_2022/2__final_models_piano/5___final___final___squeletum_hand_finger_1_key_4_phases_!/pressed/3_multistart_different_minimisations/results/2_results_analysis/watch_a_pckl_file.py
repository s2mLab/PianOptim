
import pickle


with open(
    "/home/lim/Documents/Stage Mathilde/PianOptim/Mathilde_2022/2__final_models_piano/5___final___final___squeletum_hand_finger_1_key_4_phases_!/pressed/3_multistart_different_minimisations/results/2_results_analysis/pareto_front_curve_of_one_proximal_limb_torques_d._on_distal_limbs_torques/test.pckl",
    "rb",
) as file:
    new_dict = pickle.load(file)


print(new_dict["tab"])
