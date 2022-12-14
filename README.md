# True_Pianoptim 

The aim of this project is to define, through numerical simulation and optimal control, new performance strategies that might help reduce exposition to risk of injuries.

# Grégoire 2021

The file Piano_final_version.py is a simulation of 5 chords played fortissimo with optimized movements and external forces (keys interaction).

The file Pianoptim_V1 is a simplified version of the same 5 chords but neglecting keys interaction, so without external forces.

# Mathilde 2022
FINAL MODEL DEVELOPED 

- The file "2__final_models_piano >> 5___final___final___squeletum_hand_finger_1_key_4_phases" gathered simulations of 1 key (principal LA of the piano)
played strucked staccato, and pressed staccato with optimized movements and the key interaction, for different values of torque minimisation.

- The file "2__final_models_piano >> 5___final___final___squeletum_hand_finger_1_key_4_phases >> pressed >>
4_multistart_different_minimisations >> multistart_pressed.py " is a multistart code to run multiple simulations
with different minimisation weights at the same time. Just adapted for the pressed attack for the moment. The results are saved in the first file, and analyses are done in the other one. 

OLD FINAL MODELS

- The file "2__final_models_piano" also gathered old final models which were steps to reach the 5_final_final_model.

OTHER DOCUMENTS

- The file "1__experimental_datas_and_calculations" gathered experimental datas taken from the lab server (access path 
detailed in "experimental_data_access_path" file. 
There are also calculations of useful dimensions.

- The file "3__two_models_of_squeletum_used" gathered two models of squeletum. In our model, the wrist and the hand come 
from the "stanford_model", and the other limbs come from the "wu_model". This file gathered also .slt files format for 
the used limbs, in order to open their mesh in a 3D software, in MeshLab for example.

- The file "4__lab_files" gathered non-important files.

- The file "a__show_bioMod.py" is to show a bioMod.

- The file "b__animate_results" is to animate the a .pckl results file, saved during a simulation.

- The file "c__CODE_EXPLAINATIONS_FILE" gathered technical explanations on codes developed, and some code advices.

