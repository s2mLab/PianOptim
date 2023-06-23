# """
# This file is to display the model into bioviz
# """
# import os
# import bioviz
#
# model_path = "/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/bioMod/Squeletum_hand_finger_3D_2_keys_octave_LA.bioMod"
#
# export_model = False
# background_color = (1, 1, 1) if export_model else (0.5, 0.5, 0.5)
# show_gravity_vector = False if export_model else True
# show_floor = False if export_model else True
# show_local_ref_frame = False if export_model else True
# show_global_ref_frame = False if export_model else True
# show_markers = False if export_model else True
# show_mass_center = False if export_model else True
# show_global_center_of_mass = False if export_model else True
# show_segments_center_of_mass = False if export_model else True
# def print_all_camera_parameters(biorbd_viz: bioviz.Viz):
#     print("Camera roll: ", biorbd_viz.get_camera_roll())
#     print("Camera zoom: ", biorbd_viz.get_camera_zoom())
#     print("Camera position: ", biorbd_viz.get_camera_position())
#     print("Camera focus point: ", biorbd_viz.get_camera_focus_point())
#
#
# biorbd_viz = bioviz.Viz(
# model_path=model_path,
# # show_gravity_vector=False,
# show_floor=True,
# show_local_ref_frame=False,
# show_global_ref_frame=False,
# show_markers=False,
# show_mass_center=False,
# show_global_center_of_mass=False,
# show_segments_center_of_mass=False,
# mesh_opacity=1,
# background_color=(1, 1, 1),
# )
# biorbd_viz.set_camera_position(-8.782458942185185, 0.486269131372712, 4.362010279585766)
# biorbd_viz.set_camera_roll(90)
# biorbd_viz.set_camera_zoom(0.308185240948253)
# biorbd_viz.set_camera_focus_point(1.624007185850899, 0.009961251074366406, 1.940316420941989)
# biorbd_viz.resize(900, 900)
#
#
#
# biorbd_viz.exec()
# print_all_camera_parameters(biorbd_viz)
#
# print("Done")

import pickle
import matplotlib.pyplot as plt
from numpy import concatenate as Cn
import numpy as np
from scipy.interpolate import interp1d

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/strucked/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl', 'rb') as file:
    data = pickle.load(file)

# print(data)
Time=(data['phase_time'][0]+data['phase_time'][1]+data['phase_time'][2]+data['phase_time'][3])

array1 = data['states'][0]['q']  # First array
array2 = data['states'][1]['q']  # Second array
array3 = data['states'][2]['q']  # Third array
array4 = data['states'][3]['q']  # Fourth array

# Create a new array with the same number of rows as the original arrays,
# but with additional columns to match the shape of the largest array

x1,y1=(array1.shape)
x2,y2=(array2.shape)
x3,y3=(array3.shape)
x4,y4=(array4.shape)


# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q_0 = np.concatenate((array1, array2,array3, array4), axis=1)

#####################
array1_dot = data['states'][0]['qdot']  # First array
array2_dot = data['states'][1]['qdot']  # Second array
array3_dot = data['states'][2]['qdot']  # Third array
array4_dot = data['states'][3]['qdot']  # Fourth array

# Create a new array with the same number of rows as the original arrays,
# but with additional columns to match the shape of the largest array

x1,y1=(array1_dot.shape)
x2,y2=(array2_dot.shape)
x3,y3=(array3_dot.shape)
x4,y4=(array4_dot.shape)
concatenated_array_qdot_0 = np.concatenate((array1_dot, array2_dot,array3_dot, array4_dot), axis=1)

y_q=y1+y2+y3+y4
#####################
array1_dot = data['controls'][0]['tau']  # First array
array1_dot=array1_dot[:, :-1]
array2_dot = data['controls'][1]['tau']  # Second array
array2_dot=array2_dot[:, :-1]
array3_dot = data['controls'][2]['tau']  # Third array
array3_dot=array3_dot[:, :-1]
array4_dot = data['controls'][3]['tau']  # Fourth array
array4_dot=array4_dot[:, :-1]

# Create a new array with the same number of rows as the original arrays,
# but with additional columns to match the shape of the largest array

x1,y1=(array1_dot.shape)
x2,y2=(array2_dot.shape)
x3,y3=(array3_dot.shape)
x4,y4=(array4_dot.shape)


y_t=y1+y2+y3+y4

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_tau_0 = np.concatenate((array1_dot, array2_dot,array3_dot, array4_dot), axis=1)

fig, axs = plt.subplots(nrows=3, ncols=1)
x=np.linspace(0,Time,y_q)


# Plot data on each subplot
axs[0].plot(x,concatenated_array_q_0[0,:],color='red')
axs[0].set_title('q')

axs[1].plot(x,concatenated_array_qdot_0[0,:],color='red', label="data1")
axs[1].plot(x,concatenated_array_q_0[0,:],color='blue', label="data2")

axs[1].set_title('q_dot')

x=np.linspace(0,Time,y_t)

axs[2].plot(x,concatenated_array_tau_0[0,:], color='red')
axs[2].set_title('tau')
tick_positions = [-1.3, -1, -0.7, -0.5, -0.2]
tick_positions_2 = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
axs[2].set_xticks(tick_positions_2)

plt.tight_layout()

# Display the plot
plt.show()



