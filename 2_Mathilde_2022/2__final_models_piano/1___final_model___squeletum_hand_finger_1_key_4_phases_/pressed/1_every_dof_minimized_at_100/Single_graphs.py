import pickle
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator
import math

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/pressed/Results/V_1.pckl', 'rb') as file:
    data_1 = pickle.load(file)

specific_points_s = [data_1['phase_time'][0], data_1['phase_time'][0] + data_1['phase_time'][1], data_1['phase_time'][0] + data_1['phase_time'][1] + data_1['phase_time'][2], data_1['phase_time'][0] + data_1['phase_time'][1] + data_1['phase_time'][2] + data_1['phase_time'][3],data_1['phase_time'][0] + data_1['phase_time'][1] + data_1['phase_time'][2] + data_1['phase_time'][3]++ data_1['phase_time'][4]]

#####################
array_0_q_s = data_1['states'][0]['q']  # First array
array_1_q_s = data_1['states'][1]['q']  # Second array
array_2_q_s = data_1['states'][2]['q']  # Third array
array_3_q_s = data_1['states'][3]['q']  # Fourth array

x1_s,y1_s=(array_0_q_s.shape)
x2_s,y2_s=(array_1_q_s .shape)
x3_s,y3_s=(array_2_q_s.shape)
x4_s,y4_s=(array_3_q_s.shape)

y_q_s=y1_s+y2_s+y3_s+y4_s
print(y_q_s)

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q_s= np.concatenate((array_0_q_s, array_1_q_s, array_2_q_s, array_3_q_s), axis=1)  #All Phases
#####################
array_0_q_nis= data_1['states_no_intermediate'][0]['q']  # First array    nis: ..._no_intermediate   _s is randon I am using for both struck and pressed
array_1_q_nis= data_1['states_no_intermediate'][1]['q']  # Second array
array_2_q_nis= data_1['states_no_intermediate'][2]['q']  # Third array
array_3_q_nis= data_1['states_no_intermediate'][3]['q']  # Fourth array

x1_nis,y1_nis=(array_0_q_nis.shape)
x2_nis,y2_nis=(array_1_q_nis.shape)
x3_nis,y3_nis=(array_2_q_nis.shape)
x4_nis,y4_nis=(array_3_q_nis.shape)

y_q_nis=y1_nis+y2_nis+y3_nis+y4_nis
print(y_q_nis)

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q_nis= np.concatenate((array_0_q_nis, array_1_q_nis, array_2_q_nis, array_3_q_nis), axis=1)  #All Phases

#####################
array_0_qdot_s = data_1['states'][0]['qdot']  # First array
array_1_qdot_s = data_1['states'][1]['qdot']  # Second array
array_2_qdot_s = data_1['states'][2]['qdot']  # Third array
array_3_qdot_s = data_1['states'][3]['qdot']  # Fourth array

x1_s,y1_s=(array_0_qdot_s.shape)
x2_s,y2_s=(array_1_qdot_s .shape)
x3_s,y3_s=(array_2_qdot_s.shape)
x4_s,y4_s=(array_3_qdot_s.shape)

y_qdot_s=y1_s+y2_s+y3_s+y4_s
print(y_qdot_s)

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_qdot_s= np.concatenate((array_0_qdot_s, array_1_qdot_s, array_2_qdot_s, array_3_qdot_s), axis=1)  #All Phases
#####################
array_0_qdot_nis= data_1['states_no_intermediate'][0]['qdot']  # First array    nis: ..._no_intermediate   _s is randon I am using for both struck and pressed
array_1_qdot_nis= data_1['states_no_intermediate'][1]['qdot']  # Second array
array_2_qdot_nis= data_1['states_no_intermediate'][2]['qdot']  # Third array
array_3_qdot_nis= data_1['states_no_intermediate'][3]['qdot']  # Fourth array

x1_nis,y1_nis=(array_0_qdot_nis.shape)
x2_nis,y2_nis=(array_1_qdot_nis.shape)
x3_nis,y3_nis=(array_2_qdot_nis.shape)
x4_nis,y4_nis=(array_3_qdot_nis.shape)

y_qdot_nis=y1_nis+y2_nis+y3_nis+y4_nis
print(y_qdot_nis)

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_qdot_nis= np.concatenate((array_0_qdot_nis, array_1_qdot_nis, array_2_qdot_nis, array_3_qdot_nis), axis=1)  #All Phases

#####################
array_0_tau_s = data_1['controls'][0]['tau']  # First array
repeated_array_0_tau_s = np.repeat(array_0_tau_s[:,:-1], 5, axis=1)
last_column = repeated_array_0_tau_s[:, -1]
repeated_array_0_tau_s = np.hstack((repeated_array_0_tau_s, last_column.reshape(-1, 1)))

array_1_tau_s= data_1['controls'][1]['tau']  # Second array
repeated_array_1_tau_s = np.repeat(array_1_tau_s[:,:-1], 5, axis=1)
last_column = repeated_array_1_tau_s[:, -1]
repeated_array_1_tau_s = np.hstack((repeated_array_1_tau_s, last_column.reshape(-1, 1)))

array_2_tau_s= data_1['controls'][2]['tau']  # Third array
repeated_array_2_tau_s = np.repeat(array_2_tau_s[:,:-1], 5, axis=1)
last_column = repeated_array_2_tau_s[:, -1]
repeated_array_2_tau_s = np.hstack((repeated_array_2_tau_s, last_column.reshape(-1, 1)))

array_3_tau_s= data_1['controls'][3]['tau']  # Fourth array
repeated_array_3_tau_s = np.repeat(array_3_tau_s[:,:-1], 5, axis=1)
last_column = repeated_array_3_tau_s[:, -1]
repeated_array_3_tau_s = np.hstack((repeated_array_3_tau_s, last_column.reshape(-1, 1)))


x1_s,y1_s=(repeated_array_0_tau_s.shape)
x2_s,y2_s=(repeated_array_1_tau_s.shape)
x3_s,y3_s=(repeated_array_2_tau_s.shape)
x4_s,y4_s=(repeated_array_3_tau_s.shape)

y_tau_s=y1_s+y2_s+y3_s+y4_s
print(y_tau_s)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_tau_s= np.concatenate((repeated_array_0_tau_s, repeated_array_1_tau_s, repeated_array_2_tau_s, repeated_array_3_tau_s), axis=1)

#####################
array_0_time_s = data_1['Time'][0]  # First array
array_1_time_s = data_1['Time'][1] # Second array
array_2_time_s = data_1['Time'][2] # Third array
array_3_time_s = data_1['Time'][3] # Fourth array

y1_s=(array_0_time_s.shape)
y2_s=(array_1_time_s .shape)
y3_s=(array_2_time_s.shape)
y4_s=(array_3_time_s.shape)

y_time_s=y1_s+y2_s+y3_s+y4_s
print(y_time_s)

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_time_s= np.concatenate((array_0_time_s, array_1_time_s, array_2_time_s, array_3_time_s))  #All Phases
#####################

Name=["Pelvis_RotZ","Thorax_RotY","Thorax_RotZ","Humerus_Right_RotX","Humerus_Right_RotY","Humerus_Right_RotZ","Ulna_Right_RotZ","Radius_Right_RotY","Wrist_RotX","Finger_RotX"]

margin = 0.1
margin_b = 1

for i in range(0,10):

    fig, axs = plt.subplots(nrows=4, ncols=1)
    fig.suptitle(Name[i])

    axs[0].plot(concatenated_array_time_s,concatenated_array_q_s[i,:], color='red')
    axs[0].set_title('q')
    axs[0].set_xlim(np.min(concatenated_array_time_s), np.max(concatenated_array_time_s))
    axs[0].set_ylim(np.min(concatenated_array_q_s[i,:])-margin, np.max(concatenated_array_q_s[i,:])+margin)

    axs[1].plot(concatenated_array_time_s,concatenated_array_qdot_s[i,:],color='red')
    axs[1].set_title('q_dot')
    axs[1].set_xlim(np.min(concatenated_array_time_s), np.max(concatenated_array_time_s))
    axs[1].set_ylim(np.min(concatenated_array_qdot_s[i,:])-margin_b, np.max(concatenated_array_qdot_s[i,:])+margin_b)

    axs[2].plot(concatenated_array_time_s,concatenated_array_tau_s[i,:], color='red')
    axs[2].set_title('tau')
    axs[2].set_xlabel('Time (sec)')
    axs[2].set_xlim(np.min(concatenated_array_time_s), np.max(concatenated_array_time_s))
    axs[2].set_ylim(np.min(concatenated_array_tau_s[i, :])-margin_b, np.max(concatenated_array_tau_s[i, :])+margin_b)

    for ax in axs:
        ax.grid(True)
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        ax.grid(which='minor', linestyle=':', linewidth='0.2', color='black')


        for point in specific_points_s:
            ax.axvline(x=point, color='g', linestyle='--')

    plt.tight_layout()

    # plt.figure()
    # plt.plot(concatenated_array_q_s[i, :], concatenated_array_qdot_s[i, :])
    # plt.title(Name[i])
    # plt.xlabel('q')
    # plt.ylabel('qdot')
    #
    # plt.figure()
    # ax = plt.axes(projection='3d')
    # ax.plot3D(concatenated_array_q_s[i, :], concatenated_array_qdot_s[i, :], x_q_s[i], 'gray')


plt.show()



