import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/pressed/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl', 'rb') as file:
    data_1 = pickle.load(file)

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/pressed/1_every_dof_minimized_at_100/V1.pckl', 'rb') as file:
    data_2 = pickle.load(file)

#
# T_s=(data_1['phase_time'][0]+data_1['phase_time'][1]+data_1['phase_time'][2]+data_1['phase_time'][3])  #Simulation Time
# specific_points_s = [0, data_1['phase_time'][0], data_1['phase_time'][0] + data_1['phase_time'][1], data_1['phase_time'][0] + data_1['phase_time'][1] + data_1['phase_time'][2], data_1['phase_time'][0] + data_1['phase_time'][1] + data_1['phase_time'][2] + data_1['phase_time'][3]]
#
# T_p=(data_2['phase_time'][0]+data_2['phase_time'][1]+data_2['phase_time'][2]+data_2['phase_time'][3])  #Simulation Time
# specific_points_p = [0, data_2['phase_time'][0], data_2['phase_time'][0] + data_2['phase_time'][1], data_2['phase_time'][0] + data_2['phase_time'][1] + data_2['phase_time'][2], data_2['phase_time'][0] + data_2['phase_time'][1] + data_2['phase_time'][2] + data_2['phase_time'][3]]


array_0_q_s = data_1['states'][0]['q']  # First array
array_1_q_s = data_1['states'][1]['q']  # Second array
array_2_q_s = data_1['states'][2]['q']  # Third array
array_3_q_s = data_1['states'][3]['q']  # Fourth array

array_0_q_p = data_2['states'][0]['q']  # First array
array_1_q_p = data_2['states'][1]['q']  # Second array
array_2_q_p = data_2['states'][2]['q']  # Third array
array_3_q_p = data_2['states'][3]['q']  # Fourth array



x1_p,y1_p=(array_0_q_p.shape)
x2_p,y2_p=(array_1_q_p .shape)
x3_p,y3_p=(array_2_q_p.shape)
x4_p,y4_p=(array_3_q_p.shape)

y_q_p=y1_p+y2_p+y3_p+y4_p
print(y_q_p)



x1_s,y1_s=(array_0_q_s.shape)
x2_s,y2_s=(array_1_q_s .shape)
x3_s,y3_s=(array_2_q_s.shape)
x4_s,y4_s=(array_3_q_s.shape)

y_q_s=y1_s+y2_s+y3_s+y4_s
print(y_q_s)


# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q_s= np.concatenate((array_0_q_s, array_1_q_s, array_2_q_s, array_3_q_s), axis=1)  #All Phases

# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q_p= np.concatenate((array_0_q_p, array_1_q_p, array_2_q_p, array_3_q_p), axis=1)  #All Phases

#####################
array_0_qdot_s = data_1['states'][0]['qdot']  # First array
array_1_qdot_s = data_1['states'][1]['qdot']  # Second array
array_2_qdot_s = data_1['states'][2]['qdot']  # Third array
array_3_qdot_s = data_1['states'][3]['qdot']  # Fourth array

x1_s,y1_s=(array_0_qdot_s.shape)
x2_s,y2_s=(array_1_qdot_s.shape)
x3_s,y3_s=(array_2_qdot_s.shape)
x4_s,y4_s=(array_3_qdot_s.shape)

y_qdot_s=y1_s+y2_s+y3_s+y4_s
print(y_qdot_s)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_qdot_s= np.concatenate((array_0_qdot_s, array_1_qdot_s, array_2_qdot_s, array_3_qdot_s), axis=1)  #All Phases

#####################
array_0_qdot_p = data_2['states'][0]['qdot']  # First array
array_1_qdot_p = data_2['states'][1]['qdot']  # Second array
array_2_qdot_p = data_2['states'][2]['qdot']  # Third array
array_3_qdot_p = data_2['states'][3]['qdot']  # Fourth array

x1_p,y1_p=(array_0_qdot_p.shape)
x2_p,y2_p=(array_1_qdot_p.shape)
x3_p,y3_p=(array_2_qdot_p.shape)
x4_p,y4_p=(array_3_qdot_p.shape)

y_qdot_p=y1_p+y2_p+y3_p+y4_p
print(y_qdot_p)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_qdot_p= np.concatenate((array_0_qdot_p, array_1_qdot_p, array_2_qdot_p, array_3_qdot_p), axis=1)  #All Phases

#####################

array_0_tau_s = data_1['controls'][0]['tau']  # First array
array_0_tau_s= array_0_tau_s[:, :-1]   #last node is NAN : Not a Number

array_1_tau_s= data_1['controls'][1]['tau']  # Second array
array_1_tau_s= array_1_tau_s[:, :-1]

array_2_tau_s= data_1['controls'][2]['tau']  # Third array
array_2_tau_s= array_2_tau_s[:, :-1]

array_3_tau_s= data_1['controls'][3]['tau']  # Fourth array
array_3_tau_s= array_3_tau_s[:, :-1]

x1_s,y1_s=(array_0_tau_s.shape)
x2_s,y2_s=(array_1_tau_s.shape)
x3_s,y3_s=(array_2_tau_s.shape)
x4_s,y4_s=(array_3_tau_s.shape)

y_tau_s=y1_s+y2_s+y3_s+y4_s
print(y_tau_s)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_tau_s= np.concatenate((array_0_tau_s, array_1_tau_s, array_2_tau_s, array_3_tau_s), axis=1)

##################################

array_0_tau_p = data_2['controls'][0]['tau']  # First array
array_0_tau_p= array_0_tau_p[:, :-1]   #last node is NAN : Not a Number

array_1_tau_p= data_2['controls'][1]['tau']  # Second array
array_1_tau_p= array_1_tau_p[:, :-1]

array_2_tau_p= data_2['controls'][2]['tau']  # Third array
array_2_tau_p= array_2_tau_p[:, :-1]

array_3_tau_p= data_2['controls'][3]['tau']  # Fourth array
array_3_tau_p= array_3_tau_p[:, :-1]

x1_p,y1_p=(array_0_tau_p.shape)
x2_p,y2_p=(array_1_tau_p.shape)
x3_p,y3_p=(array_2_tau_p.shape)
x4_p,y4_p=(array_3_tau_p.shape)

y_tau_p=y1_p+y2_p+y3_p+y4_p
print(y_tau_p)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_tau_p= np.concatenate((array_0_tau_p, array_1_tau_p, array_2_tau_p, array_3_tau_p), axis=1)

##################

Name=["Pelvis_RotZ","Thorax_RotY","Thorax_RotZ","Humerus_Right_RotX","Humerus_Right_RotY","Humerus_Right_RotZ","Ulna_Right_RotZ","Radius_Right_RotY","Wrist_RotX","Finger_RotX"]
# T_g_s=math.ceil(T_s + 0.2)
#
# T_g_p=math.ceil(T_p + 0.2)

for i in range(0,10):

    fig, axs = plt.subplots(nrows=4, ncols=1)
    fig.suptitle(Name[i])
    #
    # x_q_s= np.linspace(0, T_s, y_q_s, dtype=float)
    # x_qdot_s= np.linspace(0, T_s, y_qdot_s, dtype=float)
    #
    # x_q_p= np.linspace(0, T_p, y_q_p, dtype=float)
    # x_qdot_p= np.linspace(0, T_p, y_qdot_p, dtype=float)
    # # Plot data on each subplot

    axs[0].plot(concatenated_array_q_s[i,:], color='red')
    axs[0].plot(concatenated_array_q_p[i,:], color='blue')

    axs[0].set_title('q')

    # y_min = math.ceil(min(concatenated_array_q[i]) - 0.1)
    # y_max = math.ceil(max(concatenated_array_q[i]) + 0.1)
    # plt.yticks([0.2 * tick for tick in range(y_min, y_max)])

    axs[1].plot(concatenated_array_qdot_s[i,:],color='red')
    axs[1].plot(concatenated_array_qdot_p[i,:],color='blue')
    axs[1].set_title('q_dot')


    # x_tau_s=np.linspace(0, T_s, y_tau_s, dtype=float)
    axs[2].plot(concatenated_array_tau_s[i,:], color='red')

    # x_tau_p=np.linspace(0, T_p, y_tau_p, dtype=float)
    axs[2].plot(concatenated_array_tau_p[i,:], color='blue')

    axs[2].set_title('tau')
    axs[2].set_xlabel('Time (sec)')

    # for ax in axs:
    #     plt.xticks([0.1 * tick for tick in range(0, T_g_s)])
    #     plt.grid(True)
    #
    #     for point in specific_points_s:
    #         ax.axvline(x=point, color='g', linestyle='--')
    #
    #     for point in specific_points_p:
    #         ax.axvline(x=point, color='r', linestyle=':')
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



