import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/strucked/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl', 'rb') as file:
    data_struck = pickle.load(file)

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/pressed/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl', 'rb') as file:
    data_Pressed = pickle.load(file)


T_s=(data_struck['phase_time'][0]+data_struck['phase_time'][1]+data_struck['phase_time'][2]+data_struck['phase_time'][3])  #Simulation Time
specific_points = [0, data_struck['phase_time'][0], data_struck['phase_time'][0]+data_struck['phase_time'][1], data_struck['phase_time'][0]+data_struck['phase_time'][1]+data_struck['phase_time'][2], data_struck['phase_time'][0]+data_struck['phase_time'][1]+data_struck['phase_time'][2]+data_struck['phase_time'][3]]

array_0_q_s = data_struck['states'][0]['q']  # First array
array_1_q_s = data_struck['states'][1]['q']  # Second array
array_2_q_s = data_struck['states'][2]['q']  # Third array
array_3_q_s = data_struck['states'][3]['q']  # Fourth array

x1,y1=(array_0_q_s.shape)
x2,y2=(array_1_q_s .shape)
x3,y3=(array_2_q_s.shape)
x4,y4=(array_3_q_s.shape)

y_q=y1+y2+y3+y4
print(y_q)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q_s= np.concatenate((array_0_q_s, array_1_q_s, array_2_q_s, array_3_q_s), axis=1)  #All Phases

#####################
array_0_qdot_s = data_struck['states'][0]['qdot']  # First array
array_1_qdot_s = data_struck['states'][1]['qdot']  # Second array
array_2_qdot_s = data_struck['states'][2]['qdot']  # Third array
array_3_qdot_s = data_struck['states'][3]['qdot']  # Fourth array

x1,y1=(array_0_qdot_s.shape)
x2,y2=(array_1_qdot_s.shape)
x3,y3=(array_2_qdot_s.shape)
x4,y4=(array_3_qdot_s.shape)

y_qdot=y1+y2+y3+y4
print(y_qdot)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_qdot_s= np.concatenate((array_0_qdot_s, array_1_qdot_s, array_2_qdot_s, array_3_qdot_s), axis=1)  #All Phases

#####################
array_0_tau_s = data_struck['controls'][0]['tau']  # First array
array_0_tau_s= array_0_tau_s[:, :-1]   #last node is NAN : Not a Number

array_1_tau_s= data_struck['controls'][1]['tau']  # Second array
array_1_tau_s= array_1_tau_s[:, :-1]

array_2_tau_s= data_struck['controls'][2]['tau']  # Third array
array_2_tau_s= array_2_tau_s[:, :-1]

array_3_tau_s= data_struck['controls'][3]['tau']  # Fourth array
array_3_tau_s= array_3_tau_s[:, :-1]

x1,y1=(array_0_tau_s.shape)
x2,y2=(array_1_tau_s.shape)
x3,y3=(array_2_tau_s.shape)
x4,y4=(array_3_tau_s.shape)

y_tau=y1+y2+y3+y4
print(y_tau)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_tau_s= np.concatenate((array_0_tau_s, array_1_tau_s, array_2_tau_s, array_3_tau_s), axis=1)

Name=["Pelvis_RotZ","Thorax_RotY","Thorax_RotZ","Humerus_Right_RotX","Humerus_Right_RotY","Humerus_Right_RotZ","Ulna_Right_RotZ","Radius_Right_RotY","Wrist_RotX","Finger_RotX"]
T_g_s=math.ceil(T_s + 0.2)

for i in range(0,10):

    fig, axs = plt.subplots(nrows=3, ncols=1)
    fig.suptitle(Name[i])

    x_q_s= np.linspace(0, T_s, y_q)
    x_qdot_s= np.linspace(0, T_s, y_qdot)
    # Plot data on each subplot

    axs[0].plot(x_q_s, concatenated_array_q_s[i, :], color='red')
    axs[0].set_title('q')
    for point in specific_points:
        axs[0].axvline(x=point, color='g', linestyle='--')

    plt.xticks([0.1 * tick for tick in range(0, math.ceil(T_g_s))])

    # y_min = math.ceil(min(concatenated_array_q[i]) - 0.1)
    # y_max = math.ceil(max(concatenated_array_q[i]) + 0.1)
    # plt.yticks([0.2 * tick for tick in range(y_min, y_max)])

    axs[1].plot(x_qdot_s, concatenated_array_qdot_s[i, :], color='red')
    axs[1].set_title('q_dot')
    plt.xticks([0.1 * tick for tick in range(0, T_g_s)])
    for point in specific_points:
        axs[1].axvline(x=point, color='g', linestyle='--')

    x_tau_s=np.linspace(0, T_s, y_tau)
    axs[2].plot(x_tau_s, concatenated_array_tau_s[i, :], color='red')
    axs[2].set_title('tau')
    plt.xticks([0.1 * tick for tick in range(0, math.ceil(T_g_s))])
    for point in specific_points:
        axs[2].axvline(x=point, color='g', linestyle='--')
    # y_min = math.ceil(min(concatenated_array_tau[i]) - 0.1)
    # y_max = math.ceil(max(concatenated_array_tau[i]) + 0.1)
    # plt.yticks([0.2 * tick for tick in range(y_min, y_max)])
    fig.text(0.5, 0.02, 'Time (sec)', ha='center')

    plt.tight_layout()

plt.show()



