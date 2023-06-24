import pickle
import matplotlib.pyplot as plt
import numpy as np
import math

with open('/home/alpha/pianoptim/PianOptim/2_Mathilde_2022/2__final_models_piano/1___final_model___squeletum_hand_finger_1_key_4_phases_/strucked/1_every_dof_minimized_at_100/1_every_dof_minimized_at_100.pckl', 'rb') as file:
    data = pickle.load(file)

print(data)

T_s=(data['phase_time'][0]+data['phase_time'][1]+data['phase_time'][2]+data['phase_time'][3])  #Simulation Time
specific_points = [0,data['phase_time'][0], data['phase_time'][0]+data['phase_time'][1], data['phase_time'][0]+data['phase_time'][1]+data['phase_time'][2], data['phase_time'][0]+data['phase_time'][1]+data['phase_time'][2]+data['phase_time'][3]]

array_0_q = data['states'][0]['q']  # First array
array_1_q = data['states'][1]['q']  # Second array
array_2_q = data['states'][2]['q']  # Third array
array_3_q = data['states'][3]['q']  # Fourth array

x1,y1=(array_0_q.shape)
x2,y2=(array_1_q .shape)
x3,y3=(array_2_q.shape)
x4,y4=(array_3_q.shape)

y_q=y1+y2+y3+y4
print(y_q)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_q= np.concatenate((array_0_q, array_1_q, array_2_q, array_3_q), axis=1)  #All Phases

#####################
array_0_qdot = data['states'][0]['qdot']  # First array
array_1_qdot = data['states'][1]['qdot']  # Second array
array_2_qdot = data['states'][2]['qdot']  # Third array
array_3_qdot = data['states'][3]['qdot']  # Fourth array

x1,y1=(array_0_qdot.shape)
x2,y2=(array_1_qdot.shape)
x3,y3=(array_2_qdot.shape)
x4,y4=(array_3_qdot.shape)

y_qdot=y1+y2+y3+y4
print(y_qdot)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_qdot= np.concatenate((array_0_qdot, array_1_qdot, array_2_qdot, array_3_qdot), axis=1)  #All Phases

#####################
array_0_tau = data['controls'][0]['tau']  # First array
array_0_tau=array_0_tau[:, :-1]   #last node is NAN : Not a Number

array_1_tau= data['controls'][1]['tau']  # Second array
array_1_tau=array_1_tau[:, :-1]

array_2_tau= data['controls'][2]['tau']  # Third array
array_2_tau=array_2_tau[:, :-1]

array_3_tau= data['controls'][3]['tau']  # Fourth array
array_3_tau=array_3_tau[:, :-1]

x1,y1=(array_0_tau.shape)
x2,y2=(array_1_tau.shape)
x3,y3=(array_2_tau.shape)
x4,y4=(array_3_tau.shape)

y_tau=y1+y2+y3+y4
print(y_tau)
# Concatenate the new arrays along axis 1 (horizontally)
concatenated_array_tau= np.concatenate((array_0_tau, array_1_tau, array_2_tau, array_3_tau), axis=1)

Name=["Pelvis_RotZ","Thorax_RotY","Thorax_RotZ","Humerus_Right_RotX","Humerus_Right_RotY","Humerus_Right_RotZ","Ulna_Right_RotZ","Radius_Right_RotY","Wrist_RotX","Finger_RotX"]
T_g=math.ceil(T_s+0.2)

for i in range(0,10):

    fig, axs = plt.subplots(nrows=3, ncols=1)
    fig.suptitle(Name[i])

    x_q= np.linspace(0, T_s, y_q)
    x_qdot= np.linspace(0, T_s, y_qdot)
    # Plot data on each subplot
    axs[0].plot(x_q, concatenated_array_q[i,:], color='red')
    axs[0].set_title('q')
    plt.xticks([0.1 * tick for tick in range(0, math.ceil(T_g))])
    for point in specific_points:
        plt.axvline(x=point, color='blue', linestyle='--')

    # y_min = math.ceil(min(concatenated_array_q[i]) - 0.1)
    # y_max = math.ceil(max(concatenated_array_q[i]) + 0.1)
    # plt.yticks([0.2 * tick for tick in range(y_min, y_max)])

    axs[1].plot(x_qdot, concatenated_array_qdot[i,:],color='red')
    axs[1].set_title('q_dot')
    plt.xticks([0.1 * tick for tick in range(0, T_g)])
    for point in specific_points:
        plt.axvline(x=point, color='blue', linestyle='--')


    x_tau=np.linspace(0, T_s, y_tau)
    axs[2].plot(x_tau,concatenated_array_tau[i,:], color='red')
    axs[2].set_title('tau')
    plt.xticks([0.1 * tick for tick in range(0, math.ceil(T_g))])
    # y_min = math.ceil(min(concatenated_array_tau[i]) - 0.1)
    # y_max = math.ceil(max(concatenated_array_tau[i]) + 0.1)
    # plt.yticks([0.2 * tick for tick in range(y_min, y_max)])
    fig.text(0.5, 0.04, 'Time (sec)', ha='center')
    for point in specific_points:
        plt.axvline(x=point, color='blue', linestyle='--')

    plt.tight_layout()

plt.show()



