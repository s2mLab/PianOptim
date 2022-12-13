import pickle
import matplotlib.pyplot as plt
with open(
        "/a_Mathilde_2022/2__FINAL_MODELS_OSCAR/5___final___squeletum_hand_finger_1_key_4_phases/pressed/3_multistart_different_minimisations/results/2_results_analysis/pareto_front_curve_of_one_proximal_limb_torques_d._on_distal_limbs_torques/test.pckl",
          'rb') as file: tab_tau = pickle.load(file)
from scipy.integrate import quad

# # # time # # #
tmin = 0.0
tmax = 0.745

# # # x # # #

x = []
a = 0
for j in range(len(tab_tau)):
    a = 0
    for i in range(7, 11):
        a += abs(tab_tau.iloc[j][i])
    x += [a]

# # # x # # #

# x = []
# a = 0
# for j in range(len(tab_tau)):
#     a = 0
#     for i in range(7, 11):
#         a += abs(tab_tau.iloc[j][i])
#     x += [a]

# # # y # # #
# Humerus #
y_humerus = []
for j in range(len(tab_tau)):
    b = 0
    for i in range(4, 7):
        b += abs(tab_tau.iloc[j][i])
    y_humerus += [b]

# Thorax #
y_thorax = []
c = 0
for j in range(len(tab_tau)):
    c = 0
    for i in range(2, 4):
        c += abs(tab_tau.iloc[j][i])
    y_thorax += [c]

# Pelvis #
y_pelvis = []
for j in range(len(tab_tau)):
    d = 0
    for i in range(1, 2):
        d += abs(tab_tau.iloc[j][i])
    y_pelvis += [d]

# # # Plot # # #

figU, axs = plt.subplots(2, 2)
axs[0, 0].scatter(x, y_humerus)
axs[0, 0].set_title("Humerus", fontsize=14, fontweight="bold")
axs[0, 0].set_xlabel('Σ(| tau_distaux |) (en N.m)', fontsize=12)
axs[0, 0].set_ylabel('Σ(| tau_humerus |) (en N.m)', fontsize=12)
axs[0, 1].scatter(x, y_thorax)
axs[0, 1].set_title("Thorax", fontsize=14, fontweight="bold")
axs[0, 1].set_xlabel('Σ(| tau_distaux |) (en N.m)', fontsize=12)
axs[0, 1].set_ylabel('Σ(| tau_thorax |) (en N.m)', fontsize=12)
axs[1, 0].scatter(x, y_pelvis)
axs[1, 0].set_title("Pelvis", fontsize=14, fontweight="bold")
axs[1, 0].set_xlabel('Σ(| tau_distaux |) (en N.m)', fontsize=12)
axs[1, 0].set_ylabel('Σ(| tau_pelvis |) (en N.m)', fontsize=12)
figU.delaxes(axs[1][1])
figU.suptitle('Rapport entre la somme de la valeur absolue des couples de trois articulations proximales différentes'
              ' et la somme de la valeur absolue des couples des articulations distales\n'
              'pour plusieurs simulations minimisant plus ou moins les couples articulaires distaux.', fontweight="bold", size=12)
plt.subplots_adjust(top=0.88,
                    bottom=0.11,
                    left=0.125,
                    right=0.9,
                    hspace=0.295,
                    wspace=0.2)

# plt.figure()
# for y in [y_pelvis, y_humerus, y_thorax]:
#     plt.scatter(x, y, s = 100, color = ['blue', 'red', '0.70', 'r', '#ffee00'])

#  mettre courbe de pareto


plt.show()


