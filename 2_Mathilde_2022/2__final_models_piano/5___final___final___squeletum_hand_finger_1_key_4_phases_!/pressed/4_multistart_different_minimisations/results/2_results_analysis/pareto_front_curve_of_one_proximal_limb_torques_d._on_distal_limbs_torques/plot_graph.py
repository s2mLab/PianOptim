import pickle
import matplotlib.pyplot as plt
from mpldatacursor import datacursor

with open(
        "/home/lim/Documents/Stage Mathilde/PianOptim/2_Mathilde_2022/2__final_models_piano/5___final___final___squeletum_hand_finger_1_key_4_phases_!/pressed/4_multistart_different_minimisations/results/2_results_analysis/pareto_front_curve_of_one_proximal_limb_torques_d._on_distal_limbs_torques/tab_tau_each_dof.pckl",
    "rb",
) as file:
    tab_tau = pickle.load(file)
from scipy.integrate import quad

# # # labels # # #
label = tab_tau.simulation.values.tolist()

# # # x # # #
x = []
a = 0
for j in range(len(tab_tau)):
    a = 0
    for i in range(7, 11):
        a += abs(tab_tau.iloc[j][i])
    x += [a]

# Every proximal limbs #
y_every_proximal_limbs = []
for j in range(len(tab_tau)):
    b = 0
    for i in range(1, 7):
        b += abs(tab_tau.iloc[j][i])
    y_every_proximal_limbs += [b]

# # # y # # #

# Humerus #
y_humerus = []
for j in range(len(tab_tau)):
    c = 0
    for i in range(4, 7):
        c += abs(tab_tau.iloc[j][i])
    y_humerus += [c]

# Thorax #
y_thorax = []
c = 0
for j in range(len(tab_tau)):
    d = 0
    for i in range(2, 4):
        d += abs(tab_tau.iloc[j][i])
    y_thorax += [d]

# Pelvis #
y_pelvis = []
for j in range(len(tab_tau)):
    e = 0
    for i in range(1, 2):
        e += abs(tab_tau.iloc[j][i])
    y_pelvis += [e]

# # # Plot # # #

figU, axs = plt.subplots(2, 2)
plt.subplots_adjust(top=0.88, bottom=0.11, left=0.125, right=0.9, hspace=0.295, wspace=0.2)

for i, txt in enumerate(label):
    datacursor(axs[0, 0].scatter(x[i], y_every_proximal_limbs[i], label=label[i]))
axs[0, 0].set_title("Every proximal limbs", fontsize=14, fontweight="bold")
axs[0, 0].set_xlabel("Σ(| tau_distaux * dt |) (en N.m)", fontsize=12)
axs[0, 0].set_ylabel("Σ(| tau_proximaux * dt |) (en N.m)", fontsize=12)
for i, txt in enumerate(label):
    datacursor(axs[0, 1].scatter(x[i], y_humerus[i], label=label[i]))
axs[0, 1].set_title("Humerus", fontsize=14, fontweight="bold")
axs[0, 1].set_xlabel("Σ(| tau_distaux * dt |) (en N.m)", fontsize=12)
axs[0, 1].set_ylabel("Σ(| tau_humerus * dt |) (en N.m)", fontsize=12)
for i, txt in enumerate(label):
    datacursor(axs[1, 0].scatter(x[i], y_thorax[i], label=label[i]))
axs[1, 0].set_title("Thorax", fontsize=14, fontweight="bold")
axs[1, 0].set_xlabel("Σ(| tau_distaux * dt |) (en N.m)", fontsize=12)
axs[1, 0].set_ylabel("Σ(| tau_thorax * dt |) (en N.m)", fontsize=12)
for i, txt in enumerate(label):
    datacursor(axs[1, 1].scatter(x[i], y_pelvis[i], label=label[i]))
axs[1, 1].set_title("Pelvis", fontsize=14, fontweight="bold")
axs[1, 1].set_xlabel("Σ(| tau_distaux * dt |) (en N.m)", fontsize=12)
axs[1, 1].set_ylabel("Σ(| tau_pelvis * dt |) (en N.m)", fontsize=12)

figU.suptitle(
    "Rapport entre l integral de la valeur absolue des couples de trois articulations proximales différentes"
    " et l integral de la valeur absolue des couples des articulations distales\n"
    "pour plusieurs simulations minimisant plus ou moins les couples articulaires distaux.",
    fontweight="bold",
    size=12,
)
plt.show()
