from matplotlib import pyplot as plt
import pickle
import matplotlib.patches as mpatches
import numpy as np

# Import results with pelvis rotZ
# with open("Piano_results_3_phases.pckl",'rb') as file:
#     new_dict = pickle.load(file)

with open("Piano_results_3_phases_without_pelvis_rotZ.pckl",'rb') as file:
    new_dict2 = pickle.load(file)

print(new_dict2)
print(new_dict2 == "Piano_results_3_phases.pckl")
print(type(new_dict2))