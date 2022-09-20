import pickle
import numpy as np

# Import results with pelvis rotZ
with open("Results/Piano_results.pckl", 'rb') as file:new_dict = pickle.load(file)
with open("Piano_minimisation_1_humerus.pckl", 'rb') as file: new_dict2 = pickle.load(file)

# Print the dic ###########################################
print(new_dict)
print(new_dict == "Piano_results.pckl")
print(type(new_dict))