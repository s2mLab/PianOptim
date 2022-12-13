
#!/usr/bin/env python

import os
import vtk
# from numpy import stl


def convertFile(filepath, outdir):
    if not os.path.isdir(outdir):
        os.makedirs(outdir)
    if os.path.isfile(filepath):
        basename = os.path.basename(filepath)
        print("Copying file:", basename)
        basename = os.path.splitext(basename)[0]
        outfile = os.path.join(outdir, basename+".stl")
        reader = vtk.vtkXMLPolyDataReader()
        reader.SetFileName(filepath)
        reader.Update()
        writer = vtk.vtkSTLWriter()
        writer.SetInputConnection(reader.GetOutputPort())
        writer.SetFileName(outfile)
        return writer.Write() == 1
    return False


if __name__ == '__main__':
    convertFile("/0__On_going/limb_sizes/vpt_files/Geometry/clavicle.vtp",
                outdir="/home/lim/Documents/Stage Mathilde/PianOptim/a_Mathilde_2022/2__FINAL_MODELS_OSCAR/5___final___squeletum_hand_finger_1_key_4_phases/pressed/3_multistart_different_minimisations/results/2_results_analysis/pareto_front_curve_of_one_proximal_limb_torques_d._on_distal_limbs_torques")

