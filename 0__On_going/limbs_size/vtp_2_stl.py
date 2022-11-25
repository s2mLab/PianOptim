
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
    convertFile("/home/lim/Documents/Stage Mathilde/PianOptim/0__On_going/limbs_size/vpt_files/Geometry/clavicle.vtp",
                outdir="/home/lim/Documents/Stage Mathilde/PianOptim/0__On_going/limbs_size/stl_files")

