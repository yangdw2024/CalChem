# MolForge VTK Zero-Copy Streaming Bridge
import numpy as np

def update_vtk_orbital_image(wfn, orbital_index: int, vtk_image_data, dims=(80, 80, 80)):
    try:
        from calchem import _core
        origin = [0.0, 0.0, 0.0]
        spacing = [0.1, 0.1, 0.1]
        grid_3d = _core.eval_orbital_cube_zero_copy(orbital_index, origin, spacing, list(dims))
        # Zero-copy pointer attachment to VTK
        from vtkmodules.util.numpy_support import numpy_to_vtk
        vtk_arr = numpy_to_vtk(grid_3d.ravel(), deep=0)
        vtk_image_data.GetPointData().SetScalars(vtk_arr)
        return True
    except ImportError:
        return False
