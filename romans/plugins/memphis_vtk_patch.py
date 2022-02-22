# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This is not actually a plugin.  It is a patch to meshio
# for using MEMPHIS.  See note below.

# S. Martin
# 12/18/2020

# 3rd party imports
from meshio.vtk import _vtk
import numpy as np

# patch to meshio.vtk to read MEMPHIS specific vtk format
# note that just importing this module will override the
# below function and the standard meshio read will read 
# MEMPHIS .vtk format
def _read_scalar_field(f, num_data, split, is_ascii):
    data_name = split[1]
    data_type = split[2].lower()
    try:
        num_comp = int(split[3])
    except IndexError:
        num_comp = 1

    # The standard says:
    # > The parameter numComp must range between (1,4) inclusive; [...]
    if not (0 < num_comp < 5):
        raise ReadError("The parameter numComp must range between (1,4) inclusive")

    dtype = np.dtype(_vtk.vtk_to_numpy_dtype_name[data_type])
    lt, _ = f.readline().decode("utf-8").split()
    if lt.upper() != "LOOKUP_TABLE":
        raise ReadError()

    if is_ascii:
        data = np.fromfile(f, count=num_data * num_comp, sep=" ", dtype=dtype)
    else:
        # Binary data is big endian, see
        # <https://www.vtk.org/Wiki/VTK/Writing_VTK_files_using_python#.22legacy.22>.
        dtype = dtype.newbyteorder(">")
        data = np.fromfile(f, count=num_data * num_comp, dtype=dtype)
        line = f.readline().decode("utf-8")
        if line != "\n" and line != "":
            raise ReadError()

    data = data.reshape(-1, num_comp)
    return {data_name: data}

# over-ride _vtk._read_scalar_field
_vtk._read_scalar_field = _read_scalar_field