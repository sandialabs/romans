.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

.. _extension:

Extension
=========

The romans package is designed to be easily extensible and adaptable for use with
any numerical simulation.  The package consists of a core of Python modules and classes
that perform functions that can be used with any simulation.  This core ranges from
keeping track of simulation files and metadata to performing dimension reduction 
and training proxy models.

In addition to this core there are utilities that provide a command line interface to 
the romans tools as well as plugins which form the basis of providing specific
functions for different simulations.

Utilities
---------

The utilities consist of the functions described previously in :ref:`usage`, :ref:`utilities`
and :ref:`example`.  These utilities can be called directly from the command line, or can be
called from within another Python program.  The following example shows how to call a utility 
from within Python.

.. code-block:: python

    import os

    # table command line code
    import romans.table as table

    # directory containing the phase-field test dataset
    test_data_dir = 'data/phase-field/test_data'

    # output directory to use for testing
    output_dir = 'data/phase-field/test_table'

    # create metadata table
    arg_list = ['--create',
                '--output-dir', output_dir,
                '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                '--input-files', 'in.cahn_hilliard',
                '--input-header', 'Input Deck',
                '--csv-out', 'metadata.csv',
                '--csv-index', 'Simulation Index']
    table.main(arg_list)

This example produces a metadata table from the input decks of a MEMPHIS simulation.

Plugins
-------

Plugins provide the specific routines needed for a particular simulation.  These routines
include code that reads specific file formats, file-conversion, and any particular pre-processing 
requirements.

The routines are provided to romans by over-riding functions in the ``PluginTemplate`` class
described in the following API section.  The ``PluginTemplate`` class provides very
basic functionality, including mesh file reading capability (using meshio, see :ref:`installation`).
Generally speaking, however, the basic template won't work for a particular simulation.

Plugins can be found in the romans/plugins source directory, but as an example, 
here a subroutine which over-rides the standard mesh reader in ``PluginTemplate``:

.. code-block:: python

    # read npy and sim.npy (also npy) formats
    def read_file(self, file_in, file_type=None):

        # check file extension, if not provided
        if file_type is None:

            # npy file type
            if file_in.endswith('.npy'):
                file_type = 'npy'

        # check if we have npy or sim.npy
        if file_type == 'npy':
            
            # read npy file
            try:
                data = np.load(file_in)
            except ValueError:
                self.log.error("Could not read " + file_in + " as a .npy file.")
                raise ValueError("Could not read " + file_in + " as a .npy file.")

        # otherwise default to mesh
        else:
            data = super().read_file(file_in, file_type)

        return data

API
---

.. autoclass:: romans.ArgumentParser
    :members:

----

.. autofunction:: romans.init_logger

----

.. autoclass:: ensemble.Table
    :members:

----

.. autoclass:: ensemble.EnsembleSpecifierError

----

.. autoclass:: romans.PluginTemplate
    :members:

----

.. autofunction:: romans.plugin

----

.. autoclass:: romans.algorithms.reduction.DimensionReduction
    :members:

----

.. autoclass:: romans.algorithms.proxy.ProxyModel
    :members:
