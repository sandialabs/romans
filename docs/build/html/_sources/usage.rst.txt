.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

.. _usage:

Operation
=========

The general workflow used to accelerate numerical simulations with this code is as
follows:

#. Produce an ensemble of simulation runs by varying simulation input parameters.
#. Perform any necessary pre-processing of data, including file format transformations.
#. Perform dimension reduction analysis and visualization.
#. Use the reduced represention to train a proxy model.
#. Test proxy and produce statistics to determine effectiveness.
#. Acclerate simulation using proxy model.

This code is designed to be modular and extensible, so that at each step different
algorithms and/or simulations can be used.

.. _ensemble-data:

Ensemble Data
-------------

The training of machine learning proxy models requires the 
use and analysis of multiple runs of a simulation.  This type of
data is known as ensemble data.  Although different simulations
are organized differently, ensemble projects are typically
set up as follows:

.. code-block:: python

    ensemble
    |-- ensemble.info
    |-- simulation.1
        |-- simulation.info
        |-- time.step.1
        |-- time.step.2
        |-- ...
    |-- simulation.2
    |-- simulation.3
    ...

These utilities can use either a python like ``%d[::]`` specifier
to list the simulation directories or a .csv file with a header
and one row per simulation.  Each column in the .csv file
contains information and/or file links to results produced
by the simulation.

Pre-processing
--------------

table.py ``--create``
^^^^^^^^^^^^^^^^^^^^^

To create a .csv file, the table.py script can be used.  The table.py
script is a command line utility which can read the meta-data files for a simulation
ensemble and produces a .csv table with rows containing data for a given simulation
and columns containing the simulation variables.

An example would be:

.. code-block:: python

    python -m table --create \
        --ensemble data/phase-field/test_data/workdir.%d \
        --output-dir data/phase-field/test_out \
        --input-files in.cahn_hilliard \
        --input-header "Input Deck" \
        --csv-out metadata.csv \
        --over-write \
        --csv-index "Simulation Index"

The above example produces a .csv file with relative file links.  These links will only
work with other utilities if everything is run from the same directory.  To produce
absolute links, use absolute links in the command line call, e.g.:

.. code-block:: python

    python -m table --create \
        --ensemble data/phase-field/test_data/workdir.%d \
        --output-dir data/phase-field/test_out \
        --input-files in.cahn_hilliard \
        --input-header "Input Deck" \
        --csv-out metadata.csv \
        --over-write \
        --csv-index "Simulation Index"

If you want to produce a .csv table with links associated with a particular computer,
use the URI notation in the call, e.g. ``file://host/path``.

The full set of options available for the table.py utility can be found in :ref:`utilities`.

convert.py
^^^^^^^^^^

Another useful pre-processing utility is the convert.py script.  Convert.py is
a command-line utility which can convert from one file type to another, for example
from a mesh format to a matrix format, or from multiple .jpg images to an .mp4 file.

For example, to convert from MEMPHIS .vtk to matrix .npy files, use:

.. code-block:: python

    python -m convert --ensemble data/phase-field/test_data/workdir.1 \
        --input-files out.cahn_hilliard_%d.vtk \
        --output-dir data/phase-field/test_out/workdir.1 \
        --output-format npy \
        --over-write \
        --field-var phase_field

Or to create a movie from MEMPHIS .vtk files, use:

.. code-block:: python

    python -m convert --ensemble data/phase-field/test_data/workdir.1 \
        --input-files out.cahn_hilliard_%d.vtk \
        --output-dir data/phase-field/test_out/workdir.1 \
        --output-format mp4 \
        --over-write \
        --field-var phase_field \
        --color-scale 0 1

Again see :ref:`utilities` for the full set of options for convert.py.

table.py ``--join``
^^^^^^^^^^^^^^^^^^^

The table.py script can also be used to add columns to the ensemble table created using
``--create`` option.  For example, to add file pointers to movies created by convert.py to 
the ensemble table use:

.. code-block:: python

    python -m table --join data/phase-field/test_out/metadata.csv data/phase-field/test_out/end-state.csv data/phase-field/test_out/movies.csv \
        --output-dir data/phase-field/test_out \
        --csv-out ps.csv \
        --over-write \
        --csv-no-index \
        --ignore-index \
        --csv-headers mobility_coefficients-1 mobility_coefficients-2 composition_distribution-1 "End State" "Movie" \
        --uri-root-out file://memphis/phase-field/test_out \
        --convert-cols "End State" "Movie"

Full options for ``table.py`` can be found in :ref:`utilities`.

Dimension reduction
-------------------

A variety of dimension reduction techniques can be applied to simulation output
using the reduce.py script.  For example, the following command performs dimension 
reduction using PCA on the final time step of a phase-field simulation.

.. code-block:: python

    python -m reduce --ensemble data/phase-field/test_data/workdir.%d \
        --input-files out.cahn_hilliard_50000000.vtk \
        --output-dir data/phase-field/test_out \
        --output-file out.cahn_hilliard_PCA.rd.npy \
        --algorithm PCA \
        --num-dim 2 \
        --over-write \
        --field-var phase_field \
        --auto-correlate --binary \
        --xy-out auto-PCA-end-state.csv \
        --xy-header "Auto-PCA End State"

Many additional algorithms can be used, including Isomap, tSNE, deep learning
auto-encoders, and a time-aligned meta-algorithm specifically for use with simulation
output.  More examples can be found in :ref:`example`.

Training Proxy
--------------

The romans tools allows the user to train various reduced order proxy models for a simulation.
To train an LSTM model, for example, use the model.py command line interface, as follows:

.. code-block:: python

    python -m model --ensemble data/phase-field/test_out/workdir.%d[0:401] \
        --input-file out.cahn_hilliard_inc_auto_PCA_10.rd.npy \
        --train %d[20:90] \
        --over-write \
        --output-model LSTM-model.pkl \
        --algorithm LSTM \
        --num-dim 5 \
        --LSTM-steps 10

    python -m model --ensemble data/phase-field/test_out/workdir.%d[401:] \
        --input-file out.cahn_hilliard_inc_auto_PCA_10.rd.npy \
        --output-file LSTM-preds.px.npy \
        --test %d[20:90] 11 \
        --over-write \
        --input-model data/phase-field/test_out/LSTM-model.pkl

Additional algorithms are also avaialbe and can be added by the user.  More examples can be
found in :ref:`example`.

Testing Proxy
-------------

Trained proxy models can be tested by making predictions and producing simple plots showing
the proxy model predictions.  For example,

.. code-block:: python

    python -m validate --proxy \
        --ensemble data/phase-field/test_out/workdir.%d[401:] \
        --input-file out.cahn_hilliard_inc_auto_PCA_10.rd.npy \
        --output-file data/phase-field/test_out/LSTM-predictions \
        --input-pred-file LSTM-preds.px.npy \
        --input-model data/phase-field/test_out/LSTM-model.pkl \
        --test %d[10:90] 11

Acceleration
------------

The reduced order models trained using romans can also be coupled with a simulation to greatly
increase the speed of obtaining simulation results.  This depends on the particular simulation 
and has yet to be implemented.

