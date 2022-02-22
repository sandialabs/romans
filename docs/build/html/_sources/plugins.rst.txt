.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

Plugins
=======

The romans package uses plugins to provide functionality for different simulations, or
really any other outside software.  The plugins largely handle differences in file formats
between different simulations and software.  The following plugins are available for romans.

User interaction with plugins is handled by command line arguments passed through romans 
and onto the plugin.  As a result, plugin authors must not use command line arguments
with the same or even prefix matched names of the romans, or romans utility arguments.  Any such
arguments will be interpreted incorrextly and will likely fail.

To see the availble options for a romans plugin, call the plugin directly with --help option,
as in:

.. code-block:: python

    python -m romans.plugins.memphis --help

MEMPHIS
-------

The MEMPHIS plugin is the default plugin when using romans and allows interoperatability 
with the MEMPHIS phase-field simulation.  MEMPHIS command line options are given below.

.. program-output:: python -m romans.plugins.memphis --help

VideoSwarm
----------

The VideoSwarm plugin provides output support for the files required to create a
VideoSwarm model in Slycat.  The plugin can be called from the table.py utlity as
described in :ref:`example`.  The command line options for the VideoSwarm plugin are
shown below.

.. program-output:: python -m romans.plugins.videoswarm --help

Parameter Space
---------------

The Parameter Space plugin supports output that can be used to create Parameter Space
models in Slycat.  The plugin can be called from the table.py utility using the `--expand`
option, as demonstrated in :ref:`example`.  The command line options for the plugin are
given below.

.. program-output:: python -m romans.plugins.parameter_space --help

