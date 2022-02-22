.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

Algorithms
==========

The romans package uses various algorithms to analyze simulation data.  The algorithms
are implemented as plugins and include dimension reduction algorithms from the Python
package scikit-learn and the deep learning package pytorch.  Custom algoirthms are also
available.

Further, the algorithms are implemented as plugins, so user defined algorithms can also
be added.  Command line arguments are used to pass parameters to the algorithms.

To see the availble options for a romans algorithm plugin, call the plugin directly with 
--help option, as in:

.. code-block:: python

    python -m romans.algorithms.reduction --help

Dimension Reduction
-------------------

The dimension reduction algorithms are provided in the ``algorithms/reduction.py``
module.  Command line options are given below.

.. program-output:: python -m romans.algorithms.reduction --help

Proxy Models
------------

The reduced order proxy model algorithms are provided in the ``algorithms/proxy.py``
module. The command line options for the proxy models are shown below.

.. program-output:: python -m romans.algorithms.proxy --help