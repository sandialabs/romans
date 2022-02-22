.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

.. _installation:

Installation
============

Numerical simulations are typically run on high-performance computing clusters.
Thus, the romans package will typically be installed on a cluster.  Depending on
the cluster, the code can be installed in a different ways.

Requirements
------------

- Python 3: You will need Python 3 to use the romans tools.  At the time of this writing, it
  has been tested using version 3.8.3.

- ipyparallel: You will need ipyparallel to run on the clusters.  Use ``pip 
  install ipyparallel``.  You may also need to export the path to the ipython
  command in your .bash_profile file to provide access to local installs.

  .. code-block:: python

      pip install ipyparallel --proxy your-proxy:your-port --trusted-host pypi.python.org --trusted-host pypi.org --trusted-host files.pythonhosted.org
      export PATH=$PATH:path-to-ipyparallel-bin

- meshio: You will need to install the meshio package (currently using 
  https://pypi.org/project/meshio/1.8.9/), use ``pip install meshio``.

- imageio: For converting to images and videos, you will need to install imagio
  (https://pypi.org/project/imageio/) and imageio-ffmpeg (https://pypi.org/project/imageio-ffmpeg/)
  using ``pip install imageio`` and ``pip install imageio-ffmpeg``.

- sklearn: For the dimension redution algorithms, you need scikit-learn (see 
  https://scikit-learn.org/).  Use ``pip install sklearn``.

- umap-learn: For the dimension reduction algorithm Umap (https://umap-learn.readthedocs.io/en/latest/), 
  you need ``pip install umap-learn``.  Note: do not ``pip install umap`` as that is an entirely different package.

- joblib: You may also need joblib, depending on whether or not it is included with your
  version of sklearn (https://joblib.readthedocs.io/en/latest/).  You can instll joblib with
  ``pip install joblib``.

- torch: For the neural networks we use PyTorch (https://pytorch.org/), use ``pip install torch``.

- pymks: The MEMPHIS plugin uses PyMKS (http://pymks.org/en/stable/rst/README.html), use 
  ``pip install pymks``.

Note that you may need to use ``--proxy`` and ``--trusted-host`` options with pip to install
meshio from behind a firewall, as shown in the ipyparallel example above.

Local
-----

To install the package from source, first clone the repository in your home directory,
e.g.

.. code-block:: python

    git clone https://github.com/your-user-name/romans.git

Next, set your PYTHONPATH environment variable using, for example

.. code-block:: python

    export PYTHONPATH=/users/your-user-name/romans:/users/your-user-name/romans/romans

Note that you can set this path in ~/.bash_profile if you are using Unix so that
``PYTHONPATH`` will be automatically set whenever you login.