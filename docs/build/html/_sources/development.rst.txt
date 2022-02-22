.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

Development
===========

In addition to extending the toolbox, development of additional core features is also
possible.  The easiest way to further develop romans is to use the drivers in the ``test``
directory.  

The general approach taken to date is to write a utility code (:ref:`utilities`) 
that interfaces betweent the command line and the actual romans classes and plugins.  To test
these utility codes, we have written a variety of drivers which set up the parameters
and call the utilites from Python.  We use the MEMPHIS data to test the codes, and pointers to
the data is coded directly in the test scripts.

Therefore the test scripts will not work for a general user, but if you provide test data
and modify the scripts to point to that data, then the test scripts should work.  Alternatively, 
you can write your own utilities and test codes.

Finally, there is a certain amount of inter-dependence between the tests.  For example, 
``test-reduce.py`` must precede ``test-table.py`` to provide files for creating tables.

test-table.py
-------------

This code tests the ``table.py`` utility and can be executed using:

.. code-block:: python

    python test-table.py

Since the ``table.py`` utility is not computationally expensive, there are no command line options.
Rather, every test is exectued.  For the other utilities, there are command line options.

test-convert.py
---------------

.. program-output:: python ../../tests/integration/test-convert.py --help

test-reduce.py
--------------

.. program-output:: python ../../tests/integration/test-reduce.py --help

test-model.py
-------------

.. program-output:: python ../../tests/integration/test-model.py --help

test-validate.py
----------------

.. program-output:: python ../../tests/integration/test-validate.py --help
