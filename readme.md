This project consists of utitlies that facilitate the analysis 
and acceleration of numerical simulations using proxy machine 
learning models.  The name "ROMANS" is short for "Reduced
Order Modelling for Accelerating Numerical Simulations".

This code is designed to control the underlying numerical
simulation.  In order to work with different simulations it
relies on a plugin architecture, wherein code specific to
a given simulation is provided by supplementing a general class.

Documentation describing the installation, use, and extension
of the code can be found in the docs directory.  To build the
documentation you must have sphinx installed 
(https://www.sphinx-doc.org/en/master/).

Type (from the docs directory):

    $ make html

To read the documentation, click on "index.html" in the build/html
directory.

Copyright (c) 2021 National Technology and Engineering Solutions
of Sandia, LLC.  Under the terms of Contract DE-NA0003525 with
National Technology and Engineering Solutions of Sandia, LLC,
the U.S. Government retains certain rights in this software.
