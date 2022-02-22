Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

This directory contains code to exercise the romans module.  To run
the tests, type (e.g.)

$ python test-convert.py

Some of the tests take a long time, so there are flags to specify
which tests you want to run, as in

$ python test-reduce.py --test-UI

The results of the tests are usually stored in files specified
by variables defined in a given test-*.py file.  To change the
directory you need to edit the test-*.py files directly.
