.. 
   Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
   Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
   Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

Introduction
============

Numerical simulations are widely used to model physical phenomenon.  Generally speaking,
they are computationally expensive and therefore slow.  The goal of this project
is to provide tools for training a proxy model for a simulation which can be used in 
conjunction with the actual simulation to obtain results more quickly.

Although designed to accomodate general simulation data, this code has been implemented
to directly support efforts at accelerating phase field simulations using MEMPHIS
(https://github.com/memphis-snl/memphis).

This code is designed to be modular and extensible, so that different algorithms and/or 
simulations can be used.