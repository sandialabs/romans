# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This is a minimal generic plugin that provides no addditional 
# functionality beyond the standard romans behavior.

# S. Martin
# 12/16/2020


# standard library imports
import argparse

# 3rd party imports

# local imports
import romans


# functions for specific operations
# class must be named Plugin
class Plugin(romans.PluginTemplate):

    # initialize command line argument parser
    def __init__(self):

        # describe plugin
        description = "The generic plugin is a minimal plugin that provides " + \
            "no additional support beyond the standard romans behavior"

        # set up parser
        super().__init__(description=description)


# if called from the command line display plugin specific command line arguments
if __name__ == "__main__":

    generic = Plugin()
    generic.parser.parse_args()