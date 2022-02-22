# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains the underlying code for doing dimension reduction
# and time acceleration proxy models for numerical simulation codes.

# See docs/build/html/index.html file for documentation.

# S. Martin
# 11/02/2020


# standard library imports

# parsing command line
import argparse

# logging to file/screen
import logging
import sys

# parsing %d[::] format
import re

# import module from file
import importlib
import pkgutil
import os

# 3rd party imports

# meshio is default file reader/writer
import meshio

# miscellaneous array computations
import numpy as np

# parallel comutation
from ipyparallel import Client

# local imports
import plugins


# constants/parameters for the romans package
__version__ = "1.0"

# defaults
DEFAULT_LOG_LEVEL = 'info'
DEFAULT_PLUGIN = 'memphis'


# set up argument parser with logging
class ArgumentParser(argparse.ArgumentParser):
    """
    Return an instance of argparse.ArgumentParser, pre-configured with arguments
    used to run romans tools.
    
    Command line flags defined by this class are:

    * -\\-log-level
    * -\\-log-file
    * -\\-plugin

    The ``--log-level`` flag specifies the log level to use for the screen, the
    ``--log-file`` flag specifies a log file to write (debug and above), and the
    ``--plugin`` flag specifies the plugin Python file to use.

    :Example:

    .. code-block:: python

        import romans

        # describe and initialize parser
        my_description = "My extension of the romans parser."
        my_parser = romans.ArgumentParser(description=my_description)

        # add an argument
        my_parser.add_argument('--my-flag', help="My extension command line flag")
    """

    # create argument parser with logging level and log file
    def __init__(self, *arguments, **keywords):

        argparse.ArgumentParser.__init__(self, *arguments, **keywords)

        # logging options
        self.add_argument("--log-level", default=DEFAULT_LOG_LEVEL, 
            choices = ["debug", "info", "warning", "error", "critical"], 
            help = "Log level. Default: '%(default)s'")
        self.add_argument('--log-file', type=str,
            help = "Log to file. Notes: (1) If this file already exists it will be "
                 + "overwritten, (2) Log file includes time stamp and is set to "
                 + "debug level")

        # plugin identification
        self.add_argument("--plugin", default=DEFAULT_PLUGIN,
            help = "Plugin Python file name to import (defaults to memphis), "
                   "can be either a plugin from romans/plugins (no extension) "
                   'or a python file (.py extension). Use "python -m '
                   'romans.plugins.plugin --help" to see any command line '
                   'options for the plugin.')

    # parse logging arguments as well as additional arguments
    def parse_args(self, list_input=None):
        """
        Extends standard argparse.parse_args() call.  Uses parse_known_args()
        to parse the base arguments and any additional arguments.  Returns any
        unknown arguments as a list.  The unknown arguments can be used to
        set plugin specific variables.

        :Example:

        .. code-block:: python

            # parse command line
            args = my_parser.parse_args()

            # parse command line and start logger
            args, unknown_arg_list = my_parser.parse_args()

            # arguments can be accessed using, e.g.
            print(args.log_level)

            # unknown arguments are returned as a list
            print(unknown_arg_list)
            
        :Warning: 
        
        Python uses prefix-matching, so that if a plugin uses an 
        argument flag that matches the prefix of an already existing argument
        (including any argument defined in a utility), that argument will 
        be not be passed onto the plugin.
        """

        # parse arguments
        return argparse.ArgumentParser.parse_known_args(self, list_input)


# starts romans logger
def init_logger(log_file=None, log_level=DEFAULT_LOG_LEVEL):
    """
    Starts romans logger, sets logging level for console and opens log file,
    if desired.  Console only outputs messages while log file includes time stamps
    and origin of message.  Log file is set to debug and above.

    :Example:

    .. code-block:: python

        import logging

        # start logger for extension
        my_log = logging.getLogger("romans.my_extension")
        my_log.info("My log message.")

        # or use romans logger
        romans.log.info("My message.")
    """

    # convert log level to logging class attribute
    log_level_attr = getattr(logging, log_level.upper())

    # set up romans logger
    global log
    log = logging.getLogger(__name__)
    
    # log to file, if requested
    if log_file:
        logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                datefmt='%m-%d %H:%M',
                filename=log_file,
                filemode='w',
                force=True)

        # also write messages to console
        console = logging.StreamHandler()
        formatter = logging.Formatter('%(message)s')
        console.setFormatter(formatter)
        console.setLevel(log_level_attr)

        # add console
        log.addHandler(console)

    # otherwise only log to console
    else:
        logging.basicConfig(level=log_level_attr,
                            format='%(message)s')

    log.debug("Started romans log.")


# template for plugin architecture to accomodate different 
# i/o formats, ML algorithms, and simulations
class PluginTemplate:
    """
    Provides an extensible architecture for accomodating different input/output
    formats, machine learning algorithms, and simulations.

    Plugins must be defined as following in a seperate ``.py`` file:
    
    :Example:

    .. code-block:: python
    
        import romans

        class Plugin(romans.PluginTemplate):
            
            ...

    See ``memphis.py`` for an example.
    """

    # initializes plugin by setting up argument parser with
    # any additional command line flags
    def __init__(self, description=None):

        # initialize plugin parser
        self.parser = argparse.ArgumentParser(description=description)

        # add parser arguments
        self.add_args()

        # start plugin logger
        self.log = logging.getLogger("romans.plugin")
        self.log.debug("Started plugin.")

    # add any extra command line arguments
    def add_args(self):
        """
        Note these flags should not conflict with already used flags (see
        parse_args class ArgumentParser).

        :Example:

        .. code-block:: python

            # plugin adds command line argument
            self.parser.add_argument("--my_option", help="My option for plugin.")
        """

        pass

    # parse any plugin specific arguments
    def parse_args(self, arg_list=[]):
        """
        Parses arguments specific to plugin.

        Args:
            arg_list (list): list of command line flags and arguments

        Returns:
            args (object), unknown_args (list): `ArgumentParser` processed argument list,
            list of un-recognized arguments
        """

        # parse only known arguments, ignore others
        args, arg_list = self.parser.parse_known_args(arg_list)

        return args, arg_list

    # check arguments, error if incorrect input
    def check_args(self, args):
        """
        Checks plugin arguments and raises exceptions if there are errors.

        Args:
            args (`ArgumentParser` object): processed argument list
        """

        pass

    # initialize any local variables from command line arguments
    def init(self, args):
        """
        Initialize any local variables from command line arguments.

        Args:
            args (`ArgumentParser` object): processed argument list

        :Example:

        .. code-block:: python

            self.my_var = args.my_option
        """

        pass

    # generic reader
    def read_file(self, file_in, file_type=None):
        """
        Reads a file associated with a single time step in an ensemble.
        File type is inferred from extension unless provided.

        Args:
            file_in (string): name of file to read
            file_type (string): file input type (regardless of extension)

        Returns:
            data (object): file contents
        """

        # read file using meshio
        mesh = meshio.read(file_in, file_type)

        # some converters (like VTK) require `points` to 
        # be contiguous (from meshio-convert)
        mesh.points = np.ascontiguousarray(mesh.points)

        return mesh

    # read files in batches (can run in parallel)
    def read_file_batch(self, batch_files, file_type=None, parallel=False, flatten=True):
        """
        Reads a batch of files from an ensemble.  File type is inferred from
        extension unless provided.  Can be run in parallel using ipyparallel.

        Args:
            batch_files (list): list of files to read
            file_type (string): file input type (regardless of extension)
            parallel (boolean): to run in parallel using ipyparallel
            flatten (boolean): flatten matrix files to vectors (defaults True)

        Returns:
            data_list (list): file of file contents
        """

        # if not parallel, just get all sim files normally
        if not parallel:
            return self._get_file_batch(batch_files, file_type=file_type, flatten=flatten)

        # otherwise read files in parallel

        # parallel operation, using direct view
        rc = Client()
    
        # use cloudpickle because pickle won't work with plugins
        rc[:].use_cloudpickle()

        # get number of available engines
        num_engines = len(rc)

        # break files into blocks so that we use all available engine
        num_files = len(batch_files)

        # push out parallel jobs, one load per engine
        data = []
        for i in range(0, num_files, num_engines):
            
            # get block of files
            block_files = [batch_files[j] for j in range(i, i + num_engines) if j < num_files]

            # push out jobs per engine
            async_results = []
            for j in range(len(block_files)):
                async_result = rc[j].apply_async(self._get_file_batch, 
                    [block_files[j]], file_type=file_type, flatten=flatten)
                async_results.append(async_result)

            # wait for results
            rc[:].wait(async_results)

            # get results
            for j in range(len(async_results)):

                # get items in order they were put into queue
                async_result = async_results.pop(0)

                # retrieve data from result
                data += async_result.get()

            # clean up ipyparallel
            rc.purge_everything()

        # close ZMQ sockets
        rc.close()

        return data

    # helper function to read in a batch of files
    def _get_file_batch(self, batch_files, file_type=None, flatten=True):

        # read all files into list
        data = []
        for i in range(len(batch_files)):
                
            # get data from file
            data_i = self.read_file(batch_files[i], file_type=file_type)

            # do plugin preprocessing  (doesn't do anything if not enabled)
            data_i = self.preprocess(data_i, flatten=flatten)

            # add data to list
            data.append(data_i)

        return data

    # generic writer
    def write_file(self, data, file_out, file_type=None):
        """
        Writes time step data from an ensemble to a file.  File type is
        inferred from extension unless provided.

        Args:
            data (`meshio` mesh): mesh data to be written
            file_out (string): file name for output file
            file_type (string): file extension
        """

        # write out data using meshio
        if isinstance(file_type, str):
            meshio.write(file_out, data, file_format=file_type)

        else:
            self.log.error('Unrecognized output file format for "%s" -- file not written.' %
                file_out)

    # generic file converter
    def convert_file(self, file_in, file_out, file_in_type=None, file_out_type=None):
        """
        Converts from file_in to file_out, where file_in can be a string or
        a buffer.  File types are inferred from extensions unless provided.  Uses
        the meshio library.

        Args:
            file_in (string): name of input file
            file_out (string): name of output file
            file_in_type (string): file input format (regardless of extension)
            file_out_type (string): file output format (extension)
        """

        # read in file data
        mesh = self.read_file(file_in, file_type=file_in_type)

        # write out data
        self.write_file(mesh, file_out, file_type=file_out_type)

    def convert_files(self, file_list, output_dir, output_type, input_type=None):
        """
        Converts a list of files to file of type output_type in output_dir with
        same root name.  Input file types are inferred from extensions, unless type
        is provided.  Output type is also inferred, unless provided.

        Args:
            file_list (list): list of file names to read
            output_dir (string): name of output directory to write files
            output_type (string): extension of file format for output
            input_type (sring): file input type (regardless of extension)

        Returns:
            files_written (list): list of files written using full path
        """

        # keep track of files written
        files_written = []

        # convert files in list
        for file_to_convert in file_list:

            # create output file name
            file_name = os.path.basename(file_to_convert)
            file_root, file_ext = os.path.splitext(file_name)
            file_out = os.path.join(output_dir, file_root + "." + output_type)

            # convert file
            self.convert_file(file_to_convert, file_out, 
                file_in_type=input_type, file_out_type=output_type)

            files_written.append(file_out)
        
        return files_written

    # input deck reader
    def read_input_deck (self, file_list, file_type=None):
        """
        Reads a file or files which provide the input parameters for a simulation.
        Note that this code must be provided by the plugin.

        Args: 
            file_list (list): list of file names to read (can be a list of one file)
            file_type (string): file type (regardless of extension)
        
        Returns:
            file_data (object): meta data for the simulation
        """

        pass

    # pre-processing specific to simulation
    def preprocess (self, data):
        """
        Performs data pre-processing specific to a simulation.  This code must
        be provided by the plugin, otherwise the data is returned unchanged.  Note
        that this type of pre-processing is assumed to be per file (e.g. per time
        step or per simulation).  Pre-processing that occurs over the entire
        ensemble is done by the algorithm codes (e.g. dimension reduction or 
        proxy models).

        Args: 
            data (object): data to be pre-processed

        Returns:
            data_out (2d array): pre-processed numpy array with simulations per row
        """

        return data

    # expand table by reading file contents
    def expand (self, table, header, file_list, **kwargs):
        """
        Expands a column in a ensemble table by reading the file links and 
        creating files appropriate to the plugin.

        Args:
            table (ensemble Table object): table with column containing file links
            header (string or int): name of column to with file links
            file_list (list): list of files to expand
            **kwargs: additional arguments dependent on plugin
        """

        pass

# factory function to instantiate and initialize a plugin 
# using a module from a file and a list of arguments
def plugin (plugin_name, arg_list=None):
    """
    Factory function to instantiate a plugin module from a file and a list of
    command line arguments.

    Args:
        plugin_name (string): module name (no .py) or file name of module (ending in .py)
        arg_list (list): list containing command line flags and argumetns

    Returns:
        plugin (object), unknown_args (list): plugin as Python namespace,
        list of un-recognized arguments

    :Example:

    .. code-block:: python

        # import and initialize plugin
        plugin, unknown_args = romans.plugin(args.plugin, arg_list)     
    """

    # get plugin name extension
    ext = os.path.splitext(plugin_name)

    # if not a file name, check in plugins directory
    if ext != '.py':
        for loader, mod_name, is_pkg in pkgutil.iter_modules(
            plugins.__path__, plugins.__name__ + "."):

            # load module from plugins directory
            if mod_name.endswith("." + plugin_name):                
                mod = loader.find_module(mod_name).load_module(mod_name)

    else:

        # load plugin from file
        spec = importlib.util.spec_from_file_location(name, plugin_name)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

    # instantiate new plugin
    plug = mod.Plugin()

    # parse arguments 
    args, arg_list = plug.parse_args(arg_list)

    # check arguments
    plug.check_args(args)

    # initialize plugin variables
    plug.init(args)

    log.debug("Loaded plugin: %s." % plugin_name)

    # return plugin ready to go
    return plug, arg_list
