# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This is a plugin that provides support for converting dimension
# reduction results to Parameter Space format.  In particular
# it is to be called using the --expand option from table.py.

# S. Martin
# 4/21/2021


# standard library imports
import argparse
import csv
import os

# 3rd party imports
import numpy as np

# local imports
import ensemble
import romans


# functions for specific operations
# class must be named Plugin
class Plugin(romans.PluginTemplate):

    # initialize command line argument parser
    def __init__(self):

        # describe plugin
        description = "The parameter space plugin provides the ability to convert " + \
            "dimension reduction results to Slycat Parameter Space " + \
            "format."

        # set up parser
        super().__init__(description=description)

    # add any extra command line arguments
    def add_args(self):

        self.parser.add_argument("--num-dim", type=int, default=2, help="Number of " +
            "from input files to include in .csv output file.")

        self.parser.add_argument("--remove-expand-col", action="store_true", default=False,
            help="Remove the expanded column when writing out parameter space file.")

        self.parser.add_argument("--include-original-index", action="store_true", default=False,
            help="Add original (repeated) index to expanded output .csv file.")

    # check arguments, error if incorrect input
    def check_args(self, args):

        if args.num_dim is not None:
            if args.num_dim < 2:
                self.log.error("Number of dimensions to output must be >= 2.")
                raise ValueError("number of dimensions to output must be >= 2 (see " +
                                 "romans.plugins.parameter_space --help).")

    # initialize any local variables from command line arguments
    def init(self, args):

        # save args for later use
        self.args = args

    # over-ride file read, only accept .rd.npy files
    def read_file(self, file_in, file_type=None):

        # check that file ends with .rd.npy
        if not file_in.endswith('.rd.npy'):
            self.log.error("The videoswarm plugin accepts only .rd.npy files.")
            raise TypeError("videoswarm plugin accepts only .rd.npy files.")
        
        # read npy file
        try:
            data = np.load(file_in)
        except ValueError:
            self.log.error("Could not read " + file_in + " as a .npy file.")
            raise ValueError("could not read " + file_in + " as a .npy file.")

        self.log.info("Read file: " + file_in)

        return data

    # over-riding expand to generate videoswarm input files
    def expand(self, table, header, file_list, **kwargs):
        """
        Outputs the Parameter Space input file from a set of .rd.npy files.

        Optional arguments include (see also table.py options):
            output_dir -- output directory
            csv_out -- file name for csv output
            csv_no_index -- do not include pandas index in movies.csv
            csv_index_header -- use this header name for index column
            csv_headers -- output only these columns to movies.csv

        Other arguments passed via kwargs are ignored.
        """
        
        # decode extra arguments
        output_dir = kwargs.get("output_dir")
        csv_out = kwargs.get("csv_out")
        csv_no_index = kwargs.get("csv_no_index", False)
        csv_index_header = kwargs.get("csv_index_header", None)
        csv_headers = kwargs.get("csv_headers", None)

        # put csv_out in output directory
        csv_out = os.path.join(output_dir, csv_out)

        # file_list is in order of table, expecting .rd.npy files
        reduced_coords = []
        for file_name in file_list:
            
            # read file
            data = self.read_file(file_name)

            # check that data is at least two dimensional
            if data.shape[1] < 2:
                self.log.error("Data in " + file_name + " contains less than " +
                               "two dimensions.")
                raise ValueError("reduced dimension data must have at least two dimensions.")
            
            # check that number of dimension is not too large
            if data.shape[1] < self.args.num_dim:
                self.log.warning("Data in " + file_name + " contains less than " +
                                 "requested number of dimensions to output -- defaulting " +
                                 "to " + str(data.shape[1]) + " dimensions.")
                self.args.num_dim = data.shape[1]

            # collate data (each .rd.npy file contains time steps 
            # per simulation as rows and reduced data as columns)
            reduced_coords.append(data[:,0:self.args.num_dim])

        # convert reduced coords into list per dimension
        reduced_coords_per_dim = [[] for i in range(self.args.num_dim)]
        for i in range(self.args.num_dim):
            for j in range(len(reduced_coords)):
                reduced_coords_per_dim[i].append(reduced_coords[j][:,i])

        # keep track of original index
        original_index = []
        for j in range(len(reduced_coords)):
            original_index.append((j+1) * np.ones(reduced_coords[j].shape[0]))

        # write out files
        self._output_PS_files(table, header, reduced_coords_per_dim,
            original_index, csv_out, csv_no_index, csv_index_header, csv_headers)

    # write out Parameter Space files
    def _output_PS_files(self, meta_data, expand_header, reduced_coords_per_dim, 
        original_index, csv_out, csv_no_index, csv_index_header, csv_headers):
        
        # remove expanded column, if requested
        if self.args.remove_expand_col:
            meta_data.table = meta_data.table.drop(expand_header, axis=1)

        # add a new column for each requested dimension
        for i in range(self.args.num_dim):

            # for 1st dimension, expand table
            if i == 0:

                # add either original index + first dimension column
                if self.args.include_original_index:
                    expanded_data = ensemble.explode(self.log, meta_data,
                        'Original Index', original_index)
                    expanded_data.add_col(np.concatenate(reduced_coords_per_dim[i]).tolist(),
                        expand_header + " Dimension " + str(i+1))

                # or just first dimension column
                else:
                    expanded_data = ensemble.explode(self.log, meta_data,
                        expand_header + " Dimension " + str(i+1), reduced_coords_per_dim[i])

            # for other dimensions, just add column
            else:
                expanded_data.add_col(np.concatenate(reduced_coords_per_dim[i]).tolist(),
                    expand_header + " Dimension " + str(i+1))

        # write out .csv file
        expanded_data.to_csv(csv_out, index=csv_no_index, 
                              index_label=csv_index_header,
                              cols=csv_headers)

# if called from the command line display plugin specific command line arguments
if __name__ == "__main__":

    parameter_space = Plugin()
    parameter_space.parser.parse_args()