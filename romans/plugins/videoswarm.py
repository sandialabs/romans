# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This is a plugin that provides support for converting time-aligned
# dimension reduction to VideoSwarm format.  To be called using
# the --expand option from table.py.  When calling from table.py use
# a dummy value for --csv-out.

# S. Martin
# 3/23/2021


# standard library imports
import argparse
import csv
import os

# 3rd party imports
import numpy as np

# local imports
import romans


# functions for specific operations
# class must be named Plugin
class Plugin(romans.PluginTemplate):

    # initialize command line argument parser
    def __init__(self):

        # describe plugin
        description = "The videoswarm plugin provides the ability to convert " + \
            "time-aligned dimension reduction algorithms to Slycat VideoSwarm " + \
            "format."

        # set up parser
        super().__init__(description=description)

    # add any extra command line arguments
    def add_args(self):

        self.parser.add_argument("--remove-expand-col", action="store_true", default=False,
            help="Remove the expanded column when writing out videoswarm files.")
        
        # arguments to specify time scale
        self.parser.add_argument("--video-fps", type=float, default=25,
            help="Video frames per second, must be > 0, defaults to 25.")

    # check arguments, error if incorrect input
    def check_args(self, args):

        if args.video_fps is not None:
            if args.video_fps <= 0:
                self.log.error("Video frames per second must be > 0.")
                raise ValueError("video frames per second must be > 0 (see " +
                                 "romans.plugins.videoswarm --help).")

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
        Outputs the videoswarm input files from a set of .rd.npy files.

        Optional arguments include (see also table.py options):
            output_dir -- output diredtory
            csv_no_index -- do not include pandas index in movies.csv
            csv_index_header -- use this header name for index column
            csv_headers -- output only these columns to movies.csv

        Other arguments passed via kwargs are ignored.
        """

        # decode extra arguments
        output_dir = kwargs.get("output_dir", "")
        csv_no_index = kwargs.get("csv_no_index", False)
        csv_index_header = kwargs.get("csv_index_header", None)
        csv_headers = kwargs.get("csv_headers", None)

        # videoswarm input files are of the form:

        # movies.csv -- .csv metadata file with column containing file links
        # movies.trajectories -- .csv file containing one row of time
        #                         points followed by x-coordinates over simulations
        # movies.xcoords -- .csv file containing one column per
        #                    simulation of x-coordinates over time
        # movies.ycoords -- .csv file containing one column per 
        #                    simulation of y-coordinates over time

        # coordinates are scaled to lie in [0,1]^2

        # file_list is in order of table, expecting .rd.npy files
        xcoords = []
        ycoords = []
        for file_name in file_list:
            
            # read file
            data = self.read_file(file_name)

            # check that data is at least two dimensional
            if data.shape[1] < 2:
                self.log.error("Data in " + file_name + " contains less than " +
                               "two dimensions.")
                raise ValueError("reduced dimension data must have at least two dimensions.")
            
            # collate data (each .rd.npy file contains time steps 
            # per simulation as rows and reduced data as columns)
            xcoords.append(data[:,0])
            ycoords.append(data[:,1])

        # transpose xcoords, ycoords 
        xcoords = np.array(xcoords).transpose()
        ycoords = np.array(ycoords).transpose()

        # scale coords to lie in [0,1]^2
        xcoords, ycoords = self._scale_coords(xcoords, ycoords)

        # write out videoswarm files
        self._output_VS_files(table, header, xcoords, ycoords, 
            output_dir, csv_no_index, csv_index_header, csv_headers)

    # scale coordinates for VideoSwarm interface
    def _scale_coords (self, xcoords, ycoords):

        # get range for x
        min_x = np.amin(xcoords)
        max_x = np.amax(xcoords)

        # get range for y
        min_y = np.amin(ycoords)
        max_y = np.amax(ycoords)

        # scale coordinates to be in [0,1]^2
        # if constant assign value of 1/2
        xcoords = (xcoords - min_x)
        if max_x > min_x:
            xcoords = xcoords / (max_x - min_x)
        else:
            xcoords = xcoords + 0.5

        ycoords = (ycoords - min_y)
        if max_y > min_y:
            ycoords = ycoords / (max_y - min_y)
        else:
            ycoords = ycoords + 0.5

        return xcoords, ycoords

    # write out VideoSwarm files
    def _output_VS_files(self, meta_data, expand_header, xcoords, ycoords, 
        output_dir, csv_no_index, csv_index_header, csv_headers):

        # remove expanded column, if requested
        if self.args.remove_expand_col:
            meta_data.table = meta_data.table.drop(expand_header, axis=1)

        # write out movies.csv file
        movies_out = os.path.join(output_dir, 'movies.csv')
        meta_data.to_csv(movies_out, index=csv_no_index, 
            index_label=csv_index_header, cols=csv_headers)

        # write out movies.xcoords file (use only float precision)
        xcoords_file_name = os.path.join(output_dir, 'movies.xcoords')
        with open(xcoords_file_name, 'w') as xcoords_file:
            csv_xcoords_file = csv.writer(xcoords_file)
            for i in xcoords.tolist():
                csv_xcoords_file.writerow(['{:f}'.format(x) for x in i])
        self.log.info("File written: " + xcoords_file_name)

        # write out movies.xcoords file (use only float precision)
        ycoords_file_name = os.path.join(output_dir, 'movies.ycoords')
        with open(ycoords_file_name, 'w') as ycoords_file:
            csv_ycoords_file = csv.writer(ycoords_file)
            for i in ycoords.tolist():
                csv_ycoords_file.writerow(['{:f}'.format(y) for y in i])
        self.log.info("File written: " + ycoords_file_name)

        # add time to first row of xcoords to make trajectories
        num_frames, num_movies = xcoords.shape
        vid_duration = num_frames / float(self.args.video_fps)
        time_row = np.linspace(0, vid_duration, num=num_frames)
        traj = np.ones((num_movies + 1, num_frames))
        traj[0, :] = time_row
        traj[1:, :] = xcoords.transpose()

        # write out movies.trajectories
        traj_file_name = os.path.join(output_dir, 'movies.trajectories')
        with open(traj_file_name, 'w') as traj_file:
            csv_traj_file = csv.writer(traj_file)
            for i in traj.tolist():
                csv_traj_file.writerow(['{:f}'.format(t) for t in i])

        self.log.info("File written: " + traj_file_name)

# if called from the command line display plugin specific command line arguments
if __name__ == "__main__":

    videoswarm = Plugin()
    videoswarm.parser.parse_args()