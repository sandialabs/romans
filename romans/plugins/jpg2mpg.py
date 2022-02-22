# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains code to convert a jpg collection to mp4 and npy.

# J. Glen
# 8/03/2021

# standard library imports
import os

# for writing mp4
import imageio

# 3rd party imports
import numpy as np

# materials knowledge system
# for exporting images
from matplotlib import cm

# local imports
import romans

# defaults for user arguments
JPG_QUALITY = 95
VIDEO_FPS = 25

# functions for jpg converter specific operations
# class must be named Plugin
class Plugin(romans.PluginTemplate):

    # initialize jpg converter command line argument parser
    def __init__(self):

        # describe jpg converter plugin
        description = "The jpg converter plugin provides support for the jpg converter " + \
                      "phase field simulations."

        # set up parser
        super().__init__(description=description)

    # add any extra command line arguments
    def add_args(self):

        self.parser.add_argument("--field-var", help="Name of field variable to analyze, "
                                                     'e.g. "--field-var phase_field".  The field variable name is included in the name '
                                                     "of any output file.")

        self.parser.add_argument("--binary", action="store_true", help="Converts field "
                                                                       "variable to binary by clipping anything less than 0 to 0 and anyting "
                                                                       "greater than 0 to 1.")

        self.parser.add_argument("--color-scale", nargs=2, type=float, help="Gives the color "
                                                                            "scale for the field variable when creating jpg or mp4, e.g. "
                                                                            '"--color-scale 0 1" for min color value of 0 and max color value of 1. '
                                                                            "Note that values above and below the color scale are automatically clipped.")

        self.parser.add_argument("--output-format", help="The output format options "
                                                         'recognized by the jpg converter plugin include: "npy" -- saves the '
                                                         'field variable for a single timestep to a numpy array; "sim.npy" -- '
                                                         'saves the field variable for every timestep in a simulation to a 3D '
                                                         'numpy array; "rd.npy" -- saves the reduced dimensional representation '
                                                         'to a numpy array (can be either a time step matrix or a 3D full simulation '
                                                         'matrix); "jpg" -- saves a .jpg image of the field variable for a single '
                                                         'timestep; "mp4" -- saves a .mp4 movie of the field variable for every timestep '
                                                         'in a simulation.')

        self.parser.add_argument("--output-quality", type=int, default=JPG_QUALITY,
                                 help="Quality of jpg image, as a number between 1 and 95 (only relevant "
                                      "if outputing images, defaults to %s)." % str(JPG_QUALITY))

        self.parser.add_argument("--video-fps", type=float, default=VIDEO_FPS,
                                 help="Number of frames per second for video creation, defaults to %s)." %
                                      str(VIDEO_FPS))

        self.parser.add_argument("--write-raw-video", action="store_true", help="Create a video using the provided images unprocessed.")

    # check arguments, error if incorrect input
    def check_args(self, args):

        # check .jpg output_quality
        if args.output_quality < 1 or args.output_quality > 95:
            self.log.error("Quality option --output-quality must be between 1 and 95.  " +
                           "Please try again.")
            raise ValueError("output quality must be between 1 and 95 (see " +
                             "romans.plugins.jpg_converter --help).")

        # check fps is > 0
        if args.video_fps <= 0:
            self.log.error("Video frames per second --video-fps must be > 0.")
            raise ValueError("video fps must be > 0.")

        if args.field_var is None:
            self.log.error('Please specify --field-var.')
            raise ValueError("field variable required to complete request " +
                             "(see romans.plugins.jpg_converter --help).")

    # initialize any local variables from command line arguments
    def init(self, args):

        # save args for later use
        self.args = args

    # make binary version of matrix
    def _binary(self, data):

        data[data < 0] = 0
        data[data > 0] = 1

        return data

    # clip matrix values and scale matrix to [0,1]
    def _scale_matrix(self, point_data):

        # default to scaling to min/max of data
        point_data_min = point_data.min()
        point_data_max = point_data.max()

        # color-scale is provided clip data and use color scale for min/max
        if self.args.color_scale is not None:
            # clip data to min and max of color scale
            point_data[point_data < self.args.color_scale[0]] = self.args.color_scale[0]
            point_data[point_data > self.args.color_scale[1]] = self.args.color_scale[1]

            # use color scale for min and max to scale
            point_data_min = self.args.color_scale[0]
            point_data_max = self.args.color_scale[1]

        # scale data to [0,1] for image creation
        point_data = point_data - point_data_min
        if point_data_min < point_data_max:
            point_data = point_data / (point_data_max - point_data_min)

        return point_data

    # convert matrix to mp4 frame
    def _matrix_to_mp4(self, data):

        # scale matrix to [0,1]
        point_data_out = self._scale_matrix(data)

        # convert to image and remove alpha
        return np.delete(np.uint8(cm.jet(point_data_out) * 255), 3, 2)

    # over-riding convert_files to generate sim.npy and mp4 files
    def convert_files(self, file_list, output_dir, output_type, input_type=None):

        # check for sim.npy or mp4
        if output_type == "sim.npy" or output_type == "mp4":

            # go through file list and load data
            file_data = []
            num_files = len(file_list)
            for file_to_add in file_list:

                point_data_out = imageio.imread(file_to_add)

                # convert to binary, if requested
                if self.args.binary:
                    point_data_out = self._binary(point_data_out)

                # make sure we only have at most one 3D matrix
                matrix_dim = len(point_data_out.shape)

                # add as matrix
                if output_type == "sim.npy":

                    # check that it's 2D matrix:
                    if matrix_dim == 2:
                        file_data.append(point_data_out)

                    # otherwise keep data the same
                    else:
                        file_data = point_data_out

                # add as jet colormap image
                else:

                    # convert each frame for 2D
                    if matrix_dim == 2:

                        # convert to mp4 frame
                        file_data.append(self._matrix_to_mp4(point_data_out))

                    else:

                        # convert all frames for 3D
                        for i in range(point_data_out.shape[0]):
                            file_data.append(self._matrix_to_mp4(point_data_out[i, :, :]))

            # create file name by combining root file names with field variable
            if num_files > 1:

                # get common file prefix
                common_prefix = os.path.basename(os.path.commonprefix(file_list))

                # join to output directory and add field variable
                file_out = os.path.join(output_dir, common_prefix +
                                        self.args.field_var + "." + output_type)

            # if one file, change extension to output type
            else:

                file_name = os.path.basename(file_list[0])

                # check for .sim.npy file (very likely)
                if file_name.endswith('.sim.npy'):
                    file_root, file_ext = file_name.split('.sim.npy')

                # otherwise split normally
                else:
                    file_root, file_ext = os.path.splitext(file_name)

                file_out = os.path.join(output_dir, file_root + "." + output_type)

            # write out as sim.npy
            if output_type == "sim.npy":
                np.save(file_out, np.asarray(file_data))

            # write out as mp4
            else:

                # open a video writer with web browser codecs
                writer = imageio.get_writer(file_out, format="FFMPEG", mode="I",
                                            fps=self.args.video_fps, output_params=['-force_key_frames',
                                                                                    '0.0,0.04,0.08', '-vcodec',
                                                                                    'libx264', '-acodec', 'aac'])

                # write frames to movie
                if not self.args.write_raw_video:
                    for i in range(len(file_data)):
                        writer.append_data(file_data[i])
                else:
                    # add raw files to imageio video writer
                    for f in file_list:
                        writer.append_data(imageio.imread(f))

                writer.close()

            # must return a list of files written
            return [file_out]

        else:

            # revert to 1-1 file conversion
            return super().convert_files(file_list, output_dir, output_type, input_type=input_type)


# if called from the command line display plugin specific command line arguments
if __name__ == "__main__":
    plugin = Plugin()
    plugin.parser.parse_args()
