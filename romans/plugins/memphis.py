# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains the MEMPHIS 2D phase-field specific code needed to 
# run the romans tools.

# S. Martin
# 11/9/2020


# standard library imports
import argparse
import os

# 3rd party imports
import numpy as np

# for exporting images
from PIL import Image
from matplotlib import cm

# for writing mp4
import imageio

# for downsampling images
import cv2

# for testing mesh type
import meshio

# materials knowledge system
import pymks

# turn off deprecation warnings (for pymks)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# local imports
import romans
import romans.plugins.memphis_vtk_patch


# defaults for user arguments
JPG_QUALITY = 95
VIDEO_FPS = 25

# functions for MEMPHIS specific operations
# class must be named Plugin
class Plugin(romans.PluginTemplate):

    # initialize MEMPHIS command line argument parser
    def __init__(self):

        # describe MEMPHIS plugin
        description = "The MEMPHIS plugin provides support for the MEMPHIS " + \
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

        self.parser.add_argument("--auto-correlate", action="store_true", help="Performs "
            "auto-correlation as a pre-processing for dimension reduction (note this option "
            "requires the --binary flag to be used).")

        self.parser.add_argument("--color-scale", nargs=2, type=float, help="Gives the color "
            "scale for the field variable when creating jpg or mp4, e.g. "
            '"--color-scale 0 1" for min color value of 0 and max color value of 1. '
            "Note that values above and below the color scale are automatically clipped.")

        self.parser.add_argument("--output-format", help="The output format options "
            'recognized by the MEMPHIS plugin include: "npy" -- saves the '
            'field variable for a single timestep to a numpy array; "sim.npy" -- '
            'saves the field variable for every timestep in a simulation to a 3D '
            'numpy array; "rd.npy" -- saves the reduced dimensional representation '
            'to a numpy array (can be either a time step matrix or a 3D full simulation '
            'matrix); "jpg" -- saves a .jpg image of the field variable for a single '
            'timestep; "mp4" -- saves a .mp4 movie of the field variable for every timestep '
            'in a simulation.')
        
        self.parser.add_argument("--sample", type=float, help="Sample image by given "
            "percent (if < 100 it is downsampling, if > 100 it is upsampling).")

        self.parser.add_argument("--output-quality", type=int, default=JPG_QUALITY, 
            help="Quality of jpg image, as a number between 1 and 95 (only relevant "
            "if outputing images, defaults to %s)." % str(JPG_QUALITY))
        
        self.parser.add_argument('--output-color', action="store_true", default=False,
            help="Output images as color (otherwise they are output as grayscale).")
        
        self.parser.add_argument("--video-fps", type=float, default=VIDEO_FPS, 
            help="Number of frames per second for video creation, defaults to %s)." % 
            str(VIDEO_FPS))

    # check arguments, error if incorrect input
    def check_args(self, args):

        # check .jpg output_quality
        if args.output_quality < 1 or args.output_quality > 95:
            self.log.error("Quality option --output-quality must be between 1 and 95.  " +
                           "Please try again.")
            raise ValueError("output quality must be between 1 and 95 (see " +
                             "romans.plugins.memphis --help).")

        # check fps is > 0
        if args.video_fps <= 0:
            self.log.error("Video frames per second --video-fps must be > 0.")
            raise ValueError("video fps must be > 0.")

        # check sample is > 0
        if args.sample is not None:
            if args.sample <= 0:
                self.log.error('Sampling percent must be > 0.  Please try again.')
                raise ValueError("sampling percent must be > 0.")

        # check that binary is active if auto-correlate is selected
        if args.auto_correlate:
            if not args.binary:
                self.log.error("Auto-correlation requires binary input, please use --binary " +
                               "and try again.")
                raise ValueError("auto-correlation requires binary flag (see " +
                                 "romans.plugins.memphis --help.")

    # check for field variable
    def _check_field_var(self):

        if self.args.field_var is None:
            self.log.error('Please specify --field-var.')
            raise ValueError("field variable required to complete request " +
                             "(see romans.plugins.memphis --help).")

    # initialize any local variables from command line arguments
    def init(self, args):

        # save args for later use
        self.args = args

    # read point data into dictionary with numpy arrays from a 
    # meshio rectilinear mesh, returns None if mesh cannot be 
    # verified as rectilinear
    def _rectilinear_point_data(self, mesh):
        
        # check that z-axis is identically zero
        if np.any(mesh.points[:,2]):
            return None

        # get unique x,y values
        x_unique = np.unique(mesh.points[:,0])
        y_unique = np.unique(mesh.points[:,1])

        # construct meshgrid and compare
        x_mesh, y_mesh = np.meshgrid(x_unique, y_unique)
        xy_grid = np.concatenate((np.reshape(x_mesh, (len(x_mesh)*len(x_unique), 1)), 
                   np.reshape(y_mesh, (len(y_mesh)*len(y_unique), 1))), axis=1)

        # is constructed mesh grid the same?
        if not np.array_equal(mesh.points[:,0:2], xy_grid):
            return None

        # form point data into matrix, using row-order
        point_data = {}
        for key in mesh.point_data.keys():
            point_data[key] = np.reshape(mesh.point_data[key], 
                (len(x_unique), len(y_unique)), order='C')

        return point_data

    # check that point data is rectilinear
    def _check_point_data(self, point_data):

        # if data was read as None, then there was an error
        if point_data is None:
            self.log.error("Couldn''t find rectilinear mesh in data.")
            raise ValueError("not recilinear mesh data.")

        # check that field variable is present
        keys = list(point_data.keys())
        if self.args.field_var not in keys:
            self.log.error('Field variable "%s" not found in data.' % self.args.field_var)
            raise ValueError("unrecognized field variable.")

    # make binary version of matrix
    def _binary(self, data):

        data[data < 0] = 0
        data[data > 0] = 1

        return data

    # get data for a single field variable
    def _get_field_data(self, data):

        # get point data on rectilinear mesh
        point_data = self._rectilinear_point_data(data)

        # check that point data was rectilinear
        self._check_point_data(point_data)

        # get point data to output
        point_data_out = point_data[self.args.field_var]
                
        return point_data_out
            
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
            point_data = point_data/(point_data_max - point_data_min)
        
        return point_data, point_data_min, point_data_max

    # reverse matrix scaling from [0,1] to [data_min, data_max]
    def _unscale_matrix(self, data, data_min, data_max):

        # rescale data to range
        if data_min < data_max:
            data = data * (data_max - data_min)

        # move 0 to data_min
        data = data + data_min

        return data

    # down/up sample matrix (treated as image)
    def _sample_matrix(self, data):

        # scale data to [0,1]
        scaled_data, data_min, data_max = self._scale_matrix(data)

        # convert to [0,255] pixel value image
        img = np.array(scaled_data * 255, dtype = np.uint8)

        # compute new width/height according to scale provided
        width = int(img.shape[1] * self.args.sample / 100)
        height = int(img.shape[0] * self.args.sample / 100)
        dim = (width, height)

        # down/up sample image
        sampled_data = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

        # convert back to float
        sampled_data = np.array(sampled_data, dtype=np.float) / 255

        # return to original scale
        data = self._unscale_matrix(sampled_data, data_min, data_max)

        return data
        
    # read npy and sim.npy (also npy) formats
    def read_file(self, file_in, file_type=None):

        # check file extension, if not provided
        if file_type is None:

            # npy file type
            if file_in.endswith('.npy'):
                file_type = 'npy'

        # check if we have npy or sim.npy
        if file_type == 'npy':
            
            # read npy file
            try:
                data = np.load(file_in)
            except ValueError:
                self.log.error("Could not read " + file_in + " as a .npy file.")
                raise ValueError("Could not read " + file_in + " as a .npy file.")

        # otherwise default to mesh
        else:
            data = super().read_file(file_in, file_type)

        return data

    # write out .npy format or .jpg
    def write_file(self, data, file_out, file_type=None):

        # check for npy output format with color scale
        if file_type=='npy':
            if self.args.color_scale is not None:
                self.log.error("Can't use --color-scale with .npy output.")
                raise ValueError("can't use --color-scale with .npy output.")

        # infer file type, if not provided
        if file_type is None:

            if file_out.endswith('.npy'):
                file_type = 'npy'

            if file_out.endswith('.jpg'):
                file_type = 'jpg'

        # check for npy or jpg file output (data is matrix format)
        if file_type in ['npy', 'jpg']:

            # check if data is meshio and convert
            data = self._convert_meshio(data)

            # convert to binary, if requested
            if self.args.binary:
                data = self._binary(data)
            
            # downsample/upsample, if requested
            if self.args.sample is not None:
                data = self._sample_matrix(data)

            # write out field variable as npy or jpg
            if file_type == "npy":

                # save as numpy
                np.save(file_out, data)
            
            # for jpg we need to scale data and use a colormap
            elif file_type == "jpg":

                # convert to [0,1]
                data, _, _ = self._scale_matrix(data)
                
                # convert to standard jet color map image
                if self.args.output_color:
                    img = Image.fromarray(np.uint8(cm.jet(data) * 255))

                # or black and white image
                else: 
                    img = Image.fromarray(np.uint8(data * 255))

                # save image
                img.convert("RGB").save(file_out, quality=self.args.output_quality)

        else:

            # default to meshio output
            super().write_file(data, file_out, file_type=file_type)

    # check for backwards conversions (matrix to mesh)
    def convert_file(self, file_in, file_out, file_in_type=None, file_out_type=None):

        # can't convert .npy or .sim.npy to .vtk
        if file_in.endswith("npy") and file_out_type=='vtk':
            self.log.error("Can't convert .npy to .vtk format.")
            raise TypeError("Can't convert .npy to .vtk format.")

        super().convert_file(file_in, file_out, 
            file_in_type=file_in_type, file_out_type=file_out_type)

    # convert matrix to mp4 frame
    def _matrix_to_mp4(self, data):

        # scale matrix to [0,1]
        point_data_out, _, _ = self._scale_matrix(data)

        # output color or black/white
        if self.args.output_color:

            # convert to image and remove alpha
            return np.delete(np.uint8(cm.jet(point_data_out) * 255), 3, 2)
        
        else:
            return np.uint8(point_data_out * 255)

    # over-riding convert_files to generate sim.npy and mp4 files
    def convert_files(self, file_list, output_dir, output_type, input_type=None):
        
        # check for npy output format with color scale
        if output_type=='sim.npy':
            if self.args.color_scale is not None:
                self.log.error("Can't use --color-scale with .sim.npy output.")
                raise ValueError("can't use --color-scale with .sim.npy output.")

        # check for sim.npy or mp4
        if output_type == "sim.npy" or output_type == "mp4":
            
            # go through file list and load data
            file_data = []
            num_files = len(file_list)
            for file_to_add in file_list:

                # load data
                data = self.read_file(file_to_add, file_type=input_type)
                
                # convert meshio to matrix
                point_data_out = self._convert_meshio(data)
                
                # convert to binary, if requested
                if self.args.binary:
                    point_data_out = self._binary(point_data_out)

                # downsample/upsample, if requested
                if self.args.sample is not None:
                    point_data_out = self._sample_matrix(point_data_out)

                # make sure we only have at most one 3D matrix
                matrix_dim = len(point_data_out.shape)
                if matrix_dim == 3:
                    if num_files > 1:
                        self.log.error("Cannnot combine .sim.npy files in conversion.")
                        raise TypeError("Can't combine .sim.npy files in conversion.")

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
                            file_data.append(self._matrix_to_mp4(point_data_out[i,:,:]))

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
                    '0.0,0.04,0.08', '-vcodec', 'libx264', '-acodec', 'aac'])

                # write frames to movie
                for i in range(len(file_data)):
                    writer.append_data(file_data[i])
                writer.close()

            # must return a list of files written
            return [file_out]

        else:

            # revert to 1-1 file conversion
            return super().convert_files(file_list, output_dir, output_type, input_type=input_type)

    # read a MEMPHIS input file
    def read_input_deck (self, file_list, file_type=None):

        # MEMPHIS input files are all the same format and
        # don't have an extension so file_type is ignored

        # we are expecting one file, so error if multiple files
        if len(file_list) > 1:
            raise ValueError("Provided multiple files for MEMPHIS input deck, expecting only one.")

        # return is dictionary where the keys are the parameter name
        # and values are the corresponding value
        inputs = {}

        # open file and read each line
        with open(file_list[0]) as fp: 
            lines = fp.readlines()

            # go through each line and create dictionary of data
            for line in lines:

                # ignore comments (lines starting with '!')
                if line[0] == '!':
                    continue
                
                # ignore blank lines
                if not line.strip():
                    continue

                # get tokens on line
                tokens = line.split()

                # check that we have at least one non-empty token
                if len(tokens) == 0:
                    continue
                
                # token with no argument
                elif len(tokens) == 1:
                    args = ['']
                
                else:
                    args = tokens[1:]
                            
                # if the are multiple args, create keywords
                # with appendices -1, -2, -3, etc.
                keywords = []
                if len(args) == 1:
                    keywords.append(tokens[0])

                else:
                    for i in range(len(args)): 
                        keywords.append(tokens[0] + "-" + str(i+1))
                
                # check if keywords are already present in dictionary
                for i in range(len(keywords)):
                    if keywords[i] in inputs:
                        raise ValueError('Duplicate keys (first column entries) found in "%s"' %
                            file_list[0])
                    else:

                        # add key, value pairs to dictionary
                        inputs[keywords[i]] = args[i]

        return inputs
    
    # convert from meshio to numpy, 
    def _convert_meshio (self, data):

        if isinstance(data, meshio.Mesh):

            # check for a field variable
            self._check_field_var()

            # convert to matrix (checks for binary)
            data = self._get_field_data(data)
        
        return data

    # perform pre-processing on mesh or npy files
    def preprocess (self, data, flatten=True):
        
        # if mesh, convert to matrix using field variable
        data = self._convert_meshio(data)

        # convert to binary, if requested
        if self.args.binary:
            data = self._binary(data)

        # do auto-correlation, if requested
        if self.args.auto_correlate:
            
            # set up hat basis for pymks
            p_basis = pymks.bases.PrimitiveBasis(n_states=2)

            # check if this data is 2D
            if len(data.shape) == 2:

                # keep rows and columns
                num_rows, num_cols = data.shape

                # add extra dimension for 2D
                data = np.expand_dims(data, axis=0)
            
            # do auto-correlation with pymks (use binary data)
            space_stats = pymks.stats.correlate(data, p_basis, 
                                                periodic_axes=(0, 1), 
                                                correlations=[(0, 0)])

            # re-shape each time step into vector
            space_stats_vec = np.reshape(space_stats,[space_stats.shape[0],
                                         space_stats.shape[1] * space_stats.shape[2],
                                         space_stats.shape[-1]])
            data_vec = np.squeeze(space_stats_vec)

            # re-shape to 2D if we don't want flattened data
            if not flatten:

                data_vec = np.reshape(data_vec, (num_rows, num_cols))
        
        else:

            # default behaviour is to reshape matrix into one simulation per row
            if flatten:

                if len(data.shape) == 2:
                    data_vec = np.reshape(data, [data.shape[0] * data.shape[1]])

                else:
                    data_vec = np.reshape(data, [data.shape[0], 
                                        data.shape[1] * data.shape[2]])

            # otherwise, keep as matrix
            else:
                data_vec = data
                
        # make sure there are no singleton dimensions
        data_vec = np.squeeze(data_vec)
        
        return data_vec


# if called from the command line display MEMPHIS specific command line arguments
if __name__ == "__main__":

    memphis = Plugin()
    memphis.parser.parse_args()
