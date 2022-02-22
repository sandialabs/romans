# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This file provides a command line interface for performing
# dimension reduction as the first step in computing a proxy
# model for acclerating numerical simulation.

# The interface for reduce is the same interface used for
# convert, but instead of converting files, a dimension
# reduction is performed.

# S. Martin
# 1/29/2021


# standard library imports

# command line arguments
import argparse

# logging and error handling
import logging
import sys

# output
import os

# restart file
import pickle

# 3rd party imports
import numpy as np

# local imports
import romans
import ensemble
import algorithms.reduction as algorithms


# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Performs dimemsion reduction on ensemble data.  Uses Python-like %d[::] " + \
                  "notation, where %d[::] specifies a range of numbers in a file name. For " + \
                  'example "time_step_%d[0:10:2].vtp" would specify every other file from ' + \
                  '"time_step_0.vtp" to "time_step_9.vtp".  If individual time steps are ' + \
                  'provided as input, the results are combined into a single matrix and ' + \
                  'output.  The output file extension is .rd.npy.'
                  
    parser = romans.ArgumentParser (description=description)

    # get files to convert from command line
    parser.add_argument("--ensemble", help="Directory or directories to include in ensemble, "
                        "specified using the Python like %%d[::] notation described above.")
    parser.add_argument("--input-files", help="Files per ensemble directory to use as input "
                        "for reduction, specified using %%d[::] notation.  Note that these "
                        "files will be pre-fixed by the ensemble directories.")

    # or get files to convert from .csv file
    parser.add_argument("--csv-file", help="CSV file which specifies ensemble directories "
                        "and input files (alternate to using --ensemble and --input-files).")
    parser.add_argument("--csv-col", help="Column in CSV file where input files are "
                        "specified, can be either a string or an integer (1-based).")
    
    # input type (optional)
    parser.add_argument("--input-format", help="Format for input files (optional, inferred "
                        "from file extension if not provided).")

    # input model from file
    parser.add_argument("--input-model", help="Input dimension reduction model from .pkl file "
                        "(do not train a new model).")

    # output arguments and options
    parser.add_argument("--output-dir", help="Directory to place output.  All files will be "
                        "stored using directories that mirror those specified by --ensemble.")
    parser.add_argument("--output-file", help="File name for reduced data, the same name is "
                        "used for each simulation.")
    parser.add_argument("--output-model", help="Output dimension reduction model to provided "
                        "file (in output directory).")
    parser.add_argument("--over-write", action="store_true", help="Over-write output "
                        "directory if already present.")

    # output a csv file
    parser.add_argument("--csv-out", help="File name of output .csv file with file links for "
                        "reduced files (optional).  Will be written to output directory.")
    parser.add_argument("--csv-header", help="Name of output files header, needed only "
                        "if writing out a .csv file.")

    # output xy coordinate to csv file
    parser.add_argument("--xy-out", help="File name of output .csv file with the (x,y) " 
                        "coordinates (optional).  Will be written to output directory.")
    parser.add_argument("--xy-header", help="Root name of header for (x,y) coordinates " 
                        "columns in .csv file.")

    # batch option
    parser.add_argument("--file-batch-size", type=int, help="Train reduction model "
                        "incrementally using batches of files.  Not available for "
                        "all algorithms, see romans.algorithms.reduction --help for options.")

    # parallel option using ipyparallel
    parser.add_argument('--parallel', default=False, action="store_true", 
                        help="Use ipyparallel (must be available and running).")
    
    # restart file for parallel crashes
    parser.add_argument('--restart', help="File name to save intermediate results and then restart "
                        "from a crash (must also specify --output-model).")

    return parser

# check arguments
def check_arguments(log, args, algorithm=None):

    # check for command line input arguments
    if args.csv_file is None:

        # make sure the ensemble argument is present
        if args.ensemble is None:
            log.error("Ensemble directories are required.  Please use --ensemble and try again.")
            sys.exit(1)
        
        # make sure the input argument is present
        if args.input_files is None:
            log.error("Input files are required.  Please use --input-files and try again.")
            sys.exit(1)

        # disable args.csv_col (if provided)
        args.csv_col = None

    # otherwise check that ensemble arguments are not present
    else:

        if args.ensemble is not None or args.input_files is not None:
            log.error("Please specify either --csv-file or --ensemble, but not both.")
            sys.exit(1)

        # also check that csv column is provided
        if args.csv_col is None:
            log.error("Please specify --csv-col to provide input files.")
            sys.exit(1)

    # make sure input model file has .pkl extension
    if args.input_model is not None:
        if not args.input_model.endswith(".pkl"):
            log.error("Input model file must have .pkl extension.")
            sys.exit(1)

    # make sure the output directory is present
    if args.output_dir is None:
        log.error("Output directory must be specified.  Please use --output-dir and try again.")
        sys.exit(1)
    
    # make sure output file is specified
    if args.output_file is None:
        log.error("Output file name must be provided.  Please use --output-file and try again.")
        sys.exit(1)

    # check extension, if provided
    elif not args.output_file.endswith('.rd.npy'):
        log.error('Output file name must have extension ".rd.npy".')
        sys.exit(1)

    # make sure output model file has .pkl extension
    if args.output_model is not None:
        if not args.output_model.endswith(".pkl"):
            log.error("Output model file must have .pkl extension.")
            sys.exit(1)

    # if csv is to be written out, must also have column header
    if args.csv_out is not None:
        if args.csv_header is None:
            log.error("CSV header is required to output .csv file.  " +
                      "Please use --csv-header and try again.")
            sys.exit(1)

    # batch calculation checks
    if args.file_batch_size is not None:

        # check if batch size is non-negative
        if args.file_batch_size <= 0:
            log.error("Batch size must be an integer >= 1.")
            sys.exit(1)

        # check if algorithm is incremental
        if algorithm is not None:
            if not algorithm.is_incremental():
                if algorithm.time_align_dim() is None:
                    log.error("Dimension reduction algorithm selected is not incremental " +
                              "and can't be used in batch mode.  Please select a different " +
                              "algorithm and try again.")
                    sys.exit(1)

    # is restart file a .pkl file
    if args.restart is not None:

        # restart file must be .pkl file
        if not args.restart.endswith(".pkl"):
            log.error("Restart file must have .pkl extension.  Please use a different file " +
                    "name and try again.")
            sys.exit(1)

        # must also specify output model
        if args.output_model is None:
            log.error("Output model file (using --output-model) must be specified in addition to restart file.")
            sys.exit(1)

    # are there unknown arguments?
    if algorithm is not None:
        if len(algorithm.unknown_args()) > 0:
            log.error("Unrecognized arguments: %s.  Please try again." % 
                str(algorithm.unknown_args()))
            sys.exit(1)

# get simulation file list
def get_sim_files (log, args, ensemble_dirs, mirror_dirs, ensemble_table):

    # get simulations files in parallel if requested
    sim_files = ensemble_table.ensemble_files(ensemble_dirs, args.parallel) 

    # count number of files per ensemble
    file_counts = []
    for i in range(len(ensemble_dirs)):

        # count files
        file_counts.append(len(sim_files[i]))

        # report on files found
        log.info("Found %d file(s) in ensemble directory %s." % 
            (len(sim_files[i]), ensemble_dirs[i]))

    # construct output file names/check sim files
    output_files = []
    for i in range(len(ensemble_dirs)):

        # print error if no files found for conversion
        if len(sim_files[i]) == 0:
            log.error("No files to reduce, please provide existing files for input.")
            sys.exit(1)

        # output file name for each ensemble
        output_files.append(os.path.join(mirror_dirs[i], args.output_file))

    # check that time counts are uniform
    if min(file_counts) < max(file_counts):
        log.error("Simulations have different number of time step files.  Dimension " + \
                  "reduction not performed, no files written.")
        sys.exit(1)

    return sim_files, output_files

# get files for a single time step
def get_time_files (sim_files, time_step, file_batch_size):

    # sim_files is organized as a [sim][time]
    # go through all simulations and pick out desired time step files
    time_files = []
    num_sim = len(sim_files)
    for sim in range(num_sim):
        time_files.append(sim_files[sim][time_step])

    # split time files into batches
    return split_batch_files(time_files, file_batch_size)

# split a list of files into batches
def split_batch_files (batch_files, file_batch_size):

    # split into array of batch sizes
    batch_inds = list(range(0, len(batch_files), file_batch_size))
    if batch_inds[-1] < len(batch_files):
        batch_inds += [len(batch_files)]
    
    # get files by batch
    batches = []
    for i in range(len(batch_inds)-1):
        batches.append(batch_files[batch_inds[i]:batch_inds[i+1]])
    
    return batches

# get a batch of file names
def get_batch_files (sim_files, file_batch_size):

    # concatenate all sim files, in order
    batch_files = [file for sim in sim_files for file in sim]

    # return all files if no batches
    if file_batch_size is None:
        return [batch_files]
    
    # otherwise split into array of batch sizes
    else:
        return split_batch_files(batch_files, file_batch_size)

# read in a batch of files and pre-process, 
# return number of time points per file
def get_batch (log, plugin, batch_files, file_type, 
    parallel, flatten):

    # get batch of data
    data = plugin.read_file_batch(batch_files, file_type=file_type, 
        parallel=parallel, flatten=flatten)

    # pre-process and count time points
    time_counts = []
    for i in range(len(batch_files)):
            
        # data has already been loaded
        log.info("Read file %s." % batch_files[i])

        # make sure data is numpy array
        data[i] = np.asarray(data[i])
        
        # keep track of time points per file
        if len(data[i].shape) == 1:
            time_counts.append(1)
        else:
            time_counts.append(data[i].shape[0])

    # check that time points per file are identical
    num_time = max(time_counts)
    if min(time_counts) < num_time:
        log.error("Simulation time steps per file are different.  Dimension " + \
                    "reduction not performed, no files written.")
        sys.exit(1)

    # convert to 2D matrix for dimension reduction
    data_to_reduce = np.asarray(data)
    if num_time > 1:
        data_to_reduce = np.vstack(data)

    return data_to_reduce, num_time

# compute time indices into array of stacked simulation data
def compute_time_inds(num_time_steps, num_sim, time_step):

    # gather simulation data per time step
    time_inds = []
    for j in range(num_sim):
        time_inds.append(j * num_time_steps + time_step)

    return time_inds

# train a single time step, using algorithm.fit
def train_time_step(log, args, algorithm, data, time_step):
    
    # train dimension reduction model, if not loaded
    if args.input_model is None:
        log.info("Training dimension reduction model for time step %d." % time_step)
        algorithm.fit(data, time_step=time_step)

    # project data
    log.info("Projecting data to lower dimension for time step %d." % time_step)
    time_reduced_data = algorithm.transform(data, time_step=time_step)

    return time_reduced_data

# project data in batches
def project_data_batches(log, args, plugin, algorithm, batch_files, 
    time_step=0, flatten=True):

    # project data in batches
    num_batches = len(batch_files)
    reduced_data = []
    for i in range(num_batches):

        # read batch files into matrix
        data_to_reduce, num_time = get_batch(log, plugin,
            batch_files[i], args.input_format, args.parallel, flatten)

        # collect reduced data
        log.info("Projecting data to lower dimension.")
        reduced_data.append(algorithm.transform(data_to_reduce, time_step=time_step))
    
    # make each batch into one matrix
    reduced_data = np.vstack(reduced_data)

    return reduced_data, num_time

# convert from time aligned list of matrices to 
# all simulations concatenated matrix
def convert_time_sim(time_aligned_data, num_time_steps, num_sim, num_dim):

    reduced_data = np.zeros((num_time_steps * num_sim, num_dim))
    for i in range(num_time_steps):

        # gather simulation data per time step
        time_inds = compute_time_inds(num_time_steps, num_sim, i)

        # distrubte aligned data into matrix
        reduced_data[time_inds,:] = time_aligned_data[i]

    return reduced_data

# performs dimension reduction on ensemble
def main(arg_list=None):

    # initialize parser
    parser = init_parser()

    # parse arguments
    if arg_list is not None:

        # parse in case called with arg_list
        args, arg_list = parser.parse_args(arg_list)

    else:

        # command line parser
        args, arg_list = parser.parse_args()

    # start logger
    romans.init_logger(log_file=args.log_file, log_level=args.log_level)
    log = logging.getLogger("romans.reduce")
    log.debug("Started reduce.")

    # check arguments
    check_arguments(log, args)

    # import and initialize plugin
    plugin, arg_list = romans.plugin(args.plugin, arg_list)

    # initialize dimension reduction algorithm 
    algorithm = algorithms.DimensionReduction(model_file=args.input_model, 
                                              arg_list=arg_list)

    # re-check arguments with algorithm initialized
    check_arguments(log, args, algorithm=algorithm)
    
    # create ensemble table
    ensemble_table = ensemble.Table(log, csv_file=args.csv_file,
        ensemble_spec=args.ensemble, file_spec=args.input_files, 
        header="reduce")

    # get ensemble directories/file specs
    if args.csv_file is not None:
        ensemble_dirs = ensemble_table.get_col(args.csv_col)
    else:
        ensemble_dirs = ensemble_table.get_col("reduce")

    # quit if no directories found
    num_ensemble_dirs = len(ensemble_dirs)
    if num_ensemble_dirs == 0:
        log.error("No ensemble directories found.  " +
                  "Please identify existing directories and try again.")
        sys.exit(1)
    else:
        log.info("Found %d ensemble directory(ies)." % num_ensemble_dirs)

    # mirror directories to output direoctory
    mirror_dirs = ensemble_table.mirror_directories(
        args.output_dir, ensemble_dirs, args.over_write)
    
    # if directory already exists, quit
    if mirror_dirs is None:
        log.error("Output directory already exists, use --over-write if " +
            "you want to over-write or add to previous contents.")
        sys.exit(1)
    log.info("Output directory created or already exists with mirrored file structure.")

    # check for restart file option
    restart_file_loaded = False
    if args.restart is not None:

        # check for existence of restart file
        restart_file = os.path.join(args.output_dir, args.restart)
        if os.path.exists(restart_file):

            # get sim files and start index from restart file
            with open(restart_file, 'rb') as handle:
                restart_time, restart_batch, sim_files, output_files, time_reduced_data = \
                    pickle.load(handle)
            log.info("Loaded restart data from %s." % restart_file)
            log.info("Restarting at time-aligned step %d, batch %d. " % (restart_time, restart_batch))

            # get model from output model
            model_file = os.path.join(args.output_dir, args.output_model)
            algorithm = algorithms.DimensionReduction(model_file=model_file)

            restart_file_loaded = True

    # if restart file was not loaded, recompute and start from 0
    if not restart_file_loaded:

        # get simulation and output files
        sim_files, output_files = get_sim_files (log, args, 
            ensemble_dirs, mirror_dirs, ensemble_table)

        # set restart to 0
        restart_time = 0
        restart_batch = 0

        # time-reduced data for time-aligned case
        time_reduced_data = []
    
    # count up simulations and time steps
    num_sim = len(sim_files)
    num_files = len(sim_files[0])

    # determine batches
    batch_files = get_batch_files (sim_files, args.file_batch_size)
    num_batches = len(batch_files)

    # check for rd.npy file type
    rd_npy_type = True
    for i in range(num_batches):
        for j in range(len(batch_files[i])):
            if not batch_files[i][j].endswith('.rd.npy'):
                rd_npy_type = False
    
    # if rd.npy file type, do not flatten data
    flatten = True
    if rd_npy_type:
        flatten = False

    # each reduction option has to compute reduced_data
    # ordered same as sim_files, but flattened, and num_time
    # number of time steps per file

    # do reduction differently for different cases
    if algorithm.time_align_dim() is None:
        if args.file_batch_size is None:

            # 1st case is bulk reduction in memory
            
            # read all files into matrix
            data_to_reduce, num_time = get_batch(log, plugin,
                batch_files[0], args.input_format, args.parallel, flatten)

            # train dimension reduction model, if not loaded
            if args.input_model is None:
                log.info("Training dimension reduction model.")
                algorithm.fit(data_to_reduce)

            # reduce data
            log.info("Projecting data to lower dimension.")
            reduced_data = algorithm.transform(data_to_reduce)

        else:

            # 2nd case is bulk in batches

            # train in batches (unless model has been loaded)
            if args.input_model is None:
                for i in range(restart_batch, num_batches):

                    # read batch files into matrix
                    data_to_reduce, num_time = get_batch(log, plugin,
                        batch_files[i], args.input_format, args.parallel, flatten)

                    # incremental model training
                    log.info("Training dimension reduction model, batch %d." % i)
                    algorithm.partial_fit(data_to_reduce)

                    # if restart file, save partial results
                    if args.restart is not None:

                        # save model file
                        model_file = os.path.join(args.output_dir, args.output_model)
                        algorithm.save(model_file)

                        # save restart file
                        with open(restart_file, 'wb') as handle:
                            pickle.dump([0, i+1, sim_files, output_files, []], handle)
                        log.info("Saved restart file: %s." % restart_file)

            # compute reduced data, also in batches
            reduced_data, num_time = project_data_batches(log, args, plugin, algorithm, batch_files,
                flatten=flatten)

    # time aligned cases
    else:

        if args.file_batch_size is None:
            
            # 3rd case is time aligned in memory
            
            # read all files into matrix
            data_to_reduce, num_time = get_batch(log, plugin,
                batch_files[0], args.input_format, args.parallel, flatten)

            # perform dimension reduction per time step
            num_time_steps = num_time * num_files
            for i in range(restart_time, num_time_steps):
                
                # gather simulation data per time step
                time_inds = compute_time_inds (num_time_steps, num_sim, i)

                # train dimension reduction model, if not loaded
                time_reduced_data.append(train_time_step(log, args, 
                    algorithm, data_to_reduce[time_inds,:], i))
                
                # if restart file, save partial results
                if args.restart is not None:

                    # save model file
                    model_file = os.path.join(args.output_dir, args.output_model)
                    algorithm.save(model_file)

                    # save restart file
                    with open(restart_file, 'wb') as handle:
                        pickle.dump([i, 0, sim_files, output_files, time_reduced_data], handle)
                    log.info("Saved restart file: %s." % restart_file)

            # time align
            log.info("Time aligning reduced data.")
            time_aligned_data = algorithm.time_align(time_reduced_data, 
                                    compute_rotations=(args.input_model is None))

            # put back into matrix form
            reduced_data = convert_time_sim(time_aligned_data, num_time_steps, 
                num_sim, algorithm.num_dim())

        else:

            # 4th case is time aligned in batches

            # proceed in batches per time step
            for i in range(restart_time, num_files):

                # get time step files 
                time_files = get_time_files(sim_files, i, args.file_batch_size)

                # check that algorithm is incremental if more than one batch
                num_batches = len(time_files)
                if num_batches > 1 and not algorithm.is_incremental():
                    log.error("There are too many simulation files per time step for the given " +
                              "file batch size.  Please use an inremental algorithm or a larger " +
                              "file batch size.")
                    sys.exit(1)

                # train reduction in batches
                for j in range(num_batches):

                    # load batch of data
                    data_to_reduce, num_time = get_batch(log, plugin,
                        time_files[j], args.input_format, args.parallel, flatten)
                    
                    # for a single batch, use algorithm.fit
                    if num_batches == 1:
                        time_reduced_data.append(train_time_step(log, args, 
                            algorithm, data_to_reduce, i))

                    # for multiple batches, use algorthm.partial_fit,
                    # if model is not already loaded
                    elif args.input_model is None:
                        log.info("Training dimension reduction model, batch " + str(j) +", " +
                                 "time step " + str(i) + ".")
                        algorithm.partial_fit(data_to_reduce, time_step=i)

                # project data, also in batches
                if num_batches > 1:
                    proj_data, _ = project_data_batches(log, args, 
                        plugin, algorithm, time_files, time_step=i, flatten=flatten)
                    time_reduced_data.append(proj_data)
                
                # if restart file, save partial results
                if args.restart is not None:

                    # save model file
                    model_file = os.path.join(args.output_dir, args.output_model)
                    algorithm.save(model_file)

                    # save restart file
                    with open(restart_file, 'wb') as handle:
                        pickle.dump([i, 0, sim_files, output_files, time_reduced_data], handle)
                    log.info("Saved restart file: %s." % restart_file)

            # time align
            log.info("Time aligning reduced data.")
            time_aligned_data = algorithm.time_align(time_reduced_data, 
                                    compute_rotations=(args.input_model is None))

            # put back into matrix form
            num_time_steps = num_time * num_files
            reduced_data = convert_time_sim(time_aligned_data, num_time_steps, 
                num_sim, algorithm.num_dim())

    # save data
    for i in range(num_sim):

        # each file is a matrix (or vector)
        reduced_data_j = []
        for j in range(num_files):

            # get index range in data per file
            start_index = (i * num_files + j * num_time) * num_time
            end_index = start_index + num_time

            # collectreduced data representation per file
            reduced_data_j.append(reduced_data[start_index:end_index,:])
        
        # write out reduced data per simulation        
        plugin.write_file(np.vstack(reduced_data_j), output_files[i])
        log.info("Saved file %s." % output_files[i])
        
    # save links
    if args.csv_out is not None:

        # add new column to ensemble table
        ensemble_table.add_col(output_files, args.csv_header)

        # put .csv file in output directory
        ensemble_table.to_csv(args.csv_out, output_dir=args.output_dir, 
            cols=[args.csv_header])

    # save xy coords, if requested (and possible)
    if args.xy_out is not None:

        # check for one time step per simulation
        if num_time * num_files > 1:
            log.warning("More than one time step per simulation, could not output (x,y) "
                        "coordinates.")
        
        # check for both x and y coordinates
        elif reduced_data.shape[1] < 2:
            log.warning("Less than two coordinates in reduction, could not output (x,y) " 
                        "coordinates.")
        
        # output coordinates
        else:

            # get (x,y) coords
            x_coords = np.squeeze(np.asarray(reduced_data))[:,0]
            y_coords = np.squeeze(np.asarray(reduced_data))[:,1]

            # construct x, y headers
            x_header = "X" if args.xy_header is None else args.xy_header + " X"
            y_header = "Y" if args.xy_header is None else args.xy_header + " Y"

            # add new columns to ensemble table
            ensemble_table.add_col(x_coords, x_header)
            ensemble_table.add_col(y_coords, y_header)

            # put .csv file in output directory
            ensemble_table.to_csv(args.xy_out, output_dir=args.output_dir,
                cols=[x_header, y_header])

    # save model
    if args.output_model is not None:
        algorithm.save(os.path.join(args.output_dir, args.output_model))
    

# entry point for command line call
if __name__ == "__main__":
    main()