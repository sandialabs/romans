# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This file provides a command line interface for training
# and testing a reduced order proxy model given input from
# a dimension reduction.

# The interface for model is the same interface used for
# convert/reduce, but instead of converting/reducing files, 
# a proxy model is trained or tested

# S. Martin
# 6/10/2021

# standard library

# command line arguments
import argparse

# logging and error handling
import logging
import sys

# file manipulations
import os

# 3rd party libraries
import numpy as np

# local imports
import romans
import ensemble
import algorithms.proxy as algorithms

# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Trains/tests reduced order proxy model for numerical simulations.  Uses " + \
                  "Python-like %d[::] notation, where %d[::] specifies a range of numbers in " + \
                  'a directory name. For example "workdir.%d[0:10:2]" would specify every ' + \
                  'other file from "workdir.0" to "workdir.9".  Input files are expected to ' + \
                  "have the extension .rd.npy as output by the dimension reduction code " + \
                  "reduce.py.  The model is output as a .pkl file."
                  
    parser = romans.ArgumentParser (description=description)

    # get files to convert from command line
    parser.add_argument("--ensemble", help="Directory or directories to include in ensemble, "
                        "specified using the Python like %%d[::] notation described above.")
    parser.add_argument("--input-file", help="Files per ensemble directory to use as input "
                        "for a model.  The input files are expected to have the same name and "
                        "end with extension .rd.npy.")

    # or get files to convert from .csv file
    parser.add_argument("--csv-file", help="CSV file which specifies ensemble directories "
                        "and input files (alternate to using --ensemble and --input-files).")
    parser.add_argument("--csv-col", help="Column in CSV file where input files are "
                        "specified, can be either a string or an integer (1-based).")

    # input model from file
    parser.add_argument("--input-model", help="Input proxy model from .pkl file "
                        "(only with when testing).")

    # output arguments and options
    parser.add_argument("--output-model", help="Output dimension reduction model to provided "
                        "file (in ensemlbe directory).")
    parser.add_argument("--output-file", help="File name for output of test data, the same name is "
                        "used for each simulation (only for testing).  Files are written to ensemble "
                        "directory.")
    parser.add_argument("--over-write", action="store_true", help="Over-write output "
                        "if already present.")

    # train/test flags
    parser.add_argument("--train", help="Train proxy model using input files with time steps "
                        "specified using %%d[::] notation.")
    parser.add_argument("--test", nargs=2, help="Test proxy model using " + \
                        "--input-model, with %%d[::] to specify time steps to use " + \
                        "followed by number of future time steps to predict.")

    # number dimensions to use for proxy model
    parser.add_argument("--num-dim", type=int, help="Number of dimensions to use "
                        "for the proxy model.")
    # batch option
    parser.add_argument("--file-batch-size", type=int, help="Train proxy model "
                        "incrementally using batches of files.  Not available for "
                        "all algorithms, see romans.algorithms.proxy --help for options.")

    # parallel option using ipyparallel
    parser.add_argument('--parallel', default=False, action="store_true", 
                        help="Use ipyparallel (must be available and running).")

    return parser

# check arguments, return parse arguments for train, test
def check_arguments(log, args, algorithm=None):

    # train and test default to None
    train = None
    test = None

    # check for command line input arguments
    if args.csv_file is None:

        # make sure the ensemble argument is present
        if args.ensemble is None:
            log.error("Ensemble directories are required.  Please use --ensemble and try again.")
            sys.exit(1)
        
        # make sure the input argument is present
        if args.input_file is None:
            log.error("Input file is required.  Please use --input-file and try again.")
            sys.exit(1)

        # check extension of input_files
        elif not args.input_file.endswith('.rd.npy'):
            log.error('Input file name must have extension ".rd.npy".')
            sys.exit(1)

        # disable args.csv_col (if provided)
        args.csv_col = None

    # otherwise check that ensemble arguments are not present
    else:

        if args.ensemble is not None or args.input_file is not None:
            log.error("Please specify either --csv-file or --ensemble, but not both.")
            sys.exit(1)

        # also check that csv column is provided
        if args.csv_col is None:
            log.error("Please specify --csv-col to provide input files.")
            sys.exit(1)

    # must be in either training mode or testing mode
    if ((args.train is not None) and (args.test is not None)) or \
       ((args.train is None) and (args.test is None)):
        log.error("Proxy model code requires either --train or --test but not both.")
        sys.exit(1)

    # make sure output model file is present for training
    if args.train is not None:

        # specify number of dimensions to use for proxy model
        if args.num_dim is not None:
            if args.num_dim < 1:
                log.error("Must use at least one dimension for proxy model.")
                sys.exit(1)

        # make sure args.train argument is valid
        root, start, stop, step, extension = \
            ensemble.parse_d_format(log, args.train)
        
        if args.train == root:
            log.error("Please use --train %d[::] to specify time steps to use for " + \
                      "training.  Use %d for all time steps.")
            sys.exit(1)

        # change args.train to hold start, stop, step values
        train = [start, stop, step]

        if args.output_model is None:
            log.error("Output model file must be specified for training.  " + \
                      "Please use --output-model and try again.")
            sys.exit(1)

        # make sure output model has .pkl extension
        elif not args.output_model.endswith(".pkl"):
            log.error("Output model file must have .pkl extension.")
            sys.exit(1)

    # testing requirements
    if args.test is not None:

        # make sure first args.test argument is valid
        root, start, stop, step, extension = \
            ensemble.parse_d_format(log, args.test[0])

        if args.train == root:
            log.error('Please use "--test %d[::] future" to specify time steps to use for ' + \
                      "training.  Use %d for all time steps.")
            sys.exit(1)

        if int(args.test[1]) < 1:
            log.error("Test must predict into the future by at least one time step.")
            sys.exit(1)

        # set test arguments to [root, start, stop, future]
        test = [start, stop, step, int(args.test[1])]

        # need input model file
        if args.input_model is None:
            log.error("Input model file must be specified for testing.  " + \
                      "Please use --input-model and try again.")
            sys.exit(1)

        # make sure input model ends with .pkl
        elif not args.input_model.endswith(".pkl"):
            log.error("Input model file must have .pkl extension.")
            sys.exit(1)

        # need output files
        if args.output_file is None:
            log.error("Output file must be specified for testing.  " + \
                      "Plese use --output-file and try again.")
            sys.exit(1)

        # make sure output file ends with ".px.npy."
        elif not args.output_file.endswith(".px.npy"):
            log.error("Output file name must end with .px.npy.  ")
            sys.exit(1)

    # batch calculation checks
    if args.file_batch_size is not None:

        # check if batch size is non-negative
        if args.file_batch_size <= 0:
            log.error("Batch size must be an integer >= 1.")
            sys.exit(1)

    # are there unknown arguments?
    if algorithm is not None:
        if len(algorithm.unknown_args()) > 0:
            log.error("Unrecognized arguments: %s.  Please try again." % 
                str(algorithm.unknown_args()))
            sys.exit(1)

    return train, test

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

    # construct output file names/check sim files, if there are mirror directories
    output_files = None
    if mirror_dirs is not None:

        output_files = []
        for i in range(len(ensemble_dirs)):

            # print error if no files found for conversion
            if len(sim_files[i]) == 0:
                log.error("No files to test, please provide existing files for input.")
                sys.exit(1)

            # output file name for each ensemble
            output_files.append(os.path.join(mirror_dirs[i], args.output_file))

    # check that time counts are uniform
    if min(file_counts) < max(file_counts):
        log.error("Simulations have different number of time step files.  Proxy model " + \
                  "training/testing not performed, no files written.")
        sys.exit(1)

    return sim_files, output_files

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

# read in a batch of files and return a list of matrices, one per simulation
# checks that matrices are the same size
def get_batch (log, plugin, algorithm, batch_files, time_steps, parallel):

    # get batch of data, keep as list of matrices
    data = plugin.read_file_batch(batch_files, parallel=parallel, flatten=False)

    # make sure matrices are all the same size
    num_time_steps, num_components = np.asarray(data[0]).shape
    for i in range(len(batch_files)):
            
        # data has already been loaded
        log.info("Read file %s." % batch_files[i])

        # make sure data is numpy array
        data[i] = np.asarray(data[i])

        # halt if matrices change size
        if data[i].shape[0] != num_time_steps or data[i].shape[1] != num_components:
            log.error("Reduced simulation data has inconsistent matrix sizes.  Aborting " + \
                     "proxy model training, no files written.")
            sys.exit(1)

    # get indices of training data with specified time steps
    train_inds = np.arange(time_steps[0], time_steps[1], time_steps[2])

    # reduced training data
    data_time_steps = [data[i][train_inds,:] for i in range(len(data))]

    return data_time_steps

# trains proxy model using ensemble files
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
    log = logging.getLogger("romans.model")
    log.debug("Started model.")

    # check arguments
    train, test = check_arguments(log, args)

    # import and initialize plugin
    plugin, arg_list = romans.plugin(args.plugin, arg_list)

    # initialize proxy algorithm 
    algorithm = algorithms.ProxyModel(arg_list=arg_list,
                                      model_file=args.input_model)

    # re-check arguments with algorithm initialized
    train, test = check_arguments(log, args, algorithm=algorithm)

    # create ensemble table
    ensemble_table = ensemble.Table(log, csv_file=args.csv_file,
        ensemble_spec=args.ensemble, file_spec=args.input_file, 
        header="proxy")

    # get ensemble directories/file specs
    if args.csv_file is not None:
        ensemble_dirs = ensemble_table.get_col(args.csv_col)
    else:
        ensemble_dirs = ensemble_table.get_col("proxy")

    # quit if no directories found
    num_ensemble_dirs = len(ensemble_dirs)
    if num_ensemble_dirs == 0:
        log.error("No ensemble directories found.  " +
                  "Please identify existing directories and try again.")
        sys.exit(1)
    else:
        log.info("Found %d ensemble directory(ies)." % num_ensemble_dirs)

    # get common path, use as output directory
    if len(ensemble_dirs) > 1:
        common_path = os.path.commonpath(ensemble_dirs)

    # if only one simulation, commom path is everything up to file specifier
    else:
        common_path = os.path.dirname(ensemble_dirs[0])

    output_dir = common_path

    # for testing create mirror directories
    mirror_dirs = None
    if test is not None:

        # mirror directories to output direoctory
        mirror_dirs = ensemble_table.mirror_directories(
            output_dir, ensemble_dirs, args.over_write)
        
        # if directory already exists, quit
        if mirror_dirs is None:
            log.error("Output directory already exists, use --over-write if " +
                "you want to over-write or add to previous contents.")
            sys.exit(1)
        log.info("Output directory created or already exists with mirrored file structure.")
            
    # get reduced data simulation files
    sim_files, output_files = get_sim_files (log, args, 
            ensemble_dirs, mirror_dirs, ensemble_table)

    # determine batches
    batch_files = get_batch_files (sim_files, args.file_batch_size)
    num_batches = len(batch_files)

    # if there are not files, quit
    if batch_files == [[]]:
        log.error("No files found.  Please supply diredtories with existing files " +
                  "and try again.")
        sys.exit(1)

    # training
    if train is not None:

        # train in batches, if given
        for i in range(num_batches):

            # load files into memory, store as a list of matrices
            # with num_time_steps x num_components, one per simulation
            data_to_train = get_batch(log, plugin, algorithm,
                batch_files[i], train, args.parallel)

            # reduce dimension, if requested
            if args.num_dim is not None:
                num_dim = data_to_train[0].shape[1]
                data_to_train = [data_to_train[i][:,0:min(args.num_dim, num_dim)]
                    for i in range(len(data_to_train))]

            # train proxy model, if not loaded
            if args.input_model is None:
                log.info("Training reduced order proxy model, batch " + str(i) + ".")
                algorithm.train(data_to_train)

        # save model
        if args.output_model is not None:
            algorithm.save(os.path.join(output_dir, args.output_model))
        
    # testing
    if test is not None:

        # test in batches, if given
        for i in range(num_batches):

            # load files into memory, store as a list of matrices
            # with num_time_steps x num_components, one per simulation
            data_to_test = get_batch(log, plugin, algorithm,
                batch_files[i], test[0:3], args.parallel)

            # test model
            log.info("Testing reduced order proxy model, batch " + str(i) + ".")
            predictions = algorithm.test(data_to_test, test[3])

        # save predictions
        num_sim = len(predictions)
        for i in range(num_sim):
            
            # write out predicted data per simulation        
            plugin.write_file(predictions[i], output_files[i])
            log.info("Saved file %s." % output_files[i])

# entry point for command line call
if __name__ == "__main__":
    main()
