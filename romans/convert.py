# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This utility converts ensemble files from one format to
# another, for example .vtk to .npy or .jpg to .mp4.

# S. Martin
# 11/20/2020

# standard library imports

# command line arguments
import argparse

# logging and error handling
import logging
import sys

# miscellaneous calculations
import math

# 3rd party imports

# parallel comutation
from ipyparallel import Client

# local imports
import romans
import ensemble

# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Converts ensemble file formats.  Uses Python-like %d[::] notation, " + \
                  "where %d[::] specifies a range of numbers in a file name. For example " + \
                  '"time_step_%d[0:10:2].vtp" would specify every other file from ' + \
                  '"time_step_0.vtp" to "time_step_9.vtp".'
                  
    parser = romans.ArgumentParser (description=description)

    # get files to convert from command line
    parser.add_argument("--ensemble", help="Directory or directories to include in ensemble, "
                        "specified using the Python like %%d[::] notation described above.")
    parser.add_argument("--input-files", help="Files per ensemble directory to use as input "
                        "for conversion, specified using %%d[::] notation.  Note that these "
                        "files will be pre-fixed by the ensemble directories.")

    # or get files to convert from .csv file
    parser.add_argument("--csv-file", help="CSV file which specifies ensemble directories "
                        "and input files (alternate to using --ensemble and --input-files).")
    parser.add_argument("--csv-col", help="Column in CSV file where input files are "
                        "specified, can be either a string or an integer (1-based).")
    
    # input options
    parser.add_argument("--input-format", help="Format for input files (optional, inferred "
                        "from file extension if not provided).")

    # output arguments and options
    parser.add_argument("--output-dir", help="Directory to place output.  All files will be "
                        "stored using directories that mirror those specified by --ensemble.")
    parser.add_argument("--output-format", help="File output format information, "
                        "e.g. file extension.")
    parser.add_argument("--over-write", action="store_true", help="Over-write output "
                        "directory if already present.")

    # output a csv file
    parser.add_argument("--csv-out", help="File name of output .csv file with file links for "
                        "converted files (optional).  Will be written to output directory.")
    parser.add_argument("--csv-header", help="Name of output files header, needed only "
                        "if writing out a .csv file.")

    # parallel option using ipyparallel
    parser.add_argument('--parallel', default=False, action="store_true", 
                        help="Use ipyparallel (must be available and running).")

    return parser

# check arguments
def check_arguments(log, args):

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

    # make sure the output directory is present
    if args.output_dir is None:
        log.error("Output directory must be specified.  Please use --output-dir and try again.")
        sys.exit(1)
    
    # make sure the output format is specified
    if args.output_format is None:
        log.error("Output format is required.  Please use --output-format and try again.")
        sys.exit(1)

    # if csv is to be written out, must also have column header
    if args.csv_out is not None:
        if args.csv_header is None:
            log.error("CSV header is required to output .csv file.  " +
                      "Please use --csv-header and try again.")
            sys.exit(1)

# convert ensemble, either in serial or parallel
def convert_ensemble (plugin, log ,args, ensemble_table, ensemble_dirs, mirror_dirs):

    num_ensemble_dirs = len(ensemble_dirs)

    if not args.parallel:

        # go through each directory and convert files
        files_written = []
        for i in range(num_ensemble_dirs):

            # convert a single simulation
            files_to_convert, files_created, files_converted = convert_simulation(plugin, args,
                ensemble_table, ensemble_dirs[i], mirror_dirs[i])

            # log progress
            progress_report(log, files_to_convert, ensemble_dirs[i], files_created)

            # keep track of files written in csv specifier format
            files_written.append(files_converted)

        return files_written
    
    # parallel operation, using direct view
    rc = Client()

    # use cloudpickle because pickle won't work with plugins
    rc[:].use_cloudpickle()

    # get number of available engines
    num_engines = len(rc)

    # break files into blocks so that we use all available engine
    block_size = int(math.ceil(num_ensemble_dirs/num_engines))

    # push out parallel jobs, one load per engine
    files_written = []
    for i in range(0, num_ensemble_dirs, num_engines):

        # get block of simuilations
        block_dirs = [ensemble_dirs[j] 
            for j in range(i, i + num_engines) if j < num_ensemble_dirs]
        mirror_block = [mirror_dirs[j] 
            for j in range(i, i + num_engines) if j < num_ensemble_dirs]

        # push out jobs per engine
        async_results = []
        for j in range(len(block_dirs)):
            async_result = rc[j].apply_async(convert_simulation, 
                plugin, args, ensemble_table, block_dirs[j], mirror_block[j])
            async_results.append(async_result)

        # wait for results
        rc[:].wait(async_results)

        # get results
        for j in range(len(async_results)):

            # get items in order they were put into queue
            async_result = async_results.pop(0)

            # get returned values
            files_to_convert = async_result.get()[0]
            files_created = async_result.get()[1]
            files_converted = async_result.get()[2]

            # log progress
            progress_report(log, files_to_convert, block_dirs[j], files_created)

            # save csv results
            files_written.append(files_converted)

        # clean up ipyparallel
        rc.purge_everything()

    # clean up sockets
    rc.close()
    
    return files_written

# convert a single simulation (can be run in parallel)
def convert_simulation (plugin, args, ensemble_table, ensemble_dir, mirror_dir):

    # find files in directory
    files_to_convert = ensemble_table.files(ensemble_dir)

    # no files, then return empty
    if files_to_convert == []:
        return [], []

    # convert files
    files_created = plugin.convert_files(files_to_convert, mirror_dir, 
        args.output_format, input_type=args.input_format)

    # if only one file created, use the file name
    if len(files_created) == 1:
        files_written = files_created[0]

    # convert input file specifier to output file specifier
    else:
        files_written = ensemble_table.convert_specifier(ensemble_dir,
            args.output_dir, args.output_format)

    return files_to_convert, files_created, files_written

# progress report to log for conversion
def progress_report(log, files_to_convert, ensemble_dir, files_created):

    # print error if no files were converted
    if files_to_convert == []:
        log.error("No files to convert, please provide existing files for input.")
        sys.exit(1)

    # report on files converted
    log.info("Found %d file(s) in ensemble directory %s." % 
        (len(files_to_convert), ensemble_dir))
    for j in range(len(files_created)):
        log.info('File written "%s".' % files_created[j])

# converts ensemble files from one format to another
# call programmatically using arg_list
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
    log = logging.getLogger("romans.convert")
    log.debug("Started convert.")

    # check arguments
    check_arguments(log, args)

    # import and initialize plugin
    plugin, unknown_args = romans.plugin(args.plugin, arg_list)

    # check for un-recognized arguments
    if len(unknown_args) > 0:
        log.error("Unrecognized arguments: %s.  Please try again." % str(unknown_args))
        sys.exit(1)

    # create ensemble table
    ensemble_table = ensemble.Table(log, csv_file=args.csv_file,
        ensemble_spec=args.ensemble, file_spec=args.input_files, 
        header="convert")

    # get ensemble directories/file specs
    if args.csv_file is not None:
        ensemble_dirs = ensemble_table.get_col(args.csv_col)
    else:
        ensemble_dirs = ensemble_table.get_col("convert")

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

    # convert files, either in serial or parallel
    files_written = convert_ensemble(plugin, log, args, ensemble_table, 
        ensemble_dirs, mirror_dirs)

    # are we writing out a csv file?
    if args.csv_out is not None:

        # add new column to ensemble table
        ensemble_table.add_col(files_written, args.csv_header)

        # put .csv file in output directory
        ensemble_table.to_csv(args.csv_out, output_dir=args.output_dir, 
            cols=[args.csv_header])

# entry point for command line call
if __name__ == "__main__":
    main()