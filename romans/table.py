# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script contains for code manipulating ensemble .csv files, including 
# creation from input decks, joining multiple tables, and expanding
# a table according to rules provided by plugin.

# S. Martin
# 3/23/2021

# standard library imports
import argparse
import logging
import sys
import os

# 3rd party imports

# local imports
import ensemble
import romans

# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Manipulates .csv files from ensemble data."
    parser = romans.ArgumentParser (description = description)

    # major csv options (must choose one)
    parser.add_argument('--create', action="store_true", help="Create ensemble .csv file "
                        "from simulation input decks.")
    parser.add_argument("--join", nargs="+", help="List of romans .csv files to join horizontally " +
                        "(first column is assumed to be index).")
    parser.add_argument("--concat", nargs="+", help="List of romans .csv files to join vertically " +
                        "(all column headers must be identical).")
    parser.add_argument('--expand', help='Expand links in .csv file to include data in table.  Uses '
                        'plugin to expand links.')

    # create/join csv from ensemble specifier and input files
    parser.add_argument("--ensemble", help="Directory or directories to include in ensemble, "
                        "specified using the Python like %%d[::] notation described above.")
    parser.add_argument("--input-files", help="Files per ensemble directory to use as input "
                        "for metadata, specified using %%d[::] notation.  Note that these "
                        "files will be pre-fixed by the ensemble directories.")
    parser.add_argument("--input-header", help='Name to assign input file header, e.g. "Input Deck"')

    # create specific option
    parser.add_argument("--input-format", help="Format for input files.  Optional, inferred "
                        "from file extension if not provided.")

    # join specific options
    parser.add_argument("--ignore-index", action="store_true", default=False, help="Ignore "
                        "index column when joining tables.")
    parser.add_argument("--convert-cols", nargs="+", help="Converts the given columns using " +
                        "--uri-root-out when joining tables.")
    parser.add_argument("--uri-root-out", help="Root name of URI used to transform file " +
                        "pointers in .csv output file when joining files.  Note that this " +
                        "will only work if the file pointers have a common root.")

    # concat specific options
    parser.add_argument("--add-origin-col", help="Add a column containing the data origin. " +
                        "This flag gives the new column name.")
    parser.add_argument("--origin-col-names", nargs="+", help="Names to use for origin column, " +
                        "one per file to concatenate (defaults to file names).")

    # expand specific options
    parser.add_argument("--expand-header", help="Table column to expand (either name or index).")

    # output file information
    parser.add_argument("--output-dir", help="Output directory for any files produced.")
    parser.add_argument("--csv-out", help="File name of output .csv file.")
    parser.add_argument("--csv-no-index", action="store_false", default=True,
                        help="Do not output the index column.")
    parser.add_argument("--csv-index-header", default=None, help="Index header name for .csv file " +
                        "(default is None).")
    parser.add_argument("--csv-headers", nargs='*', help="Output only the given headers to " +
                        "the .csv file (defaults to all headers).")

    # over-write if file exists
    parser.add_argument("--over-write", action="store_true", help="Over-write output "
                        "file if already present.")

    return parser

# check arguments for create option
def check_create_arguments(log, args):

    # make sure the ensemble argument is present
    if args.ensemble is None:
        log.error("Ensemble directories are required.  Please use --ensemble and try again.")
        sys.exit(1)
    
    # make sure the input argument is present
    if args.input_files is None:
        log.error("Input files are required.  Please use --input-files and try again.")
        sys.exit(1)

    # make sure the input file header is present
    if args.input_header is None:
        log.error("Input header is required.  Please use --input-header and try again.")
        sys.exit(1)

# check arguments for join
def check_join_arguments(log, args):
    
    # if only one csv file, must specify ensemble parameters
    if len(args.join) == 1:

        if args.ensemble is None or \
           args.input_files is None or \
           args.input_header is None:

            log.error("If only using one .csv file, you must specify --ensemble arguments.  " +
                      "Please use --ensemble, --input-files, --input-header and try again. ")
            sys.exit(1)
    
    # if ignoring index column you can't output index
    if args.ignore_index:
        if args.csv_no_index:
            log.error("If --ignore-index is set, you must also set --csv-no-index.")
            sys.exit(1)

    # check that uri-root and convert-cols are both present, or neither
    if (args.convert_cols is None and args.uri_root_out is not None) or \
       (args.convert_cols is not None and args.uri_root_out is None):
       log.error("Must specify both --convert-cols and --uri-root-out.")
       sys.exit(1)

# check arguments for concat
def check_concat_arguments(log, args):

    # if origin names provided, check that there are same number of files
    if args.origin_col_names is not None:
        
        # check that user request an origin column
        if args.add_origin_col is None:
            log.error("Must use --add-origin-col if providing origin column names.")
            sys.exit(1)

        # check for matching origin names and files to concatenate
        if len(args.origin_col_names) != len(args.concat):
            log.error("Number of --origin-col-names does not match number of files to concatenate.")
            sys.exit(1)

# check argumetns for expand
def check_expand_arguments(log, args):

    # check that column to expand is provided
    if args.expand_header is None:
        log.error("Please specify --expand-header and try again.")
        sys.exit(1)

# check command arguments
def check_arguments(log, args):
    
    # convert user selection into True/False is option is selected
    options_selected = [vars(args)["create"]] + [vars(args)[option] is not None 
                        for option in ["join", "concat", "expand"]]

    # check that one option is selected
    if sum(options_selected) == 0:
        log.error("Please select one of --create, --join, --concat, and --expand " +
                  "and try again.")
        sys.exit(1)

    # make sure only one option is selected
    if sum(options_selected) > 1:
        log.error("Select only one of --create, --join, --concat, or --expand and try again.")
        sys.exit(1)

    # make sure the output directory is present
    if args.output_dir is None:
        log.error("Output directory must be specified.  Please use --output-dir and try again.")
        sys.exit(1)

    # make sure the output argument is present
    if args.csv_out is None:
        log.error("Name of .csv output file is required.  Please use --csv-out and try again.")
        sys.exit(1)

    # check create options
    if args.create:
        check_create_arguments(log, args)
    
    # check join options
    if args.join is not None:
        check_join_arguments(log, args)
        
    # check concat option
    if args.concat is not None:
        check_concat_arguments(log, args)

    # check expand option
    if args.expand is not None:
        check_expand_arguments(log, args)

# create .csv file
def create_csv(args, log, plugin):

    # create ensemble table
    ensemble_table = ensemble.Table(log, ensemble_spec=args.ensemble, 
        file_spec=args.input_files, header=args.input_header)
    ensemble_dirs = ensemble_table.get_col(args.input_header)

    # quit if no directories found
    num_ensemble_dirs = len(ensemble_dirs)
    if num_ensemble_dirs == 0:
        log.error("No ensemble directories found.  " +
                  "Please identify existing directories and try again.")
        sys.exit(1)
    else:
        log.info("Found %d ensemble directory(ies)." % num_ensemble_dirs)
        
    # go through each directory and read input files
    input_data = []
    for i in range(num_ensemble_dirs):

        # find files in directory
        files_to_read = ensemble_table.files(ensemble_dirs[i])
        log.info("Found %d file(s) in ensemble directory %s." % 
            (len(files_to_read), ensemble_dirs[i]))
        
        # print warning if no files found for input
        if files_to_read == []:
            log.error("No files to read, please provide existing files for input.")
            sys.exit(1)

        # read input files
        input_data.append(plugin.read_input_deck(files_to_read, file_type=args.input_format))

    # combine all input data headers
    input_headers = []
    for i in range(num_ensemble_dirs):
        for key in input_data[i].keys():
            if key not in input_headers:
                input_headers.append(key)
    
    # create table using input data headers
    input_table = []
    for header in input_headers:

        # create column for a given header
        input_col = []
        for i in range(num_ensemble_dirs):
            if header in input_data[i]:
                input_col.append(input_data[i][header])
            else:

                # use empty strings if no data
                input_col.append('')
        
        # check if we should add this column
        if args.csv_headers is not None:
            if len(args.csv_headers) > 0:
                if header not in args.csv_headers:
                    continue

        # add column to table
        ensemble_table.add_col(input_col, header)
    
    # write out table
    csv_out = os.path.join(args.output_dir, args.csv_out)
    ensemble_table.to_csv(csv_out, index=args.csv_no_index, 
                          index_label=args.csv_index_header, 
                          cols=args.csv_headers)

# join csv files
def join_csv(args, log):

    # create ensemble table for each .csv file
    ensemble_tables = []
    for csv_file in args.join:
        ensemble_tables.append(ensemble.Table(log, csv_file=csv_file))

    # create extra column if user is using --ensemble
    if args.ensemble is not None:
        ensemble_tables.append(ensemble.Table(log, ensemble_spec=args.ensemble, 
            file_spec=args.input_files, header=args.input_header))
    
    # combine tables
    combined_table = ensemble.combine(log, ensemble_tables, ignore_index=args.ignore_index)
    
    # convert file pointers, if requested
    if args.convert_cols is not None:
        combined_table.convert_cols(args.convert_cols, args.uri_root_out)

    # write out combined table
    csv_out = os.path.join(args.output_dir, args.csv_out)
    combined_table.to_csv(csv_out, index=args.csv_no_index, 
                          index_label=args.csv_index_header,
                          cols=args.csv_headers)

# concat csv files
def concat_csv(args, log):

    # create ensemble table for each .csv file
    ensemble_tables = []
    for csv_file in args.concat:
        ensemble_tables.append(ensemble.Table(log, csv_file=csv_file))

    # check that column headers are identical
    headers = list(ensemble_tables[0].table)
    headers_identical = True
    for table in ensemble_tables:
        if list(table.table) != headers:
            headers_identical = False

    # quit if headers are not identical
    if not headers_identical:
        log.error("Table headers are not identical, cannot concatenate tables.")
        sys.exit(1)

    # concatenate tables
    concat_table = ensemble.concat(log, ensemble_tables)

    # create origin column, if requested
    origin_col = []
    for i in range(len(ensemble_tables)):

        # use given origin names if provided
        if args.origin_col_names is not None:
            origin_col += [args.origin_col_names[i]] * ensemble_tables[i].table.shape[0]

        # otherwise, use file names
        else:
            origin_col += [args.concat[i]] * ensemble_tables[i].table.shape[0]

    # add origin column to concatenated table
    concat_table.add_col(origin_col, args.add_origin_col)

    # write out new table
    csv_out = os.path.join(args.output_dir, args.csv_out)
    concat_table.to_csv(csv_out, index=args.csv_no_index, 
                          index_label=args.csv_index_header,
                          cols=args.csv_headers)
    

# expand csv file
def expand_csv(args, log, plugin):

    # read main table
    table_to_expand = ensemble.Table(log, csv_file=args.expand)

    # check if column exists
    col_to_expand = table_to_expand.get_col(args.expand_header)
    
    # get files to expand, count files per specifier
    files_to_expand = []
    missing_files = False
    multiple_files = False
    for file_spec in col_to_expand:

        # get files for specifier
        expand_files = table_to_expand.files(file_spec)

        # do the references files exist?
        if len(expand_files) == 0:
            missing_files = True

        # do the references point to multiple files?
        if len(expand_files) > 1:
            multiple_files = True

        # if only one file, make into a string instead of a list
        if len(expand_files) == 1:
            expand_files = expand_files[0]
        
        files_to_expand.append(expand_files)
    
    # if there are missing files, error out
    if missing_files:
        log.error("Column to expand does not reference existing files.")
        sys.exit(1)

    # for multiple files, expand columns but do not read files
    if multiple_files:

        # explode table
        exploded_table = ensemble.explode(log, table_to_expand, 
            args.expand_header, files_to_expand)

        # write out table
        csv_out = os.path.join(args.output_dir, args.csv_out)
        exploded_table.to_csv(csv_out, index=args.csv_no_index, 
                              index_label=args.csv_index_header,
                              cols=args.csv_headers)

    # otherwise use plugin to read files and expand
    # (plugin can also write additional files, if desired)
    else:

        plugin.expand(table_to_expand, args.expand_header, files_to_expand, 
            output_dir=args.output_dir, csv_out=args.csv_out, csv_no_index=args.csv_no_index, 
            csv_index_header=args.csv_index_header, csv_headers=args.csv_headers)

# creates a .csv file for remaining ensemble tools to use as input
# call from Python using arg_list
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
    log = logging.getLogger("romans.table")
    log.debug("Started table.")

    # check arguments
    check_arguments(log, args)
    
    # import and initialize plugin
    plugin, unknown_args = romans.plugin(args.plugin, arg_list)

    # check for un-recognized arguments
    if len(unknown_args) > 0:
        log.error("Unrecognized arguments: %s.  Please try again." % str(unknown_args))
        sys.exit(1)
    
    # check if output directory exists
    if not os.path.exists(args.output_dir):
        log.warning("Output directory does not exist -- creating directory: " + 
                    args.output_dir)
        os.makedirs(args.output_dir)

    # check if output file exists
    csv_out = os.path.join(args.output_dir, args.csv_out)
    if os.path.isfile(csv_out):
        if not args.over_write:
            log.error("Output file already exists, use --over-write if you " +
                "want to over-write file.")
            sys.exit(1)

    # create csv
    if args.create:
        create_csv(args, log, plugin)
    
    # join csv
    elif args.join is not None:
        join_csv(args, log)

    # concatenate csv
    elif args.concat is not None:
        concat_csv(args, log)

    # expand csv
    elif args.expand is not None:
        expand_csv(args, log, plugin)

# entry point for command line call
if __name__ == "__main__":
    main()
