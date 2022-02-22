# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script tests the convert.py using the phase-field test
# dataset.  Tests are done on the user interface to check
# the options and on the MEMPHIS test data.  The memphis plugin is
# used so we are also checking if the memphis conversion routines 
# work.
#
# WARNING: When this script is run, the output_dir is
# deleted, unless delete_output_dir is False.
#
# Different tests can be run by setting the below flags.

# S. Martin
# 3/19/2021

# file/path manipulation
import os
import shutil

# arguments
import argparse

# romans convert utility
import romans.convert as convert

# file paths

# directory containing the phase-field test dataset
test_data_dir = 'data/phase-field/test_data'

# dimension reduction test directory
reduce_dir = 'data/phase-field/test_reduce'

# output directory to use for testing
output_dir = 'data/phase-field/test_convert'

# sampled directory
sampled_dir = 'data/phase-field/test_sampled_data'

# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Generate various test .csv files from phase-field ensemble data."
    parser = argparse.ArgumentParser (description = description)

    # delete output directory (defaults to False)
    parser.add_argument('--delete-output-dir', action="store_true", default=False, 
        help="Delete output directory before starting tests.")

    # available tests
    parser.add_argument('--test-UI', action="store_true", default=False,
        help="Test command line UI for convert.py.")
    parser.add_argument('--test-conversions', action="store_true", default=False,
        help="Test conversions.")
    parser.add_argument('--test-end-state', action="store_true", default=False,
        help="Do end-state conversions (e.g. images and movies).")
    parser.add_argument('--test-parallel', action="store_true", default=False,
        help="Run parallel tests using ipyparallel (must have ipengine running).")
    parser.add_argument('--test-sample', action="store_true", default=False,
        help="Test image downsampling.")
    parser.add_argument('--test-all', action="store_true", default=False,
        help="Run every test.")

    return parser

# delete output_dir, if present
def delete_output_dir(args):

    if args.delete_output_dir:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)

# argument parser testing
#########################

def test_UI(args):

    if args.test_UI or args.test_all:

        # no arguments
        arg_list = []
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed no argument check.\n")

        # ensemble only
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d')]
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --ensemble only check.\n")

        # ensemble and input files
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_0.vtk']
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --ensemble and --input-files check.\n")

        # ensemble, input files, and output directory
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_0.vtk',
                    '--output-dir', os.path.join(output_dir, 'workdir.1')]
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --ensemble, --input-files, and --output-dir check.\n")

        # ensemble, input files, output directory, and output format
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_0.vtk',
                    '--output-dir', os.path.join(output_dir, 'workdir.1'),
                    '--output-format', 'vtk']
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --over-write check.\n")

        # check for unrecognized argument
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_0.vtk',
                    '--output-dir', os.path.join(output_dir, 'workdir.1'),
                    '--output-format', 'npy',
                    '--foo', 'bar']
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --foo bar check.\n")

        # check for --csv file conflict
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_0.vtk',
                    '--output-dir', os.path.join(output_dir, 'workdir.1'),
                    '--output-format', 'npy',
                    '--csv-file', 'foo-bar.csv']
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --csv-file check.\n")

        # check --csv-col when using --csv-file
        arg_list = ['--csv-file', os.path.join(test_data_dir, 'metadata.csv')]
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --csv-col check.\n")

        # check --csv-header
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_0.vtk',
                    '--output-dir', os.path.join(output_dir, 'workdir.1'),
                    '--output-format', 'npy',
                    '--csv-out', 'foo-bar.csv']
        try:
            convert.main(arg_list)
        except SystemExit:
            print("Passed --csv-header check.\n")

# file conversion testing
#########################

def test_conversions(args):

    if args.test_conversions or args.test_all:

        # copy three vtk to vtk and create csv with links
        print("Converting .vtk files in workdir.%d[0:4] to .vtk files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[0:4]'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-format', 'vtk',
                    '--over-write',
                    '--csv-out', 'three-vtk.csv',
                    '--csv-header', 'Three VTK']
        convert.main(arg_list)

        # test csv as input, convert .vtk to .npy
        print("Converting .vtk to .npy ...")
        arg_list = ['--csv-file', os.path.join(output_dir, 'three-vtk.csv'),
                    '--csv-col', 'Three VTK',
                    '--output-dir', output_dir,
                    '--output-format', 'npy',
                    '--over-write',
                    '--field-var', 'phase_field']
        convert.main(arg_list)

        # test .npy to .npy
        print("Converting .npy to .npy ...")
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d[0:4]'),
                    '--input-files', 'out.cahn_hilliard_50000000.npy',
                    '--output-dir', output_dir,
                    '--output-format', 'npy',
                    '--over-write']
        convert.main(arg_list)

        # test .npy to .vtk (should error)
        print("Trying .npy to .vtk ...")
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d[0:4]'),
                    '--input-files', 'out.cahn_hilliard_50000000.npy',
                    '--output-dir', output_dir,
                    '--output-format', 'vtk',
                    '--over-write']
        try:
            convert.main(arg_list)
        except TypeError:
            print("Passed backwards conversion test.")

        # create a few sample .npy files 
        print("Converting first three .vtk files in workdir.1 to .npy files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.1'),
                    '--input-files', 'out.cahn_hilliard_%d[0:1500000].vtk',
                    '--output-dir', os.path.join(output_dir, 'workdir.1'),
                    '--output-format', 'npy',
                    '--over-write',
                    '--field-var', 'phase_field']
        convert.main(arg_list)

        # create a few sample .sim.npy files
        print("Creating first three .sim.npy files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[1:4]'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-format', 'sim.npy',
                    '--over-write',
                    '--field-var', 'phase_field']
        convert.main(arg_list)

        # convert .sim.npy to .sim.npy
        print("Convert .sim.npy to .sim.npy ...")
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d[1:4]'),
                    '--input-files', 'out.cahn_hilliard_phase_field.sim.npy',
                    '--output-dir', output_dir,
                    '--output-format', 'sim.npy',
                    '--over-write']
        convert.main(arg_list)

        # convert .npy to .jpg
        print("Converting .npy to .jpg ...")
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d[1:4]'),
                    '--input-files', 'out.cahn_hilliard_50000000.npy',
                    '--output-dir', output_dir,
                    '--output-format', 'jpg',
                    '--over-write']
        convert.main(arg_list)

        # convert .sim.npy to .mp4
        print("Converting .sim.npy to .mp4 ...")
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d[1:4]'),
                    '--input-files', 'out.cahn_hilliard_phase_field.sim.npy',
                    '--output-dir', output_dir,
                    '--output-format', 'mp4',
                    '--over-write']
        convert.main(arg_list)

# create end state images and movies
#################################### 

def test_end_state(args):

    if args.test_end_state or args.test_all:

        # create end state images
        print("Creating End State image files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-format', 'jpg',
                    '--over-write',
                    '--field-var', 'phase_field',
                    '--color-scale', '0', '1',
                    '--csv-out', 'end-state.csv',
                    '--csv-header', 'End State']
        convert.main(arg_list)

        # create simulation movies
        print("Creating simulation movie files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-format', 'mp4',
                    '--over-write',
                    '--field-var', 'phase_field',
                    '--color-scale', '0', '1',
                    '--csv-out', 'movies.csv',
                    '--csv-header', 'Movie']
        convert.main(arg_list)

# parallel testing
##################

def test_parallel(args):

    if args.test_parallel or args.test_all:

        # create end state images in parallel
        print("Creating End State image files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-format', 'jpg',
                    '--over-write',
                    '--field-var', 'phase_field',
                    '--color-scale', '0', '1',
                    '--csv-out', 'end-state.csv',
                    '--csv-header', 'End State',
                    '--parallel']
        convert.main(arg_list)

        # create simulation movies
        print("Creating simulation movie files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-format', 'mp4',
                    '--over-write',
                    '--field-var', 'phase_field',
                    '--color-scale', '0', '1',
                    '--csv-out', 'movies.csv',
                    '--csv-header', 'Movie',
                    '--parallel']
        convert.main(arg_list)

# test sampling
###############

def test_sample(args):

    if args.test_sample or args.test_all:

        print("Creating Downsampled End State image files ...")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', sampled_dir,
                    '--output-format', 'sim.npy',
                    '--sample', '10',
                    '--over-write',
                    '--field-var', 'phase_field']
        convert.main(arg_list)

if __name__ == "__main__":

    # get command line flags
    parser = init_parser()
    args = parser.parse_args()

    # delete output directory, if requested
    delete_output_dir(args)

    # run tests
    test_UI(args)
    test_conversions(args)
    test_end_state(args)
    test_parallel(args)
    test_sample(args)