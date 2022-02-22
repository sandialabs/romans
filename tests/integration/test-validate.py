# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script tests the validate.py script using the phase-field
# reduced dimension test data and proxy model results.  Tests 
# are done on the UI and on the data.
#
# NOTE: Different tests can be performed depending on the
# test_* flags set below.

# S. Martin
# 6/14/2021

# file/path manipulation
import os
import shutil

# flags for tests to run
import argparse

# reduction command line code
import romans.validate as validate

# original data directory
test_data = 'data/phase-field/test_data'

# output directory to use for testing
test_reduce_dir = 'data/phase-field/test_reduce'

# inc-PCA from cluster
inc_PCA = 'data/phase-field/inc-PCA'

# inc-whiten-PCA from cluster
inc_whiten_PCA = 'data/phase-field/inc-whiten-PCA'

# set up argument parser
def init_parser():

    # define test flags
    description = "Run various tests on the validate.py script."
    parser = argparse.ArgumentParser (description = description)

    # delete output directory (defaults to False)
    parser.add_argument('--delete-output-dir', action="store_true", default=False, 
        help="Delete output directory before starting tests.")

    # available tests
    parser.add_argument('--test-UI', action="store_true", default=False,
        help="Test command line UI for validate.py.")
    parser.add_argument('--test-reduction', action="store_true", default=False,
        help="Test reduction validation.")
    parser.add_argument('--test-proxy', action="store_true", default=False,
        help="Test proxy validation.")
        
    # test everything
    parser.add_argument('--test-all', action="store_true", default=False,
        help="Run every test.")

    return parser

# argument parser testing
#########################

def test_UI(args):

    if args.test_UI or args.test_all:

        # no arguments
        arg_list = []
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed no argument check.\n")

        # check both --reduction and --proxy
        arg_list = ['--reduction', '--proxy']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed not both --reduction and --proxy check.\n")

        # check no --input-model
        arg_list = ['--proxy']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed missing --input-model check.\n")

        # check no --output-file
        arg_list = ['--proxy',
                    '--input-model', 'LSTM-model.pkl']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed missing --output-file check.\n")

        # check --output-file wrong extension
        arg_list = ['--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --ensemble directory check.\n")

        # check --input-file
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --input-file check.\n")

        # check --input-file extension
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_10',
                    '--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --input-file extension check.\n")   

        # check --input-pred-file 
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_10.rd.npy',
                    '--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --input-pred-file check.\n")   

        # check --input-pred-file extension
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_10.rd.npy',
                    '--input-pred-file', 'LSTM-preds',
                    '--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --input-pred-file extension check.\n")

        # check --test argument
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_10.rd.npy',
                    '--input-pred-file', 'LSTM-preds.px.npy',
                    '--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --test check.\n")    

        # check --test future argument
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_10.rd.npy',
                    '--input-pred-file', 'LSTM-preds.px.npy',
                    '--test', '%d[10:90]', '0',
                    '--proxy',
                    '--input-model', 'LSTM-model.pkl',
                    '--output-file', 'LSTM-loss']
        try:
            validate.main(arg_list)
        except SystemExit:
            print("Passed --test future check.\n") 

# validate dimension reduction testing
######################################

def test_reduction(args):

    if args.test_reduction or args.test_all:

        # incremental PCA with 100 reduced dimensions
        arg_list = ['--ensemble', os.path.join(test_data, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_%d.vtk',
                    '--output-file', os.path.join(test_reduce_dir, 'inc-whiten-pca-1000-stats'),
                    '--reduction',
                    '--input-model', os.path.join(test_reduce_dir, 'inc-whiten-PCA-1000.pkl'),
                    '--field-var', 'phase_field']
        validate.main(arg_list)

# validate proxy testing
########################

def test_proxy(args):

    if args.test_proxy or args.test_all:

        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_100.rd.npy',
                    '--output-file', os.path.join(test_reduce_dir, 'LSTM-predictons'),
                    '--input-pred-file', 'LSTM-preds.px.npy',
                    '--proxy',
                    '--input-model', os.path.join(test_reduce_dir, 'LSTM-model.pkl'),
                    '--test', '%d[10:90]', '11']
        validate.main(arg_list)

# command line entry
if __name__ == "__main__":

    # get command line flags
    parser = init_parser()
    args = parser.parse_args()

    # run tests
    test_UI(args)
    test_reduction(args)
    test_proxy(args)