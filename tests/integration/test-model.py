# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script tests the model.py script using the phase-field
# reduced dimension test data.  Tests are done on the UI and 
# on the reduced data.  The reduced order proxy algorithms, 
# to a lesser degree, are also tested.
#
# NOTE: Different tests can be performed depending on the
# test_* flags set below.

# S. Martin
# 3/22/2021

# file/path manipulation
import os

# flags for tests to run
import argparse

# reduction command line code
import romans.model as model

# directory containing the reduced dimensiuon phase-field test dataset
test_reduce_dir = 'data/phase-field/test_reduce'

# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Run various tests on the model.py script."
    parser = argparse.ArgumentParser (description = description)

    # available tests
    parser.add_argument('--test-UI', action="store_true", default=False,
        help="Test command line UI for reduce.py.")
    parser.add_argument('--test-LSTM', action="store_true", default=False,
        help="Test LSTM model using reduced dimension test data.")

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
            model.main(arg_list)
        except SystemExit:
            print("Passed no argument check.\n")

        # missing --input-files
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d')]
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --input-file check.\n")

        # --input-files must have .rd.npy extension
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --input-file .rd.npy check.\n")

        # requires either --train or --test
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --train or --test check.\n")
        
        # requires either --train or --test but not both
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d',
                    '--test', '%d', '1']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed not both --train and --test check.\n")

        # check --num-dim >= 1
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d',
                    '--num-dim', '0']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --num-dim >= 1 check.\n")

        # check --train with invalid %d
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', 'hello']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --train with invalid %d check.\n")

        # check --test with invalid number
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--test', '%d', '-1']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --test with invalid argument check.\n")
                        
        # --output-model must be given for training
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --output-model check.\n")

        # --output-model must have .pkl extension for training
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d',
                    '--output-model', 'LSTM-model']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --output-model .pkl extension check.\n")

        # --input-model for testing
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--test', '%d', '1']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --input-model for testing check.\n") 

        # --input-model for testing must have .pkl extension
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--test', '%d', '1',
                    '--input-model', 'model']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --input-model must have .pkl extension check.\n") 

        # --output-file for testing
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--test', '%d', '1',
                    '--input-model', 'model.pkl']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed --output-file for testing check.\n") 

        # test missing algorithm
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d',
                    '--output-model', 'LSTM-model.pkl']
        try:
            model.main(arg_list)
        except ValueError:
            print("Passed --algorithm missing check.\n")

        # check unknown algorithm
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d',
                    '--output-model', 'LSTM-model.pkl',
                    '--algorithm', 'foo-bar']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed unknown --algorithm check.\n")

        # check unknown argument
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA.rd.npy',
                    '--train', '%d',
                    '--output-model', 'LSTM-model.pkl',
                    '--algorithm', 'LSTM',
                    '--foo-bar']
        try:
            model.main(arg_list)
        except SystemExit:
            print("Passed unknown argument check.\n")  

# test LSTM model
#################

def test_LSTM(args):

    if args.test_LSTM or args.test_all:

        # check if LSTM model has already been trained
        if not os.path.exists(os.path.join(test_reduce_dir, 'LSTM-model.pkl')):

            # train LSTM using 100 dimension auto-correlated PCA results
            # use first 400 as training set, last 100 as test set
            arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[0:401]'),
                        '--input-file', 'out.cahn_hilliard_inc_auto_PCA_100.rd.npy',
                        '--train', '%d[10:90]',
                        '--over-write',
                        '--output-model', 'LSTM-model.pkl',
                        '--algorithm', 'LSTM',
                        '--num-dim', '5',
                        '--LSTM-steps', '10',
                        '--file-batch-size', '50']
            model.main(arg_list)
                
        # compute LSTM predictions
        arg_list = ['--ensemble', os.path.join(test_reduce_dir, 'workdir.%d[401:]'),
                    '--input-file', 'out.cahn_hilliard_inc_auto_PCA_100.rd.npy',
                    '--output-file', 'LSTM-preds.px.npy',
                    '--test', '%d[10:90]', '11',
                    '--over-write',
                    '--input-model', os.path.join(test_reduce_dir, 'LSTM-model.pkl')]
        model.main(arg_list)

# command line entry
if __name__ == "__main__":

    # get command line flags
    parser = init_parser()
    args = parser.parse_args()

    # run tests
    test_UI(args)
    test_LSTM(args)
