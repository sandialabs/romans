# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script tests the reduce.py script using the phase-field
# test data and various dimension reduction algorithms.  Tests
# are done on the UI and on the MEMPHIS test data.  The reduction
# algorithms, to a lesser degree, are also tested.
#
# NOTE: Different tests can be performed depending on the
# test_* flags set below.

# S. Martin
# 3/22/2021

# file/path manipulation
import os
import shutil
import filecmp

# flags for tests to run
import argparse

# reduction command line code
import romans.reduce as reduce
import romans.table as table
import romans.validate as validate

# file paths

# directory containing the phase-field test dataset
test_data_dir = 'data/phase-field/test_data'

# output directory to use for testing
output_dir = 'data/phase-field/test_reduce'

# output directory to use for testing
table_dir = 'data/phase-field/test_table'

# convert directory containing end state images and movies
convert_dir = 'data/phase-field/test_convert'

# uri-root-out conversion (location of files on cluster)
uri_root_out = 'file://memphis/phase-field/test_out'

# inc-PCA from cluster
inc_PCA = 'data/phase-field/inc-PCA'

# inc-whiten-PCA from cluster
inc_whiten_PCA = 'data/phase-field/inc-whiten-PCA'

# inc-auto-PCA from cluster
inc_auto_PCA = 'data/phase-field/inc-auto-PCA-100'

# reduced dataset by sampling
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
        help="Test command line UI for reduce.py.")
    parser.add_argument('--test-save-load', action="store_true", default=False,
        help="Test save and load capability in reduce.py.")
    parser.add_argument('--test-split', action="store_true", default=False,
        help="Test training/test set split using phase-filed simulations 1-400 for " +
             "training set and simulations > 400 for test set.")
    parser.add_argument('--test-end-state', action="store_true", default=False,
        help="Do end-state dimension reductions.")
    parser.add_argument('--test-time-aligned', action="store_true", default=False,
        help="Do time-aligned dimension reductions.")
    parser.add_argument('--test-all-time', action="store_true", default=False,
        help="Do dimension reductions with all time steps simultaneously.")
    parser.add_argument('--test-betti', action="store_true", default=False,
        help="Do Betti number calculations.")
    parser.add_argument('--test-umap', action="store_true", default=False,
        help="Perform umap reduction on test set.")
    parser.add_argument('--test-auto-encoder', action="store_true", default=False,
        help="Do auto-encoder dimension reductions.")
    parser.add_argument('--test-var-auto', action="store_true", default=False,
        help="Test variational auto-encoder.")
    parser.add_argument('--test-parallel', action="store_true", default=False,
        help="Run parallel tests.")
    parser.add_argument('--test-restart', action="store_true", default=False,
        help="Run restart testing.")
    parser.add_argument('--test-rd-npy', action="store_true", default=False,
        help="Test dimension reduction loaded from rd.npy file.")
    parser.add_argument('--test-all', action="store_true", default=False,
        help="Run every test.")

    return parser

# delete output_dir, if present
# create if not present
def delete_output_dir(args):

    if os.path.isdir(output_dir):
        if args.delete_output_dir:
            shutil.rmtree(output_dir)
    
    else:
        os.path.mkdir(output_dir)

# argument parser testing
#########################

def test_UI(args):

    if args.test_UI or args.test_all:

        # no arguments
        arg_list = []
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed no argument check.\n")

        # missing --input-files
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d')]
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --input-files check.\n")

        # missing --output-dir
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_0.vtk']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --output-dir check.\n")

        # missing --output-file
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_0.vtk',
                    '--output-dir', output_dir]
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --output-file check.\n")

        # --output-file extension
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --output-file extension check.\n")

        # test missing --algorithm
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --algorithm missing check.\n")
        
        # test algorithm incorrect
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'super-mega-duper']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --algorithm incorrect check.\n")

        # test number of dimensions
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --num-dim check.\n")

        # test field-var
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --field-var check.\n")

        # test over-write
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--field-var', 'phase_field']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --over-write check.\n")

        # test field-var incorrect
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--field-var', 'goof-ball',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --field-var incorrect check.\n")
        
        # test --input-format
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--input-format', 'npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--field-var', 'phase_field',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --input-format check.\n")
        
        # un-recognized argument
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--input-format', 'npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--field-var', 'phase_field',
                    '--over-write', '--foo']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed un-recognized argument check.\n")

        # check auto-correlate without binary
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--field-var', 'phase_field',
                    '--auto-correlate',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print ("Passed missing --binary check.\n")
        
        # check csv-out
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--field-var', 'phase_field',
                    '--auto-correlate',
                    '--over-write',
                    '--csv-out', 'end-state-PCA-links.csv']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed missing --csv-header.\n")

        # check save model file extension
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--output-model', 'pca-model.txt',
                    '--num-dim', '50']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed .pkl extension save model.\n")
        
        # check load model file extension
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--input-model', 'pca-model.txt',
                    '--num-dim', '50']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed .pkl extension load model.\n")

# save/load models
##################

def test_save_load(args):

    if args.test_save_load or args.test_all:

        # save PCA model
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_10.rd.npy',
                    '--algorithm', 'PCA',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'PCA-end-state-trained.csv',
                    '--xy-header', 'PCA End State',
                    '--output-model', 'end-state-pca-model-10.pkl',
                    '--num-dim', '10']
        reduce.main(arg_list)
        print()

        # load PCA model with parameters provided
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_10.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--input-model', os.path.join(output_dir, 
                        'end-state-pca-model-10.pkl'),
                    '--num-dim', '10']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed load model with parameter check.\n")

        # load PCA model and run
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_10.rd.npy',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--input-model', os.path.join(output_dir, 
                        'end-state-pca-model-10.pkl'),
                    '--xy-out', 'PCA-end-state-loaded.csv',
                    '--xy-header', 'PCA End State']
        reduce.main(arg_list)

        # compare trained and loaded models
        if filecmp.cmp(os.path.join(output_dir, 'PCA-end-state-trained.csv'),
            os.path.join(output_dir, 'PCA-end-state-loaded.csv'), shallow=False):
            print("Passed load/save file comparison check.\n")

        # save time-aligned model (using only 3 time points)
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_%d[0:1500000].vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--num-dim', '2',
                    '--time-align', '10',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--file-batch-size', '500',
                    '--output-model', 'PCA-time-aligned.pkl']
        reduce.main(arg_list)

        # load time-aligned model
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_%d[0:1500000].vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA_loaded.rd.npy',
                    '--input-model', os.path.join(output_dir, 'PCA-time-aligned.pkl'),
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--file-batch-size', '500']
        reduce.main(arg_list)

        # compare trained and loaded model results
        results_same = True
        for i in range(1,501):
            if not filecmp.cmp(os.path.join(output_dir, 'workdir.' + str(i) + '/out.cahn_hilliard_time_aligned_PCA.rd.npy'),
                os.path.join(output_dir, 'workdir.' + str(i) + '/out.cahn_hilliard_time_aligned_PCA_loaded.rd.npy'), shallow=False):
                results_same = False
        print("Time aligned traied/loaded results same: " + str(results_same))

# train-test set split
######################

def test_split(args):
    if args.test_split or args.test_all:

        # train on first 400 simulations end state
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[:401]'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_50_train.rd.npy',
                    '--algorithm', 'PCA',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'auto-PCA-end-state-train.csv',
                    '--xy-header', 'Auto-PCA End State',
                    '--output-model', 'pca-model.pkl',
                    '--num-dim', '50']
        reduce.main(arg_list)
        print()

        # test on last 100 simulations
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[401:]'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_50_test.rd.npy',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--xy-out', 'auto-PCA-end-state-test.csv',
                    '--xy-header', 'Auto-PCA End State',
                    '--input-model', os.path.join(output_dir, 'pca-model.pkl')]
        reduce.main(arg_list)
        print()

# end state test reductions
###########################

def test_end_state(args):

    if args.test_end_state or args.test_all:

        # PCA on end state
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'PCA-end-state-xy.csv',
                    '--xy-header', 'PCA End State',
                    '--csv-out', 'PCA-end-state-links.csv',
                    '--csv-header', 'PCA End State',
                    '--num-dim', '50']
        reduce.main(arg_list)

        # auto-correlated PCA on end state
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_auto_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--xy-out', 'auto-PCA-end-state-xy.csv',
                    '--xy-header', 'Auto-PCA End State',
                    '--num-dim', '50']
        reduce.main(arg_list)

        # Isomap on end state
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_Isomap.rd.npy',
                    '--algorithm', 'Isomap',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'Isomap-end-state-xy.csv',
                    '--xy-header', 'Isomap End State',
                    '--num-dim', '2']
        reduce.main(arg_list)

        # auto-correlated Isomap on end state
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_auto_Isomap.rd.npy',
                    '--algorithm', 'Isomap',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'auto-Isomap-end-state-xy.csv',
                    '--xy-header', 'Auto-Isomap End State',
                    '--num-dim', '2']
        reduce.main(arg_list)

        # tSNE on end state (use PCA output)
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_end_state_PCA_50.rd.npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_tSNE.rd.npy',
                    '--algorithm', 'tSNE',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'tSNE-end-state-xy.csv',
                    '--xy-header', 'tSNE End State',
                    '--num-dim', '2']
        reduce.main(arg_list)

        # auto-correlated tSNE on end state (using auto-correlated PCA)
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_end_state_auto_PCA_50.rd.npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_auto_tSNE.rd.npy',
                    '--algorithm', 'tSNE',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--xy-out', 'auto-tSNE-end-state-xy.csv',
                    '--xy-header', 'Auto-tSNE End State',
                    '--num-dim', '2']
        reduce.main(arg_list)

# UMAP
######

def test_umap(args):

    if args.test_umap or args.test_all:

        # Umap on inc-auto test set
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_end_state_auto_PCA_50.rd.npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_umap_50.rd.npy',
                    '--algorithm', 'Umap',
                    '--over-write',
                    '--csv-out', 'umap.csv',
                    '--csv-header', 'Umap',
                    '--num-dim', '10',
                    '--output-model', 'auto-PCA-50-umap.pkl']
        reduce.main(arg_list)

        # join to create parameter space model
        arg_list = ['--join', 
                    os.path.join(table_dir, 'metadata.csv'),
                    os.path.join(convert_dir, 'end-state.csv'),
                    os.path.join(convert_dir, 'movies.csv'),
                    os.path.join(output_dir, 'umap.csv'),
                    '--output-dir', output_dir,
                    '--csv-out', 'umap-meta.csv',
                    '--over-write',
                    '--csv-no-index',
                    '--ignore-index',
                    '--csv-headers',
                    'mobility_coefficients-1', 'mobility_coefficients-2', 
                    'composition_distribution-1', 'End State', 'Movie', 'Umap',
                    '--uri-root-out', uri_root_out,
                    '--convert-cols', 'End State', 'Movie']
        table.main(arg_list)
        print("Created umap-meta.csv.\n")

        # expand to show xyz coords
        arg_list = ['--expand', os.path.join(output_dir, 'umap-meta.csv'),
                    '--expand-header', 'Umap',
                    '--output-dir', output_dir,
                    '--csv-out', 'umap-full-test-set.csv',
                    '--plugin', 'parameter_space',
                    '--remove-expand-col',
                    '--num-dim', '10',
                    '--over-write']
        table.main(arg_list)
        print("Created umap-full-test-set.csv.\n")

# auto-encoder
##############

def test_auto_encoder(args):

    if args.test_auto_encoder or args.test_all:

        # auto-encoder on full training set
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_inc_whiten_PCA_1000.rd.npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_inc_whiten_PCA_1000_autoencoder.rd.npy',
                    '--algorithm', 'auto-encoder',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--csv-out', 'inc-whiten-PCA-1000-auto-encoder.csv',
                    '--csv-header', 'Whiten PCA Auto-Encoder',
                    '--num-dim', '10',
                    '--MLP-arch', '750', '500', '250', '100', '50',
                    '--num-processes', '4',
                    '--epochs', '100',
                    '--batch-size', '250',
                    '--output-model', 'inc-whiten-PCA-1000-autoencoder.pkl']
        reduce.main(arg_list)

# variational auto-encoder
##########################

def test_var_auto(args):

    if args.test_var_auto or args.test_all:

            # auto-encoder on full training set
            arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                        '--input-files', 'out.cahn_hilliard_inc_whiten_PCA_1000.rd.npy',
                        '--output-dir', output_dir,
                        '--output-file', 'out.cahn_hilliard_inc_whiten_PCA_1000_var_auto.rd.npy',
                        '--algorithm', 'auto-encoder',
                        '--model-type', 'var',
                        '--field-var', 'phase_field',
                        '--over-write',
                        '--csv-out', 'whiten-PCA-1000-var-auto-encoder.csv',
                        '--csv-header', 'Whiten PCA Variational Auto-Encoder',
                        '--num-dim', '10',
                        '--num-processes', '4',
                        '--epochs', '50',
                        '--batch-size', '100',
                        '--output-model', 'inc-whiten-PCA-1000-var-auto.pkl']
            reduce.main(arg_list)

# betti number algorithm (experimental)
#######################################

def test_betti(args):

    if args.test_betti or args.test_all:

        # check betti arguments
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_Betti.rd.npy',
                    '--algorithm', 'Betti',
                    '--field-var', 'phase_field',
                    '--over-write']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --rows Betti number check.\n")
        
        arg_list += ['--rows', '512']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --cols Betti number check.\n")

        arg_list += ['--cols', '512']
        try:
            reduce.main(arg_list)
        except ValueError:
            print("Passed --threshold Betti number check.\n") 

        # betti number calculation
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_Betti.rd.npy',
                    '--algorithm', 'Betti',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--rows', '512',
                    '--cols', '512',
                    '--threshold', '0',
                    '--xy-out', 'Betti-end-state.csv',
                    '--xy-header', 'Betti End State',
                    '--num-dim', '2',
                    '--log-file', 'betti.log']
        reduce.main(arg_list)

# time-aligned algorithms
#########################

def test_time_aligned(args):

    if args.test_time_aligned or args.test_all:

        # time-aligned PCA in memory (no --file-batch-size)
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[0:50]'),
                    '--input-files', 'out.cahn_hilliard_%d[0:2000000].vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--time-align', '10',
                    '--num-dim', '2',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--csv-out', 'time-aligned-PCA.csv',
                    '--csv-header', 'Time Aligned PCA']
        reduce.main(arg_list)
        print("Passed time-aligned in memory.\n")

        # time-aligned PCA in batch using fit (--file-batch-size == size ensemble)
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[0:50]'),
                    '--input-files', 'out.cahn_hilliard_%d[0:2000000].vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--time-align', '10',
                    '--num-dim', '2',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--file-batch-size', '50',
                    '--csv-out', 'time-aligned-PCA.csv',
                    '--csv-header', 'Time Aligned PCA']
        reduce.main(arg_list)
        print("Passed time-aligned non-incremental.\n")

        # time-aligned PCA using incremental in batches
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[0:60]'),
                    '--input-files', 'out.cahn_hilliard_%d[0:2000000].vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                    '--algorithm', 'incremental-PCA',
                    '--time-align', '10',
                    '--num-dim', '2',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--file-batch-size', '20',
                    '--csv-out', 'time-aligned-PCA.csv',
                    '--csv-header', 'Time Aligned PCA']
        reduce.main(arg_list)
        print("Passed time-aligned incremental in batches.\n")

# all time step algorithms
##########################

def test_all_time(args):

    if args.test_all_time or args.test_all:

        # incremental PCA
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_inc_whiten_PCA_1000.rd.npy',
                    '--algorithm', 'incremental-PCA',
                    '--field-var', 'phase_field',
                    '--whiten',
                    '--over-write',
                    '--csv-out', 'incremental-whiten-PCA-1000.csv',
                    '--csv-header', 'Incremental Whiten PCA',
                    '--num-dim', '1000',
                    '--file-batch-size', '1500',
                    '--output-model', 'inc-whiten-PCA-1000.pkl']
        reduce.main(arg_list)

# parallel testing
##################

def test_parallel(args):

    if args.test_parallel or args.test_all:

        # test parallel memphis only
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_5000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_auto_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--parallel',
                    '--plugin', 'videoswarm',
                    '--xy-out', 'auto-PCA-end-state-parallel-xy.csv',
                    '--xy-header', 'Auto-PCA End State',
                    '--num-dim', '50']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --plugin memphis --parallel check.\n")

        # test parallel
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_auto_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--parallel',
                    '--xy-out', 'auto-PCA-end-state-parallel-xy.csv',
                    '--xy-header', 'Auto-PCA End State',
                    '--num-dim', '50',
                    '--log-file', os.path.join(output_dir, 'auto-PCA-parallel.log')]
        reduce.main(arg_list)
    
        # compare to serial
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_end_state_auto_PCA_50.rd.npy',
                    '--algorithm', 'PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--xy-out', 'auto-PCA-end-state-serial-xy.csv',
                    '--xy-header', 'Auto-PCA End State',
                    '--num-dim', '50']
        reduce.main(arg_list)

        # compare parallel/serial models
        if filecmp.cmp(os.path.join(output_dir, 'auto-PCA-end-state-serial-xy.csv'),
            os.path.join(output_dir, 'auto-PCA-end-state-parallel-xy.csv'), shallow=False):
            print("Passed serial/parallel file comparison check.\n")
        else:
            print("Failed serial/parallel file comparison check " +
                  "(sometimes this is due to precision errors).\n")

# restart testing
#################

def test_restart(args):

    if args.test_restart or args.test_all:

        # restart testing (must be pickle file)
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_inc_auto_PCA_50.rd.npy',
                    '--algorithm', 'incremental-PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--restart', 'restart.txt',
                    '--num-dim', '50',
                    '--file-batch-size', '1000']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --restart file.pkl check.\n")

        # test for --output-model
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_50000000.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_inc_auto_PCA_50.rd.npy',
                    '--algorithm', 'incremental-PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--restart', 'restart.pkl',
                    '--num-dim', '50',
                    '--file-batch-size', '1000']
        try:
            reduce.main(arg_list)
        except SystemExit:
            print("Passed --restart --output-model check.\n")

        # check that baseline test has been computed,
        # otherwise comute baseline for incremental test
        if not os.path.exists(os.path.join(output_dir, 'inc-PCA-model-no-restart.pkl')):

            # baseline test incremental non-restart
            arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[491:501]'),
                        '--input-files', 'out.cahn_hilliard_%d.vtk',
                        '--output-dir', output_dir,
                        '--output-file', 'out.cahn_hilliard_inc_auto_PCA_50.rd.npy',
                        '--algorithm', 'incremental-PCA',
                        '--field-var', 'phase_field',
                        '--auto-correlate', '--binary',
                        '--over-write',
                        '--num-dim', '5',
                        '--file-batch-size', '100',
                        '--output-model', 'inc-PCA-model-no-restart.pkl']
            reduce.main(arg_list)
            print("Saved incremental non-restart baseline.\n")

        # restart incremental
        print("For the incremental PCA test you must use control-C")
        print("partway through (to simulate a crash) then re-run the test.")
        input("Press enter to start incremental PCA restart test ... ")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[491:501]'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_inc_auto_PCA_50.rd.npy',
                    '--algorithm', 'incremental-PCA',
                    '--field-var', 'phase_field',
                    '--auto-correlate', '--binary',
                    '--over-write',
                    '--restart', 'inc-restart.pkl',
                    '--num-dim', '5',
                    '--file-batch-size', '100',
                    '--output-model', 'inc-PCA-model.pkl']
        try:
            reduce.main(arg_list)
        except KeyboardInterrupt:
            print("Incremental restart test interrupted, continuing to next test.\n")

        # compare restart/non-restart models
        if filecmp.cmp(os.path.join(output_dir, 'inc-PCA-model.pkl'),
            os.path.join(output_dir, 'inc-PCA-model-no-restart.pkl'), shallow=False):
            print("Passed restart incremental file comparison check.\n")
        else:
            print("Failed restart incremental file comparison check.\n")

        # check that baseline test has been computed,
        # otherwise comute baseline for time aligned test
        if not os.path.exists(os.path.join(output_dir, 'time-aligned-PCA-model-no-restart.pkl')):

            # baseline test incremental non-restart
            arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[491:501]'),
                        '--input-files', 'out.cahn_hilliard_%d.vtk',
                        '--output-dir', output_dir,
                        '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                        '--algorithm', 'PCA',
                        '--time-align', '5',
                        '--num-dim', '2',
                        '--auto-correlate', '--binary',
                        '--field-var', 'phase_field',
                        '--over-write',
                        '--output-model', 'time-aligned-PCA-model-no-restart.pkl']
            reduce.main(arg_list)
            print("Saved time-aligned non-restart baseline.\n")

        # restart testing (time-aligned)
        print("For the time aligned PCA test you must use control-C")
        print("partway through (to simulate a crash) then re-run the test.")
        input("Press enter to start the time aligned PCA restart test ... ")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[491:501]'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--time-align', '5',
                    '--num-dim', '2',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--restart', 'time-aligned-restart.pkl',
                    '--output-model', 'time-aligned-PCA-model.pkl']
        try:
            reduce.main(arg_list)
        except KeyboardInterrupt:
            print("Time-aligned restart test interrupted, continuing to next test.\n")

        # compare restart/non-restart models
        if filecmp.cmp(os.path.join(output_dir, 'time-aligned-PCA-model.pkl'),
            os.path.join(output_dir, 'time-aligned-PCA-model-no-restart.pkl'), shallow=False):
            print("Passed restart time aligned file comparison check.\n")
        else:
            print("Failed restart time aligned file comparison check " +
                  "(sometimes this is due to precision errors).\n")

        # check that baseline test has been computed,
        # otherwise comute baseline for time aligned incremental test
        if not os.path.exists(os.path.join(output_dir, 'time-aligned-inc-PCA-model-no-restart.pkl')):

            # baseline test incremental non-restart
            arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[491:501]'),
                        '--input-files', 'out.cahn_hilliard_%d.vtk',
                        '--output-dir', output_dir,
                        '--output-file', 'out.cahn_hilliard_time_aligned_inc_PCA.rd.npy',
                        '--algorithm', 'incremental-PCA',
                        '--time-align', '3',
                        '--num-dim', '2',
                        '--auto-correlate', '--binary',
                        '--field-var', 'phase_field',
                        '--over-write',
                        '--file-batch-size', '5',
                        '--output-model', 'time-aligned-inc-PCA-model-no-restart.pkl']
            reduce.main(arg_list)
            print("Saved time-aligned incremental non-restart baseline.\n")

        # restart testing (time-aligned incremental)
        print("For the time aligned incremental PCA test you must use control-C")
        print("partway through (to simulate a crash) then re-run the test.")
        input("Press enter to start the time aligned incremental PCA restart test ... ")
        arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d[491:501]'),
                    '--input-files', 'out.cahn_hilliard_%d.vtk',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_inc_PCA.rd.npy',
                    '--algorithm', 'incremental-PCA',
                    '--time-align', '3',
                    '--num-dim', '2',
                    '--auto-correlate', '--binary',
                    '--field-var', 'phase_field',
                    '--over-write',
                    '--file-batch-size', '5',
                    '--restart', 'time-aligned-inc-restart.pkl',
                    '--output-model', 'time-aligned-incremental-PCA-model.pkl']
        try:
            reduce.main(arg_list)
        except KeyboardInterrupt:
            print("Time-aligned incremental restart test interrupted, continuing to next test.\n") 

        # compare restart/non-restart models
        if filecmp.cmp(os.path.join(output_dir, 'time-aligned-incremental-PCA-model.pkl'),
            os.path.join(output_dir, 'time-aligned-inc-PCA-model-no-restart.pkl'), shallow=False):
            print("Passed restart time aligned incremental file comparison check.\n")
        else:
            print("Failed restart time aligned incremental file comparison check " +
                  "(sometimes this is due to precision errors).\n")

# test reduction using rd.npy load
def test_rd_npy(args):

    if args.test_rd_npy or args.test_all:

        # check if inc-auto PCA has already been computed
        if not os.path.exists(os.path.join(output_dir, 
            'workdir.1/out.cahn_hilliard_inc_auto_PCA_100.rd.npy')):

            # compute .rd.npy on test data using inc-auto PCA
            arg_list = ['--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                '--input-files', 'out.cahn_hilliard_%d.vtk',
                '--output-dir', output_dir,
                '--output-file', 'out.cahn_hilliard_inc_auto_PCA_100.rd.npy',
                '--algorithm', 'incremental-PCA',
                '--auto-correlate', '--binary',
                '--field-var', 'phase_field',
                '--over-write',
                '--num-dim', '100',
                '--file-batch-size', '2000']
            reduce.main(arg_list)

        # compute time-aligned PCA using inc-auto PCA
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_inc_auto_PCA_100.rd.npy',
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_PCA.rd.npy',
                    '--algorithm', 'PCA',
                    '--time-align', '20',
                    '--num-dim', '10',
                    '--over-write',
                    '--output-model', 'time-aligned-PCA.pkl']
        reduce.main(arg_list)        
        print("Finished training time-aligned PCA model.\n")

        # test rd.npy from loaded model
        arg_list = ['--ensemble', os.path.join(output_dir, 'workdir.%d'),
                    '--input-files', 'out.cahn_hilliard_inc_auto_PCA_100.rd.npy', 
                    '--output-dir', output_dir,
                    '--output-file', 'out.cahn_hilliard_time_aligned_model_PCA.rd.npy',
                    '--input-model', os.path.join(output_dir, 'time-aligned-PCA.pkl'),
                    '--over-write',
                    '--field-var', 'phase_field']
        reduce.main(arg_list)

if __name__ == "__main__":

    # get command line flags
    parser = init_parser()
    args = parser.parse_args()

    # delete output directory, if requested
    delete_output_dir(args)

    # run tests
    test_UI(args)
    test_save_load(args)
    test_split(args)
    test_end_state(args)
    test_time_aligned(args)
    test_all_time(args)
    test_betti(args)
    test_umap(args)
    test_auto_encoder(args)
    test_var_auto(args)
    test_parallel(args)
    test_restart(args)
    test_rd_npy(args)