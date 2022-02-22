# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script creates a Slycat VideoSwarm model for a
# time-aligned model on the cluster.  We assume we
# have the files available:
#
# Phase Field Data:
# phase-field/training_data
# phase-field/test_data
#
# Media Files (see run-*-media.sh):
# phase-field/train_media
# phase-field/test_media
#
# Dimension Reduction (can be any reduction, for example time-aligned-Isomap):
# phase-field/time-aligned-Isomap/train/time-aligned-Isomap-train.csv
# phase-field/time-aligned/test/time-aligned-Isomap-test.csv

# S. Martin
# 6/4/2021

from numpy.core.numeric import False_
import romans.table as table

import os
import sys

# the training data and media are in fixed locations
train_data_dir = 'phase-field/training_data'
test_data_dir = 'phase-field/test_data'

train_media = 'phase-field/train_media'
test_media = 'phase-field/test_media'

# uri-root-out conversion (location of files on cluster)
uri_root = 'file://'

# flags to prevent re-running tasks
create_train = False
create_test = False
concat_train_test = False
expand_reduction = False

# do everything
create_all = True

# look for files relative to current directory
output_dir = os.getcwd()

# get .csv file in train or test directory
def get_csv_file (output_dir, train):

    # the .csv files are always in train/*.csv and test/*.csv
    train_dir = os.path.join(output_dir, train)
    train_files = [file for file in os.listdir(train_dir) if os.path.isfile(os.path.join(train_dir, file))]
    train_csv_files = [file for file in train_files if file.endswith(".csv")]

    # check that there is only on .csv file
    if len(train_csv_files) != 1:
        print("Did not find a unique .csv file in " + train + " directory -- " +  \
            "could not generate parameter space .csv file.")
        sys.exit()

    # dimension reduction .csv file
    return os.path.join(train_dir, train_csv_files[0])

# get training/testing .csv files
train_reduction = get_csv_file(output_dir, 'train')
test_reduction = get_csv_file(output_dir, 'test')

# get dimension reduction header from .csv file
with open(train_reduction, 'r') as f:
    header = f.readline()
    dimension_reduction_header = header.split(',')[1].strip()

# create training table
if create_train or create_all:
    
    # create a metadata file for the training data
    arg_list = ['--create',
                '--output-dir', output_dir,
                '--ensemble', os.path.join(train_data_dir, 'workdir.%d'),
                '--input-files', 'in.cahn_hilliard',
                '--input-header', 'Input Deck',
                '--over-write',
                '--csv-out', os.path.join(output_dir, 'metadata-train.csv'),
                '--csv-index', 'Simulation Index']
    table.main(arg_list)
    print("Created metadata-train.csv.\n")

    # join images, videos, and dimension reduction
    arg_list = ['--join', 
                os.path.join(output_dir, 'metadata-train.csv'),
                train_reduction,
                os.path.join(train_media, 'end-state.csv'),
                os.path.join(train_media, 'movies.csv'),
                '--output-dir', output_dir,
                '--csv-out', 'metadata-reduction-media-train.csv',
                '--over-write',
                '--ignore-index',
                '--csv-no-index',
                '--uri-root-out', uri_root + train_media,
                '--convert-cols', 'End State', 'Movie']
    table.main(arg_list)
    print("Created metadata-reduction-media-train.csv.\n")

# create test table
if create_test or create_all:

    # create a metadata file for the test data
    arg_list = ['--create',
                '--output-dir', output_dir,
                '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
                '--input-files', 'in.cahn_hilliard',
                '--input-header', 'Input Deck',
                '--over-write',
                '--csv-out', os.path.join(output_dir, 'metadata-test.csv'),
                '--csv-index', 'Simulation Index']
    table.main(arg_list)
    print("Created metadata-test.csv.\n")

    # join images, videos, and dimension reduction
    arg_list = ['--join', 
                os.path.join(output_dir, 'metadata-test.csv'),
                test_reduction,
                os.path.join(test_media, 'end-state.csv'),
                os.path.join(test_media, 'movies.csv'),
                '--output-dir', output_dir,
                '--csv-out', 'metadata-reduction-media-test.csv',
                '--over-write',
                '--ignore-index',
                '--csv-no-index',
                '--uri-root-out', uri_root + test_media,
                '--convert-cols', 'End State', 'Movie']
    table.main(arg_list)
    print("Created metadata-reduction-media-test.csv.\n")

# join train/test tables
if concat_train_test or create_all:

    # now we concatenate the tables
    arg_list=['--concat',
            os.path.join(output_dir, 'metadata-reduction-media-train.csv'),
            os.path.join(output_dir, 'metadata-reduction-media-test.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'metadata-reduction-media.csv',
            '--over-write',
            '--add-origin-col', 'Train/Test',
            '--origin-col-names', 'Train', 'Test']
    table.main(arg_list)
    print("Concatenated metadata-reduction-media.csv.\n")

# expand dimension reduction coordinates
if expand_reduction or create_all:
            
    # test parameter space expansion (using full auto-Isomap dataset)
    arg_list = ['--expand', os.path.join(output_dir, 'metadata-reduction-media.csv'),
                '--expand-header', dimension_reduction_header,
                '--output-dir', os.path.join(output_dir, 'vs-dir'),
                '--csv-out', 'vs.csv',
                '--plugin', 'videoswarm',
                '--remove-expand-col',
                '--video-fps', '25',
                '--csv-no-index',
                '--over-write',
                '--csv-headers', 'mobility_coefficients-1', 'mobility_coefficients-2',
                'composition_distribution-1', 'End State', 'Movie', 'Train/Test',
                dimension_reduction_header + " Dimension 1",
                dimension_reduction_header + " Dimension 2",
                dimension_reduction_header + " Dimension 3"]
    table.main(arg_list)
    print("Created vs-dir directory.\n")
