# This is a script to test the csv utitlies using the
# memphis test data.  Tests csv operations create, join, and expand.
#
# WARNING: When this script is run, the output_dir is
# deleted, unless delete_output_dir is set to False.
# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# NOTE: To test the --join option, you must first run test-reduce.py
# with the end-state option to create the xy-out .csv files to join.

# S. Martin
# 3/23/2021

# file/path manipulation
import os
import shutil

# reduction command line code
import romans.table as table

# delete output_dir
delete_output_dir = True

# directory containing the phase-field test dataset
test_data_dir = 'data/phase-field/test_data'

# output directory to use for testing
output_dir = 'data/phase-field/test_table'

# convert directory containing end state images and movies
convert_dir = 'data/phase-field/test_convert'

# reduce directory containing the end state reductions
reduce_dir = 'data/phase-field/test_reduce'

# uri-root-out conversion (location of files on cluster)
uri_root_out = 'file://memphis/phase-field/test_out'

# delete output_dir, if present
if delete_output_dir:
    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

# test UI
#########

# check for no arguments
arg_list = []
try:
    table.main(arg_list)
except SystemExit:
    print("Passed no argument check.\n")

# check output-dir
arg_list = ['--create']
try:
    table.main(arg_list) 
except SystemExit:
    print("Passed create --output-dir check.\n")

# check for --csv-out
arg_list = ['--create',
            '--output-dir', output_dir]
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --csv-out check.\n")

# check for multiple competing requests (create & join)
arg_list = ['--create',
            '--output-dir', output_dir,
            '--csv-out', 'out.csv',
            '--join', 'join.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --join too many arguments check.\n")

# check create --ensemble argument
arg_list = ['--create',
            '--output-dir', output_dir,
            '--csv-out', 'out.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed create --ensemble check.\n")

# check create --input-files
arg_list = ['--create',
            '--output-dir', output_dir,
            '--csv-out', 'out.csv',
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d')]
try:
    table.main(arg_list)
except SystemExit:
    print("Passed create --input-files check.\n")

# check create --input-header
arg_list = ['--create',
            '--output-dir', output_dir,
            '--csv-out', 'out.csv',
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
            '--input-files', 'in.cahn_hilliard']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed create --input-header check.\n")

# check join one file
arg_list = ['--join', 'in.csv',
            '--output-dir', output_dir,
            '--csv-out', 'out.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed join check.\n")

# check join with --ignore-index
arg_list = ['--join', 'in.csv',
            '--output-dir', output_dir,
            '--csv-out', 'out.csv',
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'VTK Files',
            '--ignore-index']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --join --csv-no-index missing.\n")    

# check join with uri conversion
arg_list = ['--join', 'in.csv',
            '--output-dir', output_dir,
            '--ensemble', os.path.join(test_data_dir), 'workdir.%d',
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'VTK Files',
            '--csv-out', 'out.csv',
            '--convert-cols', 'foo', 'bar']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --join --uri-root-out missing.\n")

# check join with --convert-cols missing
arg_list = ['--join', 'in.csv',
            '--output-dir', output_dir,
            '--ensemble', os.path.join(test_data_dir), 'workdir.%d',
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'VTK Files',
            '--csv-out', 'out.csv',
            '--uri-root-out', 'uri-out.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --join --convert-cols missing.\n")

# check expand with no --expand-header option
arg_list = ['--expand', 'metadata.csv',
            '--output-dir', output_dir,
            '--csv-out', 'expand-default.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --expand-header check.\n")

# create csv testing
####################

# check create
arg_list = ['--create',
            '--output-dir', output_dir,
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'Input Deck',
            '--csv-out', 'metadata.csv',
            '--csv-index', 'Simulation Index']
table.main(arg_list)
print("Created metadata.csv.\n")

# test create --overwrite
arg_list = ['--create',
            '--output-dir', output_dir,
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'Input Deck',
            '--csv-out', 'metadata.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed create --over-write check.\n")

# join csv testing
##################

# test join end-state and movies (parameter space model)
arg_list = ['--join', 
            os.path.join(output_dir, 'metadata.csv'),
            os.path.join(convert_dir, 'end-state.csv'),
            os.path.join(convert_dir, 'movies.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'ps.csv',
            '--over-write',
            '--csv-no-index',
            '--csv-headers', 
            'mobility_coefficients-1', 'mobility_coefficients-2', 
            'composition_distribution-1', 'End State', 'Movie',
            '--uri-root-out', uri_root_out,
            '--convert-cols', 'End State', 'Movie']
table.main(arg_list)
print("Created ps.csv.\n")

# test join one file and non-existing file links
arg_list = ['--join',
            os.path.join(output_dir, 'metadata.csv'),
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
            '--input-files', 'out.cahn_hilliard_%d.jpg',
            '--input-header', 'Images',
            '--output-dir', output_dir,
            '--csv-out', 'metadata-images.csv',
            '--over-write']
table.main(arg_list)
print("Created metadata-images.csv.\n")

# test join one file and existing file links
arg_list = ['--join',
            os.path.join(output_dir, 'metadata.csv'),
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d'),
            '--input-files', 'out.cahn_hilliard_%d.vtk',
            '--input-header', 'Images',
            '--output-dir', output_dir,
            '--csv-out', 'metadata-meshes.csv',
            '--over-write']
table.main(arg_list)
print("Created metadata-meshes.csv.\n")

# join all end-state xy coords
arg_list = ['--join',
            os.path.join(output_dir, 'metadata.csv'),
            os.path.join(convert_dir, 'end-state.csv'),
            os.path.join(convert_dir, 'movies.csv'),
            os.path.join(reduce_dir, 'PCA-end-state-xy.csv'),
            os.path.join(reduce_dir, 'auto-PCA-end-state-xy.csv'),
            os.path.join(reduce_dir, 'Isomap-end-state-xy.csv'),
            os.path.join(reduce_dir, 'auto-Isomap-end-state-xy.csv'),
            os.path.join(reduce_dir, 'tSNE-end-state-xy.csv'),
            os.path.join(reduce_dir, 'auto-tSNE-end-state-xy.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'ps-PCA-Isomap-tSNE-end-state.csv',
            '--csv-no-index',
            '--ignore-index',
            '--over-write',
            '--uri-root-out', uri_root_out,
            '--convert-cols', 'End State', 'Movie']
table.main(arg_list)
print("Created ps-PCA-Isomap-tSNE-end-state.csv.\n")

# join one file and dimension reduction links
# (for testing videoswarm expansion option later)
arg_list = ['--join',
            os.path.join(output_dir, 'metadata.csv'),
            os.path.join(convert_dir, 'end-state.csv'),
            os.path.join(convert_dir, 'movies.csv'),
            os.path.join(reduce_dir, 'time-aligned-PCA.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'metadata-time-aligned-PCA.csv',
            '--csv-no-index',
            '--over-write',
            '--ignore-index',
            '--csv-headers',
            'mobility_coefficients-1', 'mobility_coefficients-2', 
            'composition_distribution-1', 'End State', 'Movie', 'Time Aligned PCA',
            '--uri-root-out', uri_root_out,
            '--convert-cols', 'End State', 'Movie']
table.main(arg_list)
print("Created metadata-time-aligned.csv.\n")

# join Betti end-state dimension reduction
arg_list = ['--join',
            os.path.join(output_dir, 'metadata.csv'),
            os.path.join(convert_dir, 'end-state.csv'),
            os.path.join(convert_dir, 'movies.csv'),
            os.path.join(reduce_dir, 'Betti-end-state.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'metadata-Betti-end-state.csv',
            '--csv-no-index',
            '--over-write',
            '--csv-headers', 
            'mobility_coefficients-1', 'mobility_coefficients-2', 
            'composition_distribution-1', 'End State', 'Movie', 
            'Betti End State X', 'Betti End State Y',
            '--uri-root-out', uri_root_out,
            '--convert-cols', 'End State', 'Movie']
table.main(arg_list)
print("Created Betti-end-state.csv.\n")

# join incremental auto-PCA with all time points
arg_list = ['--join',
            os.path.join(output_dir, 'metadata.csv'),
            os.path.join(convert_dir, 'end-state.csv'),
            os.path.join(convert_dir, 'movies.csv'),
            os.path.join(reduce_dir, 'incremental-whiten-PCA-1000.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'metadata-inc-auto-PCA.csv',
            '--over-write',
            '--uri-root-out', uri_root_out,
            '--convert-cols', 'End State', 'Movie']
table.main(arg_list)
print("Created metadata-inc-PCA.csv.\n")

# expand csv testing
####################

# test file expansion with invalid links
arg_list = ['--expand', os.path.join(output_dir, 'metadata-images.csv'),
            '--expand-header', 'Images',
            '--output-dir', output_dir,
            '--csv-out', 'metadata-expand-files.csv']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed missing files check.\n")

# test file expansion with valid links
arg_list = ['--expand', os.path.join(output_dir, 'metadata-meshes.csv'),
            '--expand-header', 'Images',
            '--output-dir', output_dir,
            '--csv-out', 'metadata-expand-files.csv']
table.main(arg_list)
print("Passed file expansion with valid links.\n")

# test parameter space number of dimensions > 1
arg_list = ['--expand', os.path.join(output_dir, 'metadata-inc-auto-PCA.csv'),
            '--expand-header', 'Incremental Auto-PCA',
            '--output-dir', output_dir,
            '--csv-out', os.path.join(output_dir, 'ps-inc-auto-PCA.csv'),
            '--plugin', 'parameter_space',
            '--num-dim', '0']
try:
    table.main(arg_list)
except ValueError:
    print("Passed number dimensions check.\n")

# test parameter space expansion (using full PCA dataset)
arg_list = ['--expand', os.path.join(output_dir, 'metadata-inc-auto-PCA.csv'),
            '--expand-header', 'Incremental Whiten PCA',
            '--output-dir', output_dir,
            '--csv-out', 'ps-inc-PCA.csv',
            '--plugin', 'parameter_space',
            '--remove-expand-col',
            '--include-original-index',
            '--num-dim', '3',
            '--csv-no-index']
table.main(arg_list)
print("Created ps-inc-PCA.csv.\n")

# test parameter space expansion (using full auto-PCA dataset)
arg_list = ['--expand', os.path.join(output_dir, 'metadata-inc-auto-PCA.csv'),
            '--expand-header', 'Incremental Whiten PCA',
            '--output-dir', output_dir,
            '--csv-out', 'ps-inc-auto-PCA.csv',
            '--plugin', 'parameter_space',
            '--remove-expand-col',
            '--include-original-index',
            '--num-dim', '3',
            '--csv-no-index',
            '--over-write']
table.main(arg_list)
print("Created ps-inc-auto-PCA.csv.\n")

# test videoswarm --video-duration parameter < 0
arg_list = ['--expand', os.path.join(output_dir, 'metadata-time-aligned-PCA.csv'),
            '--expand-header', 'Time Aligned PCA',
            '--output-dir', os.path.join(output_dir, 'videoswarm-out'),
            '--csv-out', 'videoswarm.csv',
            '--plugin', 'videoswarm',
            '--remove-expand-col',
            '--video-fps', '-1']
try:
    table.main(arg_list)
except ValueError:
    print("Passed --video-duration negative check.\n")

# test videoswarm required output-format
arg_list = ['--expand', os.path.join(output_dir, 'metadata-time-aligned-PCA.csv'),
            '--expand-header', 'Time Aligned PCA',
            '--output-dir', os.path.join(output_dir, 'time-aligned-PCA-vs'),
            '--csv-out', 'videoswarm.csv',
            '--plugin', 'videoswarm',
            '--remove-expand-col',
            '--video-fps', '25']
table.main(arg_list)
print("Created time-aligned-PCA-vs files.\n")

# concat csv testing
####################

# here we concatenate training set results from dimension reduction
# using simulations 1-400 and test set results using simulations 401-500

# first we create a metadata file using simulations 1-400 
arg_list = ['--create',
            '--output-dir', output_dir,
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d[:401]'),
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'Input Deck',
            '--csv-out', 'metadata-train.csv',
            '--csv-index', 'Simulation Index']
table.main(arg_list)
print("Created metadata-train.csv.\n")

# next we join the training metadata with the training xy coords
arg_list = ['--join',
            os.path.join(output_dir, 'metadata-train.csv'),
            os.path.join(reduce_dir, 'auto-PCA-end-state-train.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'metadata-auto-PCA-train.csv',
            '--over-write']
table.main(arg_list)
print("Created metadata-auto-PCA-train.csv.\n")

# we repeat for the test set
arg_list = ['--create',
            '--output-dir', output_dir,
            '--ensemble', os.path.join(test_data_dir, 'workdir.%d[401:]'),
            '--input-files', 'in.cahn_hilliard',
            '--input-header', 'Input Deck',
            '--csv-out', 'metadata-test.csv',
            '--csv-index', 'Simulation Index']
table.main(arg_list)
print("Created metadata-test.csv.\n")

arg_list = ['--join',
            os.path.join(output_dir, 'metadata-test.csv'),
            os.path.join(reduce_dir, 'auto-PCA-end-state-test.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'metadata-auto-PCA-test.csv',
            '--over-write']
table.main(arg_list)
print("Created metadata-auto-PCA-test.csv.\n")

# check --add-origin-col present
arg_list=['--concat',
          os.path.join(output_dir, 'metadata-auto-PCA-train.csv'),
          os.path.join(output_dir, 'metadata-auto-PCA-test.csv'),
          '--output-dir', output_dir,
          '--csv-out', 'metadata-auto-PCA.csv',
          '--over-write',
          '--origin-col-names', 'Train']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed --add-origin-col not present check.\n")

# check matching number of origin column names
arg_list=['--concat',
          os.path.join(output_dir, 'metadata-auto-PCA-train.csv'),
          os.path.join(output_dir, 'metadata-auto-PCA-test.csv'),
          '--output-dir', output_dir,
          '--csv-out', 'metadata-auto-PCA.csv',
          '--over-write',
          '--add-origin-col', 'Train/Test',
          '--origin-col-names', 'Train']
try:
    table.main(arg_list)
except SystemExit:
    print("Passed mismatching origin column names check.\n")

# now we concatenate the tables
arg_list=['--concat',
          os.path.join(output_dir, 'metadata-auto-PCA-train.csv'),
          os.path.join(output_dir, 'metadata-auto-PCA-test.csv'),
          '--output-dir', output_dir,
          '--csv-out', 'metadata-auto-PCA.csv',
          '--over-write',
          '--add-origin-col', 'Train/Test',
          '--origin-col-names', 'Train', 'Test']
table.main(arg_list)
print("Concatenated metadata-auto-PCA.csv.\n")

# finally we join the movies and images
arg_list = ['--join', 
            os.path.join(output_dir, 'metadata-auto-PCA.csv'),
            os.path.join(convert_dir, 'end-state.csv'),
            os.path.join(convert_dir, 'movies.csv'),
            '--output-dir', output_dir,
            '--csv-out', 'ps-train-test-auto-PCA.csv',
            '--over-write',
            '--csv-no-index',
            '--uri-root-out', uri_root_out,
            '--convert-cols', 'End State', 'Movie']
table.main(arg_list)
print("Created ps-train-test-auto-PCA.csv.\n")
