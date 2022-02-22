# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains a command line interface that can be
# used to validate a dimension reduction or proxy model, 
# computing statistics and producing plots.

# S. Martin
# 6/14/2021

# standard library

# logging and error handling
import logging
import sys

# 3rd party libraries
import matplotlib.pyplot as plt
import numpy as np

# local imports
import romans
import ensemble
import algorithms.reduction as reduction
import algorithms.proxy as proxy

# set up argument parser
def init_parser():

    # define our own version of the romans parser
    description = "Validates a dimension reduction or proxy model by computing statistics " + \
                  "and producing plots.  Uses Python-like %d[::] notation, where %d[::] " + \
                  'specifies a range of numbers in a directory name. For example ' + \
                  '"workdir.%d[0:10:2]" would specify every other file from "workdir.0"' + \
                  'to "workdir.9".'
                  
    parser = romans.ArgumentParser (description=description)

    # validate dimension reduction or proxy model
    parser.add_argument("--reduction", action="store_true", default=False,
                        help="Validate dimension reduction results.  Note that you need " 
                        "to set the original arguments for the reduction algorithm to "
                        "obtain accurate reconstructions.")
    parser.add_argument("--proxy", action="store_true", default=False,
                        help="Validate proxy model.")

    # get files to convert from command line
    parser.add_argument("--ensemble", help="Directory or directories to include in ensemble, "
                        "specified using the Python like %%d[::] notation described above.")
    parser.add_argument("--input-file", help="Files per ensemble directory to use as input "
                        "for a model.  For proxy validation, the input files are expected "
                        "to have the same name and end with extension .rd.npy.")
    parser.add_argument("--input-pred-file", help="Files per ensemble directory containing "
                        "proxy predictions.  The prediction files are expected to have the "
                        "same name and end with extension .px.npy.")

    # or get files to convert from .csv file
    parser.add_argument("--csv-file", help="CSV file which specifies ensemble directories "
                        "and input files (alternate to using --ensemble and --input-files).")
    parser.add_argument("--csv-input-col", help="Column in CSV file where input files are "
                        "are specified, can be either a string or integer (1-based).")
    parser.add_argument("--csv-pred-col", help="Column in CSV file where prediction files are "
                        "are specified, can be either a string or integer (1-based).")

    # input model for pre-processing from file
    parser.add_argument('--input-pre-model', help="Input model .pkl file to run before main model.")

    # input model from file
    parser.add_argument("--input-model", help="Input proxy model from .pkl file.")

    # reduction argument
    parser.add_argument("--num-recon-images", type=int, default=5, help="Number of images "
                        "to reconstruct using reduction algorithm (only applies to algorithms"
                        "that have inverse transforms.")

    # model runs on certain time steps
    parser.add_argument("--test", nargs=2, help="Test proxy model using " + \
                        "--input-model, with %%d[::] to specify time steps to use " + \
                        "followed by number of future time steps to predict.")

    # output arguments and options
    parser.add_argument("--output-file", help="Base file name for output of validation plots. " + \
                        "Multiple plots use numbered suffixes.  Any provided extension " +
                        "will be ignored (but not removed).")

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

    # must be in either training mode or testing mode
    if args.reduction and args.proxy or \
       not args.reduction and not args.proxy:
        log.error("Validation code requires either --reduction or --proxy but not both.")
        sys.exit(1)

    # reduction and proxy both require --input-model
    if args.input_model is None:
        log.error("Input model is required.  Please use --input-model and try again.")
        sys.exit(1)

    # output file is also required
    if args.output_file is None:
        log.error("Output file is required.  Please use --output-file and try again.")
        sys.exit(1)

    # ensemble arguments
    if args.csv_file is None:

        # make sure the ensemble argument is present
        if args.ensemble is None:
            log.error("Ensemble directories are required.  Please use --ensemble and try again.")
            sys.exit(1)
        
        # make sure the input argument is present
        if args.input_file is None:
            log.error("Input file is required.  Please use --input-file and try again.")
            sys.exit(1)
    

        # disable args.csv_col (if provided)
        args.csv_input_col = None

    # otherwise check that ensemble arguments are not present
    else:

        if args.ensemble is not None or \
            args.input_file is not None or \
            args.input_pred_file is not None:
            log.error("Please specify either --csv-file or --ensemble, but not both.")
            sys.exit(1)

        # check that csv input column is provided
        if args.csv_input_col is None:
            log.error("Please specify --input-file to provide input files.")
            sys.exit(1)

    # reduction checks
    if args.reduction:

        # check num images to plot
        if args.num_recon_images < 1:
            log.error("Number of images to reconstruct must be >= 1.")
            sys.exit(1)

        # check that pre-model has .pkl extension
        if args.input_pre_model is not None:
            if not args.input_pre_model.endswith('.pkl'):
                log.error("Input pre-model must end with .pkl extension.")
                sys.exit(1)

    # test is returned only for proxy validation
    test = None

    # check proxy arguments
    if args.proxy:

        if args.csv_file is None:

            # check extension of input_files
            if not args.input_file.endswith('.rd.npy'):
                log.error('Input file name must have extension ".rd.npy".')
                sys.exit(1)

            # make sure the input prediction argument is present
            if args.input_pred_file is None:
                log.error("Input prediction file is required.  " + \
                            "Please use --input-pred-file and try again.")
                sys.exit(1)

            # check extension of input_files
            elif not args.input_pred_file.endswith('.px.npy'):
                log.error('Input prediction file name must have extension ".px.npy".')
                sys.exit(1)

            # require testing requirements
            if args.test is None:
                log.error('The "--test %d[::] future" argument must be specified to test ' + \
                            "a proxy model.")
                sys.exit(1)
                
            # make sure first args.test argument is valid
            root, start, stop, step, extension = \
                ensemble.parse_d_format(log, args.test[0])

            if int(args.test[1]) < 1:
                log.error("Test must predict into the future by at least one time step.")
                sys.exit(1)

            # set test arguments to [root, start, stop, future]
            test = [start, stop, step, int(args.test[1])]

        # if not csv file then must specify a prediction column
        elif args.csv_pred_col is None:
                log.error("Please specify --input-pred-file to provide prediction files.")
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
    
    return test

# get simulation file list
def get_sim_files (log, args, ensemble_dirs, ensemble_table):

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

    # check that time counts are uniform
    if min(file_counts) < max(file_counts):
        log.error("Simulations have different number of time step files.  Proxy model " + \
                  "training/testing not performed, no files written.")
        sys.exit(1)

    return sim_files

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
def get_batch (log, plugin, batch_files, time_steps, parallel):
    
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
            log.error("Simulation data has inconsistent matrix sizes.  Aborting " + \
                     "proxy model training, no files written.")
            sys.exit(1)

    if time_steps is not None:

        # get indices of training data with specified time steps
        train_inds = np.arange(time_steps[0], time_steps[1], time_steps[2])

        # reduced training data
        return [data[i][train_inds,:] for i in range(len(data))]

    else:

        return data

# save variance explained plot, returns true if plot was saved, false otherwise
def plot_var_explained(log, args, fig, algorithm, plot_num, title_str):

    # get percent variance captured per dimension, if applicable
    data_explained = algorithm.data_explained()

    if len(data_explained) > 0:

        # plot percent variance captured per dimension per model
        for i in range(len(data_explained)):

            # plot data explained
            x_vals = range(len(data_explained[i]))
            plt.plot(x_vals, data_explained[i], linestyle=":", linewidth=2)

            log.info("Maximum explained variance for time step %d: %f" % (i, data_explained[i][-1]))

        # draw horizontal line at 95%
        plt.plot(x_vals, [95] * len(x_vals), 'g--')

        # label 95% on y-axis
        ninetyfive = [95]
        plt.yticks(list(plt.yticks()[0]) + ninetyfive)

        # show grid
        plt.grid()

        # make labels
        plt.xlabel("Dimension")
        plt.title(title_str)

        # for time-aligned reductions, add explanation for multiple lines
        if algorithm.time_align_dim() is None:
            plt.ylabel("Percent Data Explained")
        else:
            plt.ylabel("Percent Data Explained Per Time Step")

        # save figure
        plt.savefig(args.output_file + "-" + str(plot_num) + ".png")
        log.info("Saved file: %s" % args.output_file + "-" + str(plot_num) + ".png")

        return True
    
    else:

        return False

# validates dimension reduction or proxy models
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

    # start loggera
    romans.init_logger(log_file=args.log_file, log_level=args.log_level)
    log = logging.getLogger("romans.validate")
    log.debug("Started validate.")

    # check arguments
    test = check_arguments(log, args)

    # import and initialize plugin
    plugin, arg_list = romans.plugin(args.plugin, arg_list)

    # initialize figure, keep track of current plot
    fig = plt.figure()
    current_plot = 1

    # are we validating dimension reduction?
    if args.reduction:

        # initialize pre-model, if given
        pre_algorithm = None
        if args.input_pre_model is not None:
            pre_algorithm = reduction.DimensionReduction(arg_list=arg_list,
                                                         model_file=args.input_pre_model)

            # check pre-model arguments
            check_arguments(log, args, algorithm=pre_algorithm)

            # plot variance captured for pre-modle, if applicable
            fig_plotted = plot_var_explained(log, args, fig, pre_algorithm, current_plot,
                "Pre-Model Reduction Effectiveness")

            # if plot was saved, advance plot counter and reset figure
            if fig_plotted:
                current_plot += 1
                plt.clf()

        # initialize dimension reduction algorithm 
        algorithm = reduction.DimensionReduction(arg_list=arg_list,
                                                 model_file=args.input_model)

        # re-check arguments with algorithm initialized
        check_arguments(log, args, algorithm=algorithm)

        # plot variance captured for model, if applicable
        fig_plotted = plot_var_explained(log, args, fig, algorithm, current_plot, 
            "Dimension Reduction Effectiveness")

        # if plot was saved, advance plot counter and reset figure
        if fig_plotted:
            current_plot += 1
            plt.clf()

        # check for pre-algorithm with inverse
        pre_algorithm_has_inverse = False
        if pre_algorithm is not None:
            pre_algorithm_has_inverse = pre_algorithm.has_inverse()

        # show reconstructions, if available
        if algorithm.has_inverse():

            # if pre-algorithm exists, check that it has an inverse
            if pre_algorithm is None or \
                pre_algorithm_has_inverse:

                # create ensemble table with reduced files
                ensemble_table = ensemble.Table(log, csv_file=args.csv_file,
                    ensemble_spec=args.ensemble, file_spec=args.input_file, 
                    header="validate")

                # get ensemble directories/file specs
                if args.csv_file is not None:
                    ensemble_dirs = ensemble_table.get_col(args.csv_input_col)
                else:
                    ensemble_dirs = ensemble_table.get_col("validate")

                # quit if no directories found
                num_ensemble_dirs = len(ensemble_dirs)
                if num_ensemble_dirs == 0:
                    log.error("No ensemble directories found.  " +
                            "Please identify existing directories and try again.")
                    sys.exit(1)
                else:
                    log.info("Found %d ensemble directory(ies)." % num_ensemble_dirs)

                # get reduced data simulation files, in batches
                sim_files = get_sim_files (log, args, ensemble_dirs, ensemble_table)
                
                # pick random simulations, at random time steps
                num_sim_files = len(sim_files)
                num_time_steps = len(sim_files[0])

                # might not be any actual sim files
                if num_time_steps == 0:
                    log.error("No simulation files found.  Please provide simulation files and try again.")
                    sys.exit(1)
                    
                rand_sim_inds = np.random.permutation(num_sim_files)[0:args.num_recon_images]
                rand_time_inds = np.random.permutation(num_time_steps)[0:args.num_recon_images]
                
                # add repeated time inds if not enough time steps
                if len(rand_time_inds) < args.num_recon_images:
                    rand_time_inds = np.repeat(rand_time_inds, args.num_recon_images)

                # get random sim files
                rand_sim_files = [sim_files[rand_sim_inds[i]][rand_time_inds[i]] 
                    for i in range(args.num_recon_images)]

                # get simulation data
                sim_data = get_batch(log, plugin, rand_sim_files, None, args.parallel)
                image_width, image_height = sim_data[0].shape

                # re-shape data into vectors for dimension reduction
                sim_data_vec = np.asarray([np.ndarray.flatten(sim_data[i]) 
                    for i in range(args.num_recon_images)])

                if algorithm.time_align_dim() is not None:

                    # pre-algorithm
                    if pre_algorithm_has_inverse:
                        sim_data_vec = np.asarray([pre_algorithm.transform(sim_data_vec[i],
                            time_step=rand_time_inds[i]) for i in range(args.num_recon_images)])

                    # main algorithm
                    sim_data_reduced = np.asarray([algorithm.transform(sim_data_vec[i],
                        time_step=rand_time_inds[i]) for i in range(args.num_recon_images)])
                    sim_data_recon = np.asarray([algorithm.inverse_transform(sim_data_reduced[i],
                        time_step=rand_time_inds[i]) for i in range(args.num_recon_images)])

                    # post-algorithm
                    if pre_algorithm_has_inverse:
                        sim_data_recon = np.asarray([pre_algorithm.inverse_transform(sim_data_recon[i],
                            time_step=rand_time_inds[i]) for i in range(args.num_recon_images)])
                    
                else:

                    # pre-algorithm
                    if pre_algorithm_has_inverse:
                        sim_data_vec = pre_algorithm.transform(sim_data_vec)

                    # main algorithm
                    sim_data_reduced = algorithm.transform(sim_data_vec)
                    sim_data_recon = algorithm.inverse_transform(sim_data_reduced)
                    
                    # post-algorithm
                    if pre_algorithm_has_inverse:
                        sim_data_recon = pre_algorithm.inverse_transform(sim_data_recon)

                # plot images
                plt.clf()
                fig.set_size_inches(args.num_recon_images * 2, 4)

                # plot original images
                for i in range(args.num_recon_images):
                    plt.subplot(2, args.num_recon_images, i+1)
                    fig = plt.imshow(np.reshape(sim_data[i], (image_width, image_height)))

                    # hide ticks/tick labels
                    fig.axes.get_xaxis().set_ticklabels([])
                    fig.axes.get_yaxis().set_ticklabels([])
                    fig.axes.get_xaxis().set_ticks([])
                    fig.axes.get_yaxis().set_ticks([])

                    # first row is original images
                    if i==0:
                        fig.axes.set_ylabel("Original")
                    fig.axes.title.set_text(str(rand_sim_inds[i]) + ", " + str(rand_time_inds[i]))
                
                # plot reconstructed images
                for i in range(args.num_recon_images):
                    plt.subplot(2, args.num_recon_images, i+args.num_recon_images+1)
                    fig = plt.imshow(np.reshape(sim_data_recon[i], (image_width, image_height)))

                    # hide ticks/tick labels
                    fig.axes.get_xaxis().set_ticklabels([])
                    fig.axes.get_yaxis().set_ticklabels([])
                    fig.axes.get_xaxis().set_ticks([])
                    fig.axes.get_yaxis().set_ticks([])

                    # second row is reconstructed images
                    if i==0:
                        fig.axes.set_ylabel("Reconstruction")

                # save to file
                plt.savefig(args.output_file + "-2.png")
                log.info("Saved file: %s" % args.output_file + "-2.png")

    # are we validating a proxy model
    if args.proxy:
        
        # initialize proxy model algorithm 
        algorithm = proxy.ProxyModel(arg_list=arg_list,
                                     model_file=args.input_model)

        # re-check arguments with algorithm initialized
        check_arguments(log, args, algorithm=algorithm)

        # create ensemble table
        ensemble_table = ensemble.Table(log, csv_file=args.csv_file,
            ensemble_spec=args.ensemble, file_spec=args.input_file, 
            header="validate")

        # get ensemble directories/file specs
        if args.csv_file is not None:
            ensemble_dirs = ensemble_table.get_col(args.csv_input_col)
        else:
            ensemble_dirs = ensemble_table.get_col("validate")

        # quit if no directories found
        num_ensemble_dirs = len(ensemble_dirs)
        if num_ensemble_dirs == 0:
            log.error("No ensemble directories found.  " +
                    "Please identify existing directories and try again.")
            sys.exit(1)
        else:
            log.info("Found %d ensemble directory(ies)." % num_ensemble_dirs)

        # get reduced data simulation files, in batches
        sim_files = get_sim_files (log, args, ensemble_dirs, ensemble_table)
        batch_files = get_batch_files (sim_files, args.file_batch_size)
        num_batches = len(batch_files)

        # create prediction table
        pred_table = ensemble.Table(log, csv_file=args.csv_file,
            ensemble_spec=args.ensemble, file_spec=args.input_pred_file, 
            header="preds")

        # get prediction directories/file specs
        if args.csv_file is not None:
            pred_dirs = pred_table.get_col(args.csv_pred_col)
        else:
            pred_dirs = pred_table.get_col("preds")

        pred_files = get_sim_files (log, args, pred_dirs, pred_table)
        pred_batches = get_batch_files (pred_files, args.file_batch_size)

        # make sure prediction files exist
        if pred_files[0] == []:
            log.error("Files " + args.input_pred_file + " does not exist.  Please supply files " + \
                "and try again.")
            sys.exit(1)

        # indices of ground truth values (start, stop, step)
        data_inds = [test[1], test[1] + test[3], test[2]]

        # indices of predictions (start, stop, step)
        pred_inds = [0, test[3], 1]

        # load prediction and test data, in batches
        data = []
        preds = []
        for i in range(num_batches):

            # load files into memory, store as a list of matrices
            # with num_time_steps x num_components, one per simulation
            data_i = get_batch(log, plugin, batch_files[i], data_inds, args.parallel)
            
            # load predictions
            preds_i = get_batch(log, plugin, pred_batches[i], pred_inds, args.parallel)

            # keep only last time step and predicted number of dimensions
            num_dim = preds_i[0].shape[1]
            data += [data_i[j][-1,0:num_dim] for j in range(len(data_i))]
            preds += [preds_i[j][-1,:] for j in range(len(preds_i))]
        
        # convert to numpy arrays
        data_dim = np.vstack(data)
        preds_dim = np.vstack(preds)
        
        # plot loss (up to 4 dimensions)
        loss = algorithm.loss()
        colors = ['blue', 'orange', 'green', 'red']
        for i in range(min(len(loss),4)):
            plt.plot(loss[i], label="Dimension " + str(i+1), 
                linestyle=':', linewidth=2, color=colors[i])
        plt.legend()
        plt.ylabel('MSE Loss')
        plt.xlabel("Optimization Progress")
        plt.title('Proxy Model Training')
        
        # save plot to output-file
        plt.savefig(args.output_file + "-1.png")
        log.info("Saved file: %s" % args.output_file + "-1.png")
        
        # plot actual data versus predictions (use last time step)
        plt.clf()
        for i in range(min(num_dim,4)):
            plt.subplot(2,2,i+1)
            plt.scatter(data_dim[:,i], preds_dim[:,i], color=colors[i],
                label="Dimension " + str(i+1))
            plt.legend()
            
            # put titles only on top
            if i==0 or i==1:
                plt.title("Proxy Model Error")
            # put x-labels only on bottom
            if i==2 or i==3:
                plt.xlabel("Actual")
            
            # put y-labels only on left
            if i==0 or i==2:
                plt.ylabel('Predicted')

        # save figure
        plt.savefig(args.output_file + "-2.png")
        log.info("Saved file: %s" % args.output_file + "-2.png")

# entry point for command line call
if __name__ == "__main__":
    main()