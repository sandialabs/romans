# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains all of the available algorithms for
# training and testing reduced order proxy models.

# S. Martin
# 6/10/2021

# standard library imports

# command line arguments
import argparse

# logging and error handling
import logging
import warnings

# loading/saving models
import pickle

# LSTM
import torch
import numpy as np

# list of available reduced order proxy algorithms
ALGORITHMS = ["LSTM"]
INCREMENTAL = [True]

# LSTM training defaults
LEARNING_RATE = 0.8
HIDDEN_SIZE = 80
NUM_STEPS = 15

class Sequence(torch.nn.Module):
    """
    This class contains the torch functions necessary to 
    use a four cell LSTM network to predict sequences.
    """

    # set up network
    def __init__(self):

        super().__init__()
        
        self.lstm1 = torch.nn.LSTMCell(1, HIDDEN_SIZE)
        self.lstm2 = torch.nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.lstm3 = torch.nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.lstm4 = torch.nn.LSTMCell(HIDDEN_SIZE, HIDDEN_SIZE)
        self.linear = torch.nn.Linear(HIDDEN_SIZE, 1)
        
    # feed data through network
    def forward(self, input, future=0):

        # append outputs per time step to list
        outputs = []

        # initialize inputs to LSTM cells
        h_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        c_t = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        h_t2 = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        c_t2 = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        h_t3 = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        c_t3 = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        h_t4 = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        c_t4 = torch.zeros(input.size(0), HIDDEN_SIZE, dtype=torch.double)
        
        # split input into columns (time axis)
        for input_t in input.split(1, dim=1):

            # feed input through network
            h_t, c_t = self.lstm1(input_t, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))

            # save output
            output = self.linear(h_t2)
            outputs += [output]

        # continue propagating data into the future
        for i in range(future):

            # feed last output through network
            h_t, c_t = self.lstm1(output, (h_t, c_t))
            h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
            h_t3, c_t3 = self.lstm3(h_t2, (h_t3, c_t3))
            h_t4, c_t4 = self.lstm4(h_t3, (h_t4, c_t4))

            # save output
            output = self.linear(h_t2)
            outputs += [output]

        # combine outputs
        outputs = torch.cat(outputs, dim=1)

        return outputs


class LSTM:
    """
    The class mplements the basic functions expected from 
    the Proxy class for an LSTM, including train and test.
    """

    # initialize auto-encoder with user parameters, or defaults
    def __init__(self, log, num_steps=NUM_STEPS, 
            learning_rate=LEARNING_RATE):

        # set defaults
        self.learning_rate = learning_rate
        self.num_steps = num_steps

        # use log from calling routine
        self.log = log

        # set up simple sequential LSTM
        self.LSTM = Sequence()
        self.LSTM.double()

        # save loss as optimization progresses
        self.loss = []

    # retrieve optimization loss vector
    def optimization_loss(self):
        return self.loss

    # train model, returns loss over optimization
    def train(self, data):

        # construct input and target
        input = torch.from_numpy(data[:, :-1])
        target = torch.from_numpy(data[:, 1:])

        # use mean square error loss
        criterion = torch.nn.MSELoss()

        # use LBFGS as optimizer since we can load the whole data to train
        optimizer = torch.optim.LBFGS(self.LSTM.parameters(), lr=self.learning_rate)

        # train LSTM in steps
        for i in range(self.num_steps):

            self.log.info("Training LSTM step %d." % i)

            # define closure function for LBFGS optimizer
            def closure():
                optimizer.zero_grad()
                out = self.LSTM(input)
                loss = criterion(out, target)
                self.loss.append(loss.item())
                self.log.info("Loss is %f." % self.loss[-1])
                loss.backward()
                return loss

            # do optimization step
            optimizer.step(closure)

    # test model, returns predictions
    def test(self, data, future):

        # construct input
        input = torch.from_numpy(data)

        # use mean square error loss
        criterion = torch.nn.MSELoss()

        # don't compute gradient
        with torch.no_grad():
            predictions = self.LSTM(input, future=future)

        return predictions[:,-future:].detach().numpy()

class ProxyModel:
    """
    This class contains the algorithms for trainig and testing reduced order
    proxy models using reduced dimension data.  It includes it's own parser 
    to specify algorithms and algorithm parameters.

    Args:
        arg_list (list): list of arguments to specific to proxy model
        model_file (string): name of model file containing proxy model

    :Example:

    .. code-block:: python

        # get parser and algorithms
        import romans
        import algorithms.proxy as algorithms

        # parse command line
        my_parser = romans.ArgumentParser()

        # parse command line and start logger
        args, arg_list = my_parser.parse_args()

        # set up proxy model algorithm using command line arguments
        algorithm = algorithms.ProxyModel(arg_list=arg_list)

        # set up data in variable X as list of reduced dimension
        # matrices, one matrix per simulation

        # train proxy model reduction
        algorithm.train(X)

        # get training losses per dimension
        print(algorithm.loss())

    """

    # parse proxy model algorithm specific arguments
    def __init__(self, arg_list=None, model_file=None):

        # create parser
        self._init_parser()

        # start logger
        self.log = logging.getLogger("romans.algorithms.reduction")
        self.log.debug("Started dimension reduction.")

        # parse known arguments, keep unknown arguments
        args, arg_list = self.parser.parse_known_args(arg_list)
        self.unknown_arglist = arg_list

        # check for model inputs passed as arguments
        parms_present = [True for key in vars(args).keys() \
                if vars(args)[key] is not None]
        parms_present = len(parms_present) > 0

        # check that arg_list or model_file is provided
        if not parms_present and model_file is None:
            self.log.error("Must provide either model input file or model arguments to " +
                "perform proxy model training/testing.")
            raise ValueError("must provide either model input file or model arguments to " +
                "perform proxy model training/testing.")

        # check that both arg_list and model_file are not provided
        if parms_present and model_file is not None:
            self.log.error("Use either a model input file or model arguments, but not both.")
            raise ValueError("use either a model input file or model arguments, but not both.")

        # initialize model as list, one per reduced dimension
        self.model = []

        # initialize model parameters
        self.model_parms = {}
        self.model_parms["algorithm"] = None

        # initialize output parameters
        self.model_outputs = {}
        self.model_outputs['loss'] = []

        # parse & check arguments, if given
        if parms_present:
            
            # check arguments
            self._check_args(args)

            # init model
            self._set_parms(args)
            self._init_model()

        # load a model, if requested
        if model_file is not None:
            self.load(model_file)

    # init parser
    def _init_parser(self):

        # includes parser for dimension reduction options
        description = "Reduced order proxy model support for the romans tools."
        self.parser = argparse.ArgumentParser(description=description)

        # type of algorithm
        self.parser.add_argument("--algorithm", choices=ALGORITHMS, help=
            "Proxy model algorithm to apply.  Options are: {%s}." % 
            ", ".join(ALGORITHMS))
        
        # LSTM arguments
        self.parser.add_argument("--LSTM-steps", type=int, help="Number of "
            "steps to use in LSTM optimization.")

    # check parameters for valid values
    def _check_args(self, args):

        # check if algorithm is present
        if args.algorithm is None:
            self.log.error("Algorithm choice is required for proxy model.  Please use " +
                           "--algorithm and try again.")
            raise ValueError("algorithm choice is required for proxy model.  Please use " +
                             "--algorithm or a model input file and try again.")
        
        if args.LSTM_steps is not None:
            if args.LSTM_steps < 1:
                self.log.error("LSTM optimizer requires at least one step.")
                raise ValueError("LSTM optimizer requires at least one step.")

    # set parameters using arguments
    def _set_parms(self, args):

        # general arguments
        self.model_parms["algorithm"] = args.algorithm

        # LSTM arguments
        self.model_parms["LSTM-steps"] = args.LSTM_steps

    # init model according to arguments
    def _init_model(self):
        pass

    # init algorithm, per dimension
    def _init_algorithm(self):

        if self.model_parms['algorithm'] == 'LSTM':
            self.model.append(LSTM(self.log, 
                num_steps=self.model_parms["LSTM-steps"]))
            self.model_outputs['loss'].append([])

    # get any unrecognized arguments
    def unknown_args(self):
        return self.unknown_arglist

    # return optimization losses
    def loss(self):
        """
        Return optimization loss.

        Returns:
            loss (list of arrays): loss over optimization, per dimension
        """

        return self.model_outputs['loss']

    # train model
    def train(self, data):
        """
        Train a proxy model, using reduced dimension data.

        Args:
            data (list of arrays): list of data matrices per simulation
        """

        # make dataset into 3D array
        data = np.stack(data)
        num_dim = data.shape[2]

        # check for new model
        new_model = False
        if len(self.model) == 0:
            self.model_parms['num_dim'] = num_dim
            new_model = True

        # if existing model, check that number dimensions match
        elif len(self.model) != num_dim:
            self.log.error('Existing proxy model has different number of dimensions.')
            raise ValueError('existing proxy model has different number of dimension.')

        # train a model for each dimension
        for i in range(self.model_parms['num_dim']):

            self.log.info("Training model for dimension %d." % i)

            # set up model, unless it already exists
            if new_model:
                self._init_algorithm()

            # train model using sequences of time steps
            self.model[i].train(data[:,:,i])

            # save optimization loss for model, if present
            self.model_outputs['loss'][i] = self.model[i].optimization_loss()

    # test model
    def test(self, data, future):
        """
        Test a proxy model, by producing predictions from data.

        Args:
            data (list of arrays): list of data matrices per simulation
            future (int): number of time steps to predict past end of data

        Returns:
            predicted_data (list of arrays): list of predicted data matrices
        """

        # make dataset into 3D array
        data = np.stack(data)

        # save predicted data
        num_sim = data.shape[0]
        predicted_data = np.zeros((num_sim, future, self.model_parms['num_dim']))

        # prediction using a model for each dimension
        for i in range(self.model_parms['num_dim']):

            self.log.info("Computing predictions from model for dimension %d." % i)

            # train model using sequences of time steps
            predicted_data[:,:,i] = self.model[i].test(data[:,:,i], future)
    
        # convert data back to list per simulation
        per_sim_data = [predicted_data[i,:,:] for i in range(num_sim)]

        return per_sim_data

    # save proxy model
    def save(self, file_out):
        """
        Saves a proxy model to a .pkl file.

        Args:
            file_out (string): file name to save file
        """

        with open(file_out, 'wb') as handle:
            pickle.dump([self.model, self.model_parms, self.model_outputs], handle)

        self.log.info("Saved model file to %s." % file_out)

    # load proxy model
    def load(self, file_in):
        """
        Loads a proxy model from a .pkl file.

        Args:
            file_in (string): file name with saved model
        """

        # load model and parameters
        with open(file_in, 'rb') as handle:
            self.model, self.model_parms, self.model_outputs = pickle.load(handle)

        self.log.info("Loaded model file from %s." % file_in)


# if called from the command line display algorithm specific command line arguments
if __name__ == "__main__":

    # show options on command line if requested
    algorithms = ProxyModel()
    algorithms.parser._parse_args()
