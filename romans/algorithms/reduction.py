# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains all of the available algorithms for
# performing dimension reduction using the romans tools.

# This code serves mainly as a wrapper for algorithms available
# from other libraries, such as sklearn and pytorch.

# S. Martin
# 2/17/2021


# standard library imports

# command line arguments
import argparse

# logging and error handling
import logging
import warnings

# model save/load
import joblib

# 3rd party imports

# matrix manipulations
import numpy as np

# pre-processing
import sklearn.preprocessing as preprocessing

# dimension reduction
import sklearn.decomposition as decomposition
import sklearn.manifold as manifold

# auto-encoder
import torch
import torch.multiprocessing as mp

# umap
import umap

# comparing runing times (for debugging)
import time
from datetime import timedelta

# list of available preprocessing and dimension reduction algorithms
PREPROCESS = ["standard", "minmax"]
ALGORITHMS = ["PCA", "incremental-PCA", "Isomap", 
              "tSNE", "auto-encoder", "Umap"]
INCREMENTAL = [False, True, False, 
               False, True, False, False]
INVERSE = [True, True, False,
           False, True, False, True]

# types of autoencoders
MODEL_TYPES = ["MLP", "var"]

# number of default dimensions for reduction
NUM_DIM = 2

# auto-encoder training defaults
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
MLP_ARCH = [512,128]
NUM_PROCESSES = 1


class TorchDataset(torch.utils.data.Dataset):
    """
    This class contains a torch dataset constructor that converts a numpy
    array with rows as points to a torch dataset for use with the 
    auto-encoder.

    :Example:

    .. code-block:: python

        # initialize a random dataset, 100 data points in 10 dimensions
        random_data = numpy.random.rand (100,10)

        # convert to a torch dataset
        torch_data = TorchDataset(random_data)

        # put data in a torch dataloader using a batch size of 10
        data_loader = torch.utils.data.DataLoader(torch_data, batch_size=10, shuffle=True)

    """

    # stores a numpy array as a torch tensor
    def __init__(self, X):        
        self.X = torch.from_numpy(X)

    # number of samples
    def __len__(self):
        return self.X.size()[0]

    # get item
    def __getitem__(self, index):
        return self.X[index,:]


class MLP(torch.nn.Module):
    """
    This class contains a basic MLP auto-encoder that can be used for dimension 
    reduction.  The class implements only the pytorch architecture for a model.
    """

    # initialize the network using the input dimension and bottleneck dimension
    def __init__(self, input_dim, bottleneck_dim, arch=MLP_ARCH):
        
        super().__init__()

        # keep track of network architecture
        self.num_inputs = input_dim
        self.num_dim = bottleneck_dim

        # encoder/decoder layers
        self.encoder_arch = [self.num_inputs] + arch
        self.decoder_arch = [self.num_dim] + arch[::-1]
        self.num_layers = len(self.encoder_arch) - 1

        # set up architecture
        self._init_encoder()
        self._init_decoder()

    # define encoder layers (num_inputs -> MLP -> num_dim)
    def _init_encoder(self):

        # append hidden layers
        self.encoder_layers = torch.nn.ModuleList()
        for k in range(self.num_layers):
            self.encoder_layers.append(
                torch.nn.Linear(self.encoder_arch[k], self.encoder_arch[k+1]))

        # bottleneck layer
        self.bottleneck_layer = torch.nn.Linear(self.encoder_arch[-1], self.num_dim)

    # define decoder layers (num_dim -> reverse MLP -> num_inputs)
    def _init_decoder(self):        

        # append hidden layers
        self.decoder_layers = torch.nn.ModuleList()
        for k in range(self.num_layers):
            self.decoder_layers.append(
                torch.nn.Linear(self.decoder_arch[k], self.decoder_arch[k+1]))
        
        # reconstruction layer
        self.reconstruction_layer = torch.nn.Linear(self.decoder_arch[-1], self.num_inputs)

    # execute encoder
    def _encoder(self, data):

        # initial encoder layers
        for layer in self.encoder_layers:
            data = torch.nn.functional.relu(layer(data))

        # bottleneck layer
        bottleneck = self.bottleneck_layer(data)

        return bottleneck

    # execute decoder
    def _decoder(self, data):

        # decoder
        for layer in self.decoder_layers:
            data = torch.nn.functional.relu(layer(data))

        return torch.tanh(self.reconstruction_layer(data))

    # auto-encoder architecture definition for pytorch
    def forward(self, data):

        # encode-decode
        bottleneck = self._encoder(data)
        reconstruction = self._decoder(bottleneck)

        return reconstruction

    # return reduced data using encoder on numpy array
    def encoder(self, data):

        # convert numpy array to torch format
        torch_data = torch.from_numpy(data.astype(np.float32))

        # get bottleneck layer
        bottleneck = self._encoder(torch_data)

        # convert back to numpy array
        return bottleneck.data.numpy()

    # return reconstructed data using decoder on numpy array
    def decoder(self, data):

        # convet numpy array to torch format
        torch_data = torch.from_numpy(data.astype(np.float32))

        # reconstruct data using decoder
        recon = self._decoder(torch_data)

        # convert back to numpy array
        return recon.data.numpy()


# variational autoencoder architecture
class VAE(torch.nn.Module):
    """
    This class contains a basic variational auto-encoder that can be used for dimension 
    reduction.  The class implements only the pytorch architecture for a model.
    """

    # initialize network using input, bottleneck dimension
    def __init__(self, input_dim, bottleneck_dim):

        super().__init__()

        # network architecture
        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # set up architecture
        self._init_encoder()
        self._init_decoder()

    # encoder architecture
    def _init_encoder(self):

        self.fc1 = torch.nn.Linear(self.input_dim, 500)
        self.fc21 = torch.nn.Linear(500, self.bottleneck_dim)
        self.fc22 = torch.nn.Linear(500, self.bottleneck_dim)
    
    # decoder architecture
    def _init_decoder(self):

        self.fc3 = torch.nn.Linear(self.bottleneck_dim, 500)
        self.fc4 = torch.nn.Linear(500, self.input_dim)

    # execute encoder (using torch data)
    def _encode(self, x):

        h1 = torch.nn.functional.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    # execute decoder (using torch data)
    def _decode(self, z):
    
        h3 = torch.nn.functional.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    # reparameterize for variational layer
    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    # execute auto-encoder (torch data)
    def forward(self, x):
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        return self._decode(z), mu, logvar

    # return reduced data using encoder on numpy array
    def encoder(self, data):

        # convert numpy array to torch format
        torch_data = torch.from_numpy(data.astype(np.float32))

        # get bottleneck layer (variational)
        mu, logvar = self._encode(torch_data)

        # re-parameterize
        bottleneck = self._reparameterize(mu, logvar)

        # convert back to numpy array
        return bottleneck.data.numpy()

    # return reconstructed data using decoder on numpy array
    def decoder(self, data):

        # convet numpy array to torch format
        torch_data = torch.from_numpy(data.astype(np.float32))

        # reconstruct data using decoder
        recon = self._decode(torch_data)

        # convert back to numpy array
        return recon.data.numpy()


class AutoEncoder(torch.nn.Module):
    """
    This class is a factory class defines and trains a pytorch auto-encoder model.
    The class implements the basic functions expected from the DimensionReduction 
    class, including fit, partial_fit, and transform.
    """

    # initialize auto-encoder with user parameters
    def __init__(self, log, num_dim=NUM_DIM, batch_size=None,
            learning_rate=LEARNING_RATE, epochs=None, num_processes=NUM_PROCESSES,
            model_type='MLP'):

        super().__init__()

        # number of encoder output neurons
        self.num_dim = num_dim

        # set default batch size
        self.batch_size = batch_size
        if self.batch_size is None:
            self.batch_size = BATCH_SIZE

        # learning rate for training
        self.learning_rate = learning_rate

        # set default epochs
        self.epochs = epochs
        if self.epochs is None:
            self.epochs = EPOCHS

        # use multiprocessing
        self.num_processes = num_processes
        if self.num_processes is None:
            self.num_processes = NUM_PROCESSES

        # use calling function log for output
        self.log = log

        # model type for autoencoder
        self.model_type = model_type

        # actual autencoder model has not been initialized
        self.model = None

    # variational reconstruction + KL divergence losses summed over all elements and batch
    def _var_loss(self, recon_x, x, mu, logvar):

        BCE = torch.nn.functional.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

    # train auto-encoder (can be called in parallel)
    def _train(self, rank, model, model_type, device, data_loader, 
        optimizer, scheduler, epochs):

        # for MLP use mean square error loss
        if model_type == 'MLP':
            criterion = torch.nn.MSELoss()

        # train auto-encoder
        model.train()
        for epoch in range(epochs):
            for batch_idx, data in enumerate(data_loader):

                # zero gradients
                optimizer.zero_grad()

                # re-construct data & compute loss
                if model_type == 'MLP':
                    recon = model(data.to(device))
                    loss = criterion(recon.to(device), data)
                elif model_type == 'var':
                    recon, mu, logvar = model(data.to(device))
                    loss = self._var_loss(recon.to(device), data, mu, logvar)
                
                # do back-propagation
                loss.backward()
                optimizer.step()

                # using print because we are in a thread
                if batch_idx % 10 == 0:
                    print('Process {} -- Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                        rank, epoch, batch_idx * len(data), len(data_loader.dataset),
                        100. * batch_idx / len(data_loader), loss.item()))

            # adjust optimization parameters
            scheduler.step(loss.item())

    # define and fit auto-encoder
    def fit(self, data):

        # check batch size
        num_data, num_inputs = data.shape
        if self.batch_size > num_data:
            warnings.warn("Batch size exceeds number of data points -- resetting to " +
                          "number data points.")
            self.batch_size = num_data

        # convert data to torch format
        torch_data = TorchDataset(data.astype(np.float32))
        
        # set up encoder/decoder layers/optimizer
        if self.model is None:

            if self.model_type == 'MLP':
                self.model = MLP(num_inputs, self.num_dim)
            
            elif self.model_type == 'var':
                self.model = VAE(num_inputs, self.num_dim)

            # use Adam optimizer
            self.optimizer = torch.optim.AdamW(self.parameters(),
                                               lr=self.learning_rate,
                                               weight_decay=1e-5)

            # use reduce on stall scheduler
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=50)

        # use CUDA, if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # share model between processes
        self.model.to(device)
        self.model.share_memory()

        # time training 
        t0 = time.time()

        processes = []
        for rank in range(self.num_processes):

            # partition data per process
            data_loader = torch.utils.data.DataLoader(
                dataset=torch_data,
                sampler=torch.utils.data.distributed.DistributedSampler(
                    dataset=torch_data,
                    num_replicas=self.num_processes,
                    rank=rank
                ),
                batch_size=self.batch_size,
            )

            # launch processes
            process = mp.Process(target=self._train, 
                args=(rank, self.model, self.model_type, device, 
                    data_loader, self.optimizer, self.scheduler, self.epochs))
            process.start()
            processes.append(process)

        # collect results
        for process in processes:
            process.join()

        self.log.info("Training Time: %s" % str(timedelta(seconds=time.time()-t0)))

    # perform dimension reduction
    def transform(self, data):
        
        # get reduced data from encoder
        return self.model.encoder(data)

    # inverse transofmr
    def inverse_transform(self, data):

        recon = self.model.decoder(data)

        return recon


class DimensionReduction:
    """
    This class contains wrappers for dimension reduction algorithms in sci-kit
    learn and pytorch for use with the romans tools.  It includes it's own parser 
    to specify algorithms and algorithm parameters.

    Args:
        arg_list (list): list of arguments to specific to reduction
        model_file (string): name of model file containing reduction

    :Example:

    .. code-block:: python

        # get parser and reduction algorithm code
        import romans
        import algorithms.reduction as algorithms

        # parse command line
        my_parser = romans.ArgumentParser()

        # parse command line and start logger
        args, arg_list = my_parser.parse_args()

        # set up dimension reduction algorithm using command line arguments
        algorithm = algorithms.DimensionReduction(arg_list=arg_list)

        # use time_align to use a time-aligned model, where time_align 
        # specifies the number of dimension to use per time step
        time_aligned_algorithm = algorithms.DimensionReduction(time_align=10)

        # set up data in variable X, data points per row

        # do dimension reduction (ala sklearn)
        algorithm.fit(X)

        # reduced data to lower dimension
        reduced_data = algorithm.transform(X)

    """

    # set up dimension reduction algorithm specific arguments
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
                "perform dimension reduction.")
            raise ValueError("must provide either model input file or model arguments to " +
                "perform dimension reduction.")

        # check that both arg_list and model_file are not provided
        if parms_present and model_file is not None:
            self.log.error("Use either a model input file or model arguments, but not both.")
            raise ValueError("use either a model input file or model arguments, but not both.")

        # initialize model, use list for time aligned case
        self.model = []
        self.pre_process = []

        # initialize model parameters
        self.model_parms = {}
        self.model_parms["num_dim"] = None
        self.model_parms["algorithm"] = None
        self.model_parms["pre_process"] = None

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
        description = "Dimension reduction support for the romans tools."
        self.parser = argparse.ArgumentParser(description=description)

        # type of algorithm
        self.parser.add_argument("--algorithm", choices=ALGORITHMS, help=
            "Dimension reduction algorithm to apply.  Options are: {%s}." % 
            ", ".join(ALGORITHMS))

        # type of pre-processing
        self.parser.add_argument("--pre-process", choices=PREPROCESS, help=
            "Preprocessing for dimension reduction.  Options are: {%s}." %
            ", ".join(PREPROCESS))

        # number of dimensions (required for any dimension reduction algorithm)
        self.parser.add_argument("--num-dim", type=int, help="Number of desired "
            "dimensions in reduction.")

        # time aligned option
        self.parser.add_argument("--time-align", type=int, help="Train reduction model " 
            "per time step to given dimension then align using Kabsch algorithm.")

        # PCA parameters
        self.parser.add_argument('--whiten', action="store_true", default=None,
            help="Whiten before PCA.")

        # auto-encoder parameters

        # model type
        self.parser.add_argument("--model-type", choices=MODEL_TYPES,
            help="Type of auto-encoder.  Options are: {%s}." % ", ".join(MODEL_TYPES))

        # number of epochs
        self.parser.add_argument("--epochs", type=int, help="Number of epochs "
            "to use for training auto-encoder.")

        # batch size
        self.parser.add_argument("--batch-size", type=int, help="Batch size "
            "for training auto-encoder.")

        # MLP architecture
        self.parser.add_argument('--MLP-arch', type=int, nargs='+', help="Integers specifying "
            "size of hidden layers in the MLP.")

        # number of processors
        self.parser.add_argument("--num-processes", type=int, help="Number of processes to "
            "use for training auto-encoder.")

    # check parameters for valid values
    def _check_args(self, args):

        # check if algorithm is present
        if args.algorithm is None:
            self.log.error("Algorithm choice is required for dimension reduction.  Please use " +
                           "--algorithm and try again.")
            raise ValueError("algorithm choice is required for dimension reduction.  Please use " +
                             "--algorithm or a model input file and try again.")

        # check if number of dimensions exists
        if args.num_dim is None:
            self.log.error("Number of dimensions in reduction must be provided.")
            raise ValueError("number of dimensions in reduction must be provided.")
        
        # check if number of dimensions >= 2
        else:
            if args.num_dim < 2:
                self.log.error("Number of dimensions in reduction must be integer >= 2.")
                raise ValueError("number of dimensions in reduction must be integer >= 2.")

        # check to time aligned >= 2
        if args.time_align is not None:
            if args.time_align < 2:
                self.log.error("Time aligned dimnension must be integer >= 2.")
                raise ValueError("time algined dimension must be integer > = 2.")

        # check for specific auto-encoder arguments
        if args.algorithm == "auto-encoder":

            # check epochs >= 1
            if args.epochs is not None:
                if args.epochs < 1:
                    self.log.error("Epochs must be >= 1.")
                    raise ValueError("epochs must be >= 1.")
            
            # check batch-siae >= 1
            if args.batch_size is not None:
                if args.batch_size < 1:
                    self.log.error("Batch size must be >= 1.")
                    raise ValueError("batch size must be >= 1.")

            # check number of process is >= 1
            if args.num_processes is not None:
                if args.num_processes < 1:
                    self.log.error("Number of processes must be >= 1.")
                    raise ValueError("number of prodesses must be >= 1.")

            # check that MLP layer sizes are >= 1
            if args.MLP_arch is not None:
                for layer_size in args.MLP_arch:
                    if layer_size < 1:
                        self.log.error("MLP architecture layer sizes must be >= 1.")
                        raise ValueError("MLP architecture layer sizes must be >= 1.") 

    # set parameters using arguments
    def _set_parms(self, args):

        # general arguments
        self.model_parms["num_dim"] = args.num_dim
        self.model_parms["algorithm"] = args.algorithm
        self.model_parms["pre_process"] = args.pre_process

        # PCA arguments (must be True or False)
        if args.whiten is None:
            self.model_parms["whiten"] = False
        else:
            self.model_parms["whiten"] = args.whiten

        # time alignment arguments
        self.model_parms["time_align_dim"] = args.time_align

        # auto-encoder arguments
        if args.model_type is None:
            self.model_parms["model_type"] = MODEL_TYPES[0]
        else:
            self.model_parms["model_type"] = args.model_type
        self.model_parms["epochs"] = args.epochs
        self.model_parms["batch_size"] = args.batch_size
        self.model_parms["num_processes"] = args.num_processes

    # add particular algorithm to end of model list
    def _init_algorithm(self, num_dim):

        # init pre-processing
        if self.model_parms["pre_process"] == "standard":
            self.pre_process.append(preprocessing.StandardScaler())

        elif self.model_parms["pre_process"] == "minmax":
            self.pre_process.append(preprocessing.MinMaxScaler())

        # init algorithm
        if self.model_parms["algorithm"] == "PCA":
            self.model.append(decomposition.PCA(n_components = num_dim, 
            whiten=self.model_parms["whiten"]))
        
        elif self.model_parms["algorithm"] == "incremental-PCA":
            self.model.append(decomposition.IncrementalPCA(n_components = num_dim,
                whiten=self.model_parms["whiten"]))

        elif self.model_parms["algorithm"] == "Isomap":
            self.model.append(manifold.Isomap(n_components = num_dim))
        
        elif self.model_parms["algorithm"] == "tSNE":
            self.model.append(manifold.TSNE(n_components = num_dim))
        
        elif self.model_parms["algorithm"] == "auto-encoder":
            self.model.append(AutoEncoder(self.log, num_dim = num_dim,
                                          model_type=self.model_parms["model_type"],
                                          epochs=self.model_parms["epochs"],
                                          batch_size = self.model_parms["batch_size"],
                                          num_processes=self.model_parms["num_processes"]))
                            
        elif self.model_parms["algorithm"] == "Umap":
            self.model.append(umap.UMAP(n_components = num_dim))

    # init model according to arguments
    def _init_model(self):

        # set time aligned rotation matrices to empty list by default
        self.align_rot_mats = []

        # for time aligned models we initialize the model when fit is called
        if self.model_parms["time_align_dim"] is None:
            
            # otherwise we add to the first item in model list
            self._init_algorithm(self.model_parms["num_dim"])

    # get any unrecognized arguments
    def unknown_args(self):
        return self.unknown_arglist

    # check if algorithm has partial fit
    def is_incremental(self):
        """
        Test if the user selected an incremental algorithm.

        Returns:
            is_incremental (bool): True if algorithm can be used in batch mode
        """

        # find algorithm in list of algorithms
        alg_ind = ALGORITHMS.index(self.model_parms["algorithm"])

        return INCREMENTAL[alg_ind]

    # check if algorithm can use multi-processing
    def has_inverse(self):
        """
        Check if user selected algorithm has an inverse.

        Returns:
            has_inverse (bool): True if algorithm has an inverse
        """

        # find algorithm in list of algorithms
        alg_ind = ALGORITHMS.index(self.model_parms["algorithm"])

        return INVERSE[alg_ind]

    # return goal number of dimensions
    def num_dim(self):
        """
        Get desired number of dimensions in reduction.

        Returns:
            num_dim (int): number of dimensions for desired reduction
        """

        return self.model_parms["num_dim"]

    # return number of dimensions to use for alignment
    def time_align_dim(self):
        """
        Get number of dimensions to use for time alignment.

        Returns:
            time_align (int): number of time alignment dimension
        """

        return self.model_parms["time_align_dim"]
    
    # perform dimension reduction 
    def fit(self, data, time_step=0):
        """
        Train dimension reduction model using samples.

        Args:
            data (array): data with points as rows
            time_step (int): train model at given time step 
        """
        
        # for time aligned models we initialize and fit model
        if self.model_parms["time_align_dim"] is not None:
            
            # check that time_step is in list or at end
            if time_step > len(self.model):
                self.log.error("Model does not exist and cannot be added.")
                raise IndexError("Model does not exist and cannot be added.")

            # add model to list
            if time_step == len(self.model):
                self._init_algorithm(self.model_parms["time_align_dim"])

        # most models in sklearn have a fit function 
        # but for tSNE we use fit_transform instead
        if self.model_parms["algorithm"] == "PCA" or \
            self.model_parms["algorithm"] == "incremental-PCA" or \
            self.model_parms["algorithm"] == "Isomap" or \
            self.model_parms["algorithm"] == "auto-encoder" or \
            self.model_parms["algorithm"] == "Umap":

            # check for pre-processing
            if self.model_parms["pre_process"] is not None:
                data = self.pre_process[time_step].fit_transform(data)

            self.model[time_step].fit(data)
    
    # perform incremental dimension reduction
    def partial_fit(self, data, time_step=0):
        """
        Train an incremental model using samples.

        Args:
            data (array): data with points as rows
            time_step (int): model time step
        """

        # for time aligne3d models we initialize and do partial fit
        if self.model_parms["time_align_dim"] is not None:

            # check that time_step is in list or at end
            if time_step > len(self.model):
                self.log.error("Model does not exist and cannot be added.")
                raise IndexError("Model does not exist and cannot be added.")

            # add model to list, if at new time step 
            # (otherwise should already be initialized)
            if time_step == len(self.model):
                self._init_algorithm(self.model_parms["time_align_dim"])

        # check for pre-processing
        if self.model_parms["pre_process"] is not None:
            self.pre_process[time_step].partial_fit(data)
            data = self.pre_process[time_step].transform(data)

        if self.model_parms["algorithm"] == "incremental-PCA":

            # make sure we have enough samples
            if data.shape[0] < self.model_parms["num_dim"]:
                self.log.warn("Need more samples to perform partial fit, ignoring batch.")
            else:
                self.model[time_step].partial_fit(data)
        
        if self.model_parms["algorithm"] == "auto-encoder":
            self.model[time_step].fit(data)

    # transform data using model
    def transform(self, data, time_step=0):
        """
        Transform data to lower dimensional representation.

        Args:
            data (numpy array): data with points as rows
            time_step (int): transform data at given time step

        Returns:
            data (array): reduced data
        """

        # check that data is 2D
        if len(data.shape) == 1:
            data = data.reshape(1, -1)

        # check for pre-processing
        if self.model_parms["pre_process"] is not None:
            data = self.pre_process[time_step].transform(data)

        # for tSNE do fit_transform
        if self.model_parms["algorithm"] == "tSNE":
            data = self.model[time_step].fit_transform(data)

        # other models have transform
        elif self.model is not None:
            data = self.model[time_step].transform(data)

        return data
    
    # inverse transform from data
    def inverse_transform(self, data, time_step=0):

        # inverse transform of latent data
        data = self.model[time_step].inverse_transform(data)

        # check for post-processing
        if self.model_parms["pre_process"] is not None:
            data = self.pre_process[time_step].inverse_transform(data)

        return data

    # time align reduced data using Kabsch algorithm
    def time_align(self, data, compute_rotations=True):
        """
        Time align reduced data using the Kabsch algorithm.  Expects the incoming
        dimension to be time_align_dim and truncates the dimension to num_dim
        after alignment.

        Args: data (list of array): list of data matrices of shape (sim, dim)
              compute_rotations (boolean): False to use existing rotation matrices

        Returns:
            aligned_data (list of array): list of data with shape (sim, reduced dim)
        """

        num_time = len(data)
        
        # align from last to first frame
        aligned_data = []
        align_rot_mats = []
        for i in reversed(range(num_time)):

            # rotate to previous coordinates
            if i == num_time-1:
                old_coords = data[i]
                align_rot_mats.insert(0, np.identity(data[i].shape[1]))

            else:

                # compute new rotations
                if compute_rotations:

                    # do Kabsch algorithm
                    A = data[i].transpose().dot(old_coords)
                    U, S, V = np.linalg.svd(A)
                    rot_mat = (V.transpose()).dot(U.transpose())

                # or use existing rotation matrices
                else:
                    rot_mat = self.align_rot_mats[i]

                # rotate to get new coordinates
                data[i] = data[i].dot(rot_mat.transpose())

                # update old coords
                old_coords = data[i]

                # save rotation matrices as part of model
                align_rot_mats.insert(0, rot_mat)

            # collect aligned coodinates
            aligned_data.insert(0, data[i][:,0:self.model_parms["num_dim"]])

        # save rotation matrices to model
        self.align_rot_mats = align_rot_mats

        return aligned_data
    
    # return percent variance captured per dimension per model
    # if a given algorithm doesn't have this information, return []
    def data_explained(self):
        """
        Returns the percent of information captured per dimension
        per model.  For PCA this would be the explained variance
        ratio.  If a model doesn't compute this information, an
        empty list is returned.

        Returns:
            data_explained (list): list of vectors of percent captured
        """

        variance_ratio_per_model = []
        if self.model_parms['algorithm'] == 'PCA' or \
            self.model_parms['algorithm'] == 'incremental-PCA':

            # collect results per model
            for i in range(len(self.model)):
                variance_ratio_per_model.append(np.cumsum(100 * 
                    self.model[i].explained_variance_ratio_))
        
        return variance_ratio_per_model

    # save dimension reduction model
    def save(self, file_out):
        """
        Saves a dimension reduction model to a .pkl file.

        Args:
            file_out (string): file name to save file
        """

        joblib.dump([self.model, self.pre_process, self.model_parms, self.align_rot_mats], 
            file_out, compress=True)

        self.log.info("Saved model file to %s." % file_out)
        
    # load dimension reduction model
    def load(self, file_in):
        """
        Loads a dimension reduction model from a .pkl file.

        Args:
            file_in (string): file name with saved model
        """

        # load model and parameters
        self.model, self.pre_process, self.model_parms, self.align_rot_mats = joblib.load(file_in)

        self.log.info("Loaded model file from %s." % file_in)


# if called from the command line display algorithm specific command line arguments
if __name__ == "__main__":

    # show options on command line if requested
    algorithms = DimensionReduction()
    algorithms.parser._parse_args()
    
    # anchor multi-processing to this module
    mp.set_start_method('spawn')