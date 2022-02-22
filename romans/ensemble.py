# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This module contains utility funtions for use in processing 
# ensemble data.

# S. Martin
# 11/25/2020

# standard library
import re
import os

# 3rd party libraries
import numpy as np
import pandas as pd
from ipyparallel import Client

# class for keeping track of ensemble files
class Table:
    """
    Provides storage and methods for keeping track of simulation data and
    files in an ensemble.  The central assumption is that the esnemble is
    stored in a directory structure of the form (names are arbitrary):

    ::

        ensemble
        |-- ensemble.info
        |-- simulation.1
            |-- simulation.info
            |-- time.step.1
            |-- time.step.2
            |-- ...
        |-- simulation.2
        |-- simulation.3
        ...

    where ensemble is the central directory, containing multiple simulations,
    each having it's own directory simulation.1, simulation.2, and so on.  The
    simulation directories then contain files containg time step data time.step.1,
    time.step.2, etc.  The directory names can be somewhat arbitrary and can 
    have additional subdirectories and files, but the convention for using
    these utitilities is that the simulation directories can be specified using
    a Python like ``%d[::]`` specifier.  For example, for the above ensemble,
    we would use ``ensemble/simulation.%d[1:]`` to specify the file name 
    format for each of the simulation folders, then we would use
    ``time.step.%d[1:]`` to specify the time step files within each simulation
    folder.

    The ``%d[::]`` notation specifies the order and numbers in the directory/file
    names and using the Python slicing conventions.  For example ``%d`` specifies
    all names with an integer in the given location, ``%d[5:10]`` specifies
    all names with an integer starting at 5 and ending at 9, and ``%d[100:2:-2]``
    specifies all names with an integer starting at 100 and descending by 2 to 2.

    The numbers in the directory/file names are assumed to be >= 0 but are otherwise
    unrestriced.

    To instantiate an ensemble.Table object, use either a .csv file to
    create a full table, or an ensemble_spec, file_spec, and table header to
    create a single column table.

    Args:
        csv_file=None (string): file name of .csv file
        ensemble_spec=None (string): string with %d[::] giving 
            simulation directories
        file_spec=None (string): string with %d[::] giving time step file names
        header=None (string): name for column header
    
    Note: If csv_file is provided then the other inputs are ignored.  If no
    inputs are provided then an empty table is created.
    """

    # the ensemble information is stored internally using all available
    # information, for example full file names.
    def __init__(self, log, data_frame=None, csv_file=None, ensemble_spec=None, 
                 file_spec=None, header=None):

        # logger to use
        self.log = log

        # initialize dataframe to empty
        self.table = pd.DataFrame()

        # data_frame over-rides all other options
        if data_frame is not None:

            self.table = data_frame

        # csv file over-rides ensemble/file/header spec
        elif csv_file is not None:

            # load csv file into data frame
            self.table = pd.read_csv(csv_file, index_col=0)

        # otherwise use ensemble specifier
        elif ensemble_spec is not None:

            # get ensemble directories (in order)
            ensemble_dirs = self.directories(ensemble_spec)

            # combine with ensemble directories to create table
            table_col = []
            for ensemble_dir in ensemble_dirs:
                if file_spec is not None:
                    table_col.append(os.path.join(ensemble_dir, file_spec))
                else:
                    table_col.append(ensemble_dir)
            
            if header is not None:
                self.table = pd.DataFrame(data={header: table_col}, index=ensemble_dirs)
            else:
                self.table = pd.DataFrame(data={'': table_col}, index=ensemble_dirs)

        log.debug('Created ensemble table.')

    # parses a file or directory name according to %d[::] format
    # and returns the associated integer, if present, otherwise returns None
    def _parse_d_name(self, file_or_dir, root, ext):

        # skip non-root matches
        if file_or_dir.find(root) != 0:
            return None

        # skip non-extension matches
        if not file_or_dir.endswith(ext):
            return None

        # make sure dir_name can contain both root and extension
        if len(file_or_dir) < len(root) + len(ext):
            return None
        
        # strip off root and extension
        file_or_dir_num = file_or_dir[len(root) : len(file_or_dir) - len(ext)]

        # check that dir_num is an integer
        try:
            file_or_dir_num = int(file_or_dir_num)
        except ValueError:
            return None
        
        return file_or_dir_num

    # finds and orders all files or directories in a path matching specifier
    # use is_dir == True to list directories, False to list files
    # root, start, stop, step, and ext specify the %d[::] format,
    # and path_ext supplies any trailing sub-directories (in case is_dir==True)
    def _catalog_path_contents(self, path, is_dir, root, 
        start, stop, step, ext, path_ext=""):

        # get contents of directory
        path_contents = os.listdir(path)
        
        # catalog directories matching %d[::] format
        catalog_name = []
        catalog_num = []
        for file_or_dir in path_contents:

            # skip files
            if is_dir:
                if not os.path.isdir(os.path.join(path, file_or_dir)):
                    continue

            # skip directories
            else:
                if not os.path.isfile(os.path.join(path, file_or_dir)):
                    continue
                    
            # get associated number, if any
            file_or_dir_num = self._parse_d_name(file_or_dir, root, ext)

            # if successful match, return name and number
            if file_or_dir_num is not None:
                catalog_name.append(file_or_dir)
                catalog_num.append(file_or_dir_num)

        # check that files were found
        if len(catalog_name) == 0:
            return []

        # if start is not given use min or max
        if start is None:

            # if step is positive, start at min
            if step > 0:
                start = min(catalog_num)

            # if step is negative, start at max
            if step < 0:
                start = max(catalog_num)

        # if stop is not given use min or max
        if stop is None:

            # if step is positive, stop at max (including last file)
            if step > 0:
                stop = max(catalog_num) + 1

            # if step is negative, stop at min (including last file)
            if step < 0:
                stop = min(catalog_num) - 1

        # precompute range of acceptable numbers
        range_nums = range(start, stop, step)

        # filter according to range
        for i in reversed(range(len(catalog_num))):

            # check if number is in range
            if catalog_num[i] not in range_nums:
                del catalog_name[i]
                del catalog_num[i]

        # sort ascending or descending depending on step sign
        sort_inds = np.argsort(catalog_num)
        if step < 0:
            catalog_name = list(np.array(catalog_name)[sort_inds[::-1]])
        else:
            catalog_name = list(np.array(catalog_name)[sort_inds])
        
        # add directory information to names, and double check
        # that files/directories exist
        sorted_catalog_names = []
        for file_or_dir in catalog_name:

            # add directories
            if is_dir:
                if os.path.exists(os.path.join(path, file_or_dir, path_ext)):
                    sorted_catalog_names.append(os.path.join(path, file_or_dir, path_ext))

            # add files (including path)
            else:
                if os.path.exists(os.path.join(path, file_or_dir)):
                    sorted_catalog_names.append(os.path.join(path, file_or_dir))

        return sorted_catalog_names

    # returns a list of directories matching %d[::] specifier
    def directories(self, directory_spec):
        """
        Return a list of directories matching ``%d[::]`` specifier.  The specifier
        is expanded and existing directories are identified and returned.

        Args:
            directory_spec (string): directory name with ``%d[::]`` specifier

        Returns:
            directory_list (list): list of directories matching specifier
        """

        # parse directory spec, note that if %d is in middle of multiple 
        # directories then dir_ext could include other directories
        dir_root, start, stop, step, dir_ext = \
            parse_d_format(self.log, directory_spec)

        # in case %d was not given, check that directory exists
        if step is None:
            if os.path.isdir(directory_spec):
                return [directory_spec]
            else:
                return []

        # parse directory specifier into parts path/root%d[::]ext/path_ext

        # get path of directories and root of spec
        path, root = os.path.split(dir_root)

        # get directory spec and path extensions
        path_ext, ext = os.path.split(dir_ext[::-1])
        ext = ext[::-1]
        path_ext = path_ext[::-1]

        return self._catalog_path_contents(path, True, 
            root, start, stop, step, ext, path_ext=path_ext)

    # creates output directories, return false if already exist
    def mirror_directories (self, output_dir, ensemble_dirs, over_write):
        """
        Creates a set of directories in the output directory which mirror
        the ensemble directory structure, unless output directory already
        exists.

        Args:
            output_dir (string): name of output directory to create
            ensemble_dirs (list): list of ensemble directories to mirror
            over_write (boolean): true to over-write existing directories

        Returns:
            mirror_dirs (list): list of mirror directories (including output_dir),
                None if directories were not created
        """

        # get common path
        if len(ensemble_dirs) > 1:
            common_path = os.path.commonpath(ensemble_dirs)

        # if only one simulation, commom path is everything up to file specifier
        else:
            common_path = os.path.dirname(ensemble_dirs[0])

        # construct mirror directories
        mirror_dirs = []
        for i in range(len(ensemble_dirs)):

            # strip off file
            ensemble_path = os.path.dirname(ensemble_dirs[i])

            # remove common path
            mirror_path = ensemble_path.split(common_path)[1]
            mirror_path = os.path.basename(mirror_path)

            # strip off base dir
            mirror_dirs.append(os.path.join(output_dir, mirror_path))

        # check if output directory exists
        if os.path.exists(output_dir):
            if not over_write:
                return None
        
        # create mirror directories
        for mirror_dir in mirror_dirs:
            try:
                os.makedirs(mirror_dir)
            except FileExistsError:

                # directory already exists, but that's OK because
                # user has allowed over-writing
                pass

        return mirror_dirs
    
    # locate and order existing files in ensemble directory matching %d[::] specifier
    def files (self, file_spec):
        """
        Return a list of files matching ``%d[::]`` specifier.  The specifier
        is expanded and existing files are identified and returned.

        Args:
            file_spec (string): file path with ``%d[::]`` specifier

        Returns:
            file_list (list): list of files matching specifier
        """

        # split into path and specifier
        path, files = os.path.split(file_spec)

        # parse file spec
        root, start, stop, step, ext = parse_d_format(self.log, files)

        # in case %d was not given, check that file exists
        if step is None:
            if os.path.isfile(file_spec):
                return [file_spec]
            else:
                return []

        return self._catalog_path_contents(path, False, 
            root, start, stop, step, ext)
    
    # get all files in an ensemble
    def ensemble_files (self, ensemble_dirs, parallel=False):
        """
        Returns a list of lists of files matching ``%d[::]`` specifier in
        the ensemble_dirs.  The specifier is expanded and existing files are
        identified and returned.  The directories are expected to exist.

        Args:
            ensemble_dirs (list): list of ensemble directories to expand (with specifier)
            parallel (boolean): run in parallel with ipyparallel (default False)

        Returns:
            sim_files (list): list of list of files matching specifier
        """

        # if not parallel, just get all sim files normally
        if not parallel:
            return self._get_sim_files(ensemble_dirs)

        # otherwise start parallel version

        # parallel operation, using direct view
        rc = Client()
        view = rc.load_balanced_view()

        # get number of available engines
        num_engines = len(rc)

        # break files into blocks so that we use all available engine
        num_dirs = len(ensemble_dirs)
        block_size = int(np.ceil(num_dirs/num_engines))

        # push out parallel jobs finding ensemble files
        async_results = []
        for i in range(num_engines):

            # get block of ensemble directories to expand
            block_dirs = [ensemble_dirs[j] 
                for j in range(i*block_size, i*block_size + block_size) if j < num_dirs]

            # expand ensemble directories as nodes are available
            async_result = rc[i].apply_async(self._get_sim_files, block_dirs)
            async_results.append(async_result)
        
        # wait for calculation to complete
        view.wait(async_results)

        # collect results
        sim_files = []
        for i in range(len(async_results)):

            # get items in order they were put into queue
            async_result = async_results.pop(0)

            # retrieve data from result
            sim_files += async_result.get()

        # clean up ipyparallel
        rc.purge_everything()

        # close ZMQ sockets
        rc.close()

        return sim_files

    # get simulation files for a block of ensemble directories
    def _get_sim_files (self, ensemble_dirs):

        sim_files = []
        for i in range(len(ensemble_dirs)):

            # find files in directory
            files_to_convert = self.files(ensemble_dirs[i])

            # add files to list
            sim_files.append(files_to_convert)

        return sim_files

    # convert an input file specifier to an output file specifier
    def convert_specifier (self, file_spec, output_dir, output_type):
        """
        Returns an output file specifier matching the provided input specifier.

        Args:
            file_spec (string): file path with ``%d[::]`` specifier
            output_dir (string): output directory for file spec files
            output_type (string): file extension of output files

        Returns:
            out_file_spec (string): output file path with ``%d[::]`` specifier
        """

        # create output file specifier
        file_name = os.path.basename(file_spec)
        file_root, file_ext = os.path.splitext(file_name)
        file_spec_out = os.path.join(output_dir, file_root + "." + output_type)

        return file_spec_out

    # get a column from the table
    def get_col (self, col):
        """
        Returns a column from the ensemble table.

        Args:
            col (string): name of column to return
        
        Returns:
            col_list (list): list of contents in column        
        """

        return self.table[col].tolist()

    # add a column to the table
    def add_col (self, col, header):
        """
        Adds a column to the ensemble table.  This column can be a list or a dictionary.
        If it is a list, the column is added in the list order.  If it is a dictionary,
        the column is added in the order of the dictionary keys by matching with an
        existing column.  The column is added at the end of the table.

        Args:
            col (list or dict): column data to add
            header (string): name of new column
        """

        self.table[header] = col

    # convert file pointers in specified columns
    def convert_cols (self, cols, uri_root):
        """
        Converts the specified columns in the table to have the given URI root.

        Args:
            cols (list): names of columns to convert
            uri_root (string): URI root to use for conversion
        """

        # check that columns exist in table
        self._check_cols(cols)

        # go through each column and convert file paths
        for col in cols:
            
            # get column
            old_col = self.table[col].to_list()

            # convert column
            new_col = self._convert_col_uri(old_col, uri_root)

            # replace column in dataframe
            self.table[col] = new_col

    # converts a list of paths to a URI based path
    def _convert_col_uri (self, paths, uri_root):

        # get common path
        common_path = os.path.commonpath(paths)

        # go through list
        new_paths = []
        for path in paths:

            # remove common path
            new_path = path.split(common_path)[1]

            # reverse path to get rid of any leading "/" which
            # might be interpreted as a root directory
            dir_path = os.path.basename(new_path[::-1])[::-1]
            base_path = os.path.dirname(new_path[::-1])[::-1]
            new_path = os.path.join(dir_path, base_path)

            # add URI root
            uri_path = os.path.join(uri_root, new_path)

            # save result in list of new paths
            new_paths.append(uri_path)

        return new_paths

    # check that provided columns exist in table
    # this prevents an ugly and uninterpretable pandas error
    def _check_cols (self, cols):

        # make sure columns are specified
        if cols is None:
            return

        # check that columns exist in table
        for col in cols:            
            if col not in self.table.columns:
                raise ValueError('Could not find column "%s" in table.' % col)

    # write out .csv file
    def to_csv (self, file_out, output_dir='', cols=None, index=True, index_label=None):
        """
        Writes out the table to a .csv file.

        Args:
            file_out (string): name of .csv file
            output_dir (string): output directory to use for .csv file
            cols (list): list of column headers to output
        """

        # check that columns exist in table
        self._check_cols(cols)

        # put output file into output_dir
        csv_out_file = os.path.join(output_dir, file_out)

        self.table.to_csv(path_or_buf=csv_out_file, columns=cols, 
            index=index, index_label=index_label)
        self.log.info('File written: %s.' % csv_out_file)


# helper function for parsing %d[::] format string
def parse_d_format(log, d_str):
    """
    Parse %d[::] format into root, start, stop, step, and extension,

    where root is the string before the %d[], start, stop, step are
    integers to put into range(start, stop, step), and extension is 
    the string after the %d[::].

    if %d is not given, start, stop, step are None, and ext is ''.
    if %d is given, but [] is empty, start=None, stop=None, step=1.
    if %d is given as [::], start=None, stop=None, step=1.
    if %d is given as [::-1], start=None, stop=None, step=-1.
    if %d is given as [:stop] then start=0, stop=None, step=1.

    In other words, step==None only if %d is not given, in all other
    cases, step is non-zero, but start, stop might have to be inferred.

    Args:
        (object) log: logger for messages
        (string) d_str: string to parse

    Returns:
        root (string): string before %d
        start (int): start index
        stop (int): stop index
        step (int): step between start and stop
        ext (string): extension after %d[::]
    """

    # get root
    d_str_parts = d_str.split("%d")

    # if there is no %d return original string
    if len(d_str_parts) == 1:
        return d_str, None, None, None, ''

    # if there is more than one %d raise exception
    if len(d_str_parts) > 2:
        log.error('Invalid %d[::] format, ' +
            'can not have more than one "%%d": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'can not have more than one "%d"')

    # seperate root, possibly extension
    root = d_str_parts[0]
    ext = d_str_parts[1]

    # split on [ after %d
    left_brackets = ext.split("[")

    # if no brackets, return empty range list
    if len(left_brackets) == 1:
        return root, 0, None, 1, ext

    # if more than one left bracket then error
    if len(left_brackets) > 2:
        log.error('Invalid %d[::] format, ' +
            'can not have more that one "[": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'can not have more than one "["')

    # check that left bracket is after %d
    if left_brackets[0] != '':
        log.error("Invalid %%d[::] format: %s." % d_str)
        raise EnsembleSpecifierError(d_str, "invalid %d[::] format")

    # split on ] after [
    right_brackets = left_brackets[1].split("]")

    # error if no closing bracket
    if len(right_brackets) == 1:
        log.error('Invalid %%d[::] format, has no closing "]": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'has no closing "]"')

    # error if more than one right bracket
    if len(right_brackets) > 2:
        log.error('Invalid %d[::] format, ' +
            'can not have more than one "]": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'can not have more than one "]"')

    # save extension, what's between the brackets
    ext = right_brackets[1]
    in_brackets = right_brackets[0]

    # check that between brackets has only : or digits
    if len(re.findall('[^:\-\d]', in_brackets)) > 0:
        log.error('Invalid %d[::] format, ' +
            'can only have integers between "[]": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'can only have integers between "[]"')

    # get numbers between brackets
    nums_in_brackets = in_brackets.split(":")

    # check that there are at least two numbers
    if len(nums_in_brackets) < 2 or len(nums_in_brackets) > 3:
        log.error('Invalid %d[::] format, ' +
            'must have at two or three numbers between "[]": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'must have two or three numbers between "[]"')

    # convert numbers to start, stop, step
    # unspecified numbers are returned as None
    start = None
    stop = None
    step = 1
    try:

        # convert start
        if nums_in_brackets[0] != '':
            start = int(nums_in_brackets[0])

        # convert stop
        if nums_in_brackets[1] != '':
            stop = int(nums_in_brackets[1])

        # convert step
        if len(nums_in_brackets) == 3:
            if nums_in_brackets[2] != '':
                step = int(nums_in_brackets[2])
    
    except ValueError:
        log.error('Invalid %d[::] format, ' +
            'must have integer values between "[]": %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'must have integer values between "[]"')

    # check value for start is non-negative
    if start is not None:
        if start < 0:
            log.error('Invalid %d[::] format, ' +
                'must have non-negative start between "[]": %s' % d_str)
            raise EnsembleSpecifierError(d_str, 'must have non-negative start between "[]"')

    # check value for stop is non-negative
    if stop is not None:
        if stop < 0:
            log.error('Invalid %d[::] format, ' +
                'must have non-negative stop between [""]: %s' % d_str)
            raise EnsembleSpecifierError(d_str, 'must have non-negative stop between "[]"')
    
    # check that step is not zero
    if step == 0:
        log.error('Invalid %d[::] format, ' +
            'must have non-zero step between [""]: %s' % d_str)
        raise EnsembleSpecifierError(d_str, 'must have non-zero step between "[]"')

    return root, start, stop, step, ext


# factory method to combine tables
def combine (log, tables, ignore_index=False):
    """
    Combines multiple tables to create a new table.  Tables are assumed
    to have identical index values and are joined as columns.

    Args:
        log (logger object): logger for writing output
        tables (list): list of ensemble tables to combine
        ignore_index (bool): ignore index values and join as rows

    Returns:
        combined_table (Table): ensemble table made up of input tables
    """
    
    # if no tables are given, return empty table
    if len(tables) == 0:
        return Table(log)
    
    # if one table is given, return table
    if len(tables) == 1:
        return tables[0]
    
    pd_tables = []
    for table in tables:

        # check that indices are identical
        if not ignore_index:
            if not np.array_equal(table.table.index.to_numpy(), 
                                  tables[0].table.index.to_numpy()):
                log.error("Indices do not match -- tables cannot be combined. " +
                                 "Use --ignore-index to combine anyway.")
                raise ValueError("Indices do not match -- tables cannot be combined. " +
                                 "Use --ignore-index to combine anyway.")

            # use original indices
            reset_table = table.table

        # otherwise reset indices
        else:
            reset_table = table.table.reset_index(drop=True)
    
        # remove Table information, leaving only pandas table
        pd_tables.append(reset_table)
        
    # concatenate tables
    return Table(log, data_frame=pd.concat(pd_tables, axis=1, join='inner', verify_integrity=True))


# factory method to explode a table using a column of lists
def explode(log, table, table_col, table_list):
    """
    Expand a table by replacing a given col with an exploded column,
    where other rows are duplicated.

    Args:
        log (logger object): logger for writing output
        table (Table object): ensemble table to explode
        table_col (string or integer): column in table to replace with exploded column
        table_list (list): column with list items to explode to replace table_col
    """

    # replace table_col with table_list
    new_col = pd.DataFrame({table_col: table_list},
        index=table.table.index)
    table.table[table_col] = new_col[table_col]
    
    # explode table using table_list column
    return Table(log, data_frame=table.table.explode(table_col))

# factory method to concatenate tables
def concat(log, tables):
    """
    Concatenates multiple tables to create a new table.  Tables are assumed
    to have identical header values and are joined vertically.

    Args:
        log (logger object): logger for writing output
        tables (list): list of ensemble tables to combine

    Returns:
        concat_table (Table): ensemble table made up of input tables
    """

    # if no tables are given, return empty table
    if len(tables) == 0:
        return Table(log)
    
    # if one table is given, return table
    if len(tables) == 1:
        return tables[0]
    
    # check that column headers are identical
    headers = list(tables[0].table)
    headers_identical = True
    for table in tables:
        if list(table.table) != headers:
            headers_identical = False

    # quit if headers are not identical
    if not headers_identical:
        log.error("Table headers are not identical, cannot concatenate tables.")
        raise ValueError ("Table headers are not identical, cannot concatenate tables.")

    # make list of tables
    pd_tables = []
    for table in tables:
        pd_tables.append(table.table)

    return Table(log, data_frame=pd.concat(pd_tables))


# error message for incorrect %d[::] specifier
class EnsembleSpecifierError(Exception):
    """
    Exception raised for errors in %d[::] format specifier.

    Args:
        specifier (string): input specifier which caused the error
        message (string): explanation of the error
    """

    def __init__(self, specifier, message="invalid %d[::] format"):

        # specifics for error
        self.specifier = specifier
        self.message = message

        # raise exception
        super().__init__(self.message)

    def __str__(self):
        return f'{self.message}: {self.specifier}'
