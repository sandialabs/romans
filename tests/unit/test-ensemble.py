# Copyright (c) 2021 National Technology and Engineering Solutions of Sandia, LLC.  
# Under the terms of Contract DE-NA0003525 with National Technology and Engineering 
# Solutions of Sandia, LLC, the U.S. Government retains certain rights in this software.

# This script does unit testing on ensemble.py.  To run the
# tests, use:
#
# $ python test-ensemble.py

# S. Martin
# 11/26/2020

# standard libraries
import unittest
import logging

# local libraries
import romans
import romans.ensemble

# test the ensemble.py module
class TestEnsemble(unittest.TestCase):

    # call ensemblse._d_parte_format
    def check_parse_d_format(self, log, test_data):

        # parse test string
        root, start, stop, step, ext = \
            romans.ensemble.parse_d_format(log, test_data["str"])

        # check that results are as expected
        assert root == test_data["root"], "root of " + \
            test_data["str"] + " is " + test_data["root"]
        assert start == test_data["start"], "start of " + \
            test_data["str"] + " is " + str(test_data["start"])
        assert stop == test_data["stop"], "stop of " + \
            test_data["str"] + " is " + str(test_data["stop"])
        assert step == test_data["step"], "step of " + \
            test_data["str"] + " is " + str(test_data["step"])
        assert ext == test_data["ext"], "ext of " + \
            test_data["str"] + " is " + test_data["ext"]

    # test parsing of %d[::] format
    def test_parse_d_format(self):

        # initialize logger
        romans.init_logger(log_level='debug')
        log = logging.getLogger("romans.ensemble_test")
        log.info("Started ensemble test log.")

        # check for correct input options

        # test no %d
        test_data = {"str": "example.txt",
                     "root": "example.txt",
                     "start": None,
                     "stop": None,
                     "step": None,
                     "ext": ""}
        self.check_parse_d_format(log, test_data)

        # test one %d
        test_data = {"str": "example.%d.txt",
                     "root": "example.",
                     "start": 0,
                     "stop": None,
                     "step": 1,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)    

        # test %d[::]
        test_data = {"str": "example.%d[::].txt",
                     "root": "example.",
                     "start": None,
                     "stop": None,
                     "step": 1,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)

        # test %d[10:]
        test_data = {"str": "example.%d[10:].txt",
                     "root": "example.",
                     "start": 10,
                     "stop": None,
                     "step": 1,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)

        # test %d[:10]      
        test_data = {"str": "example.%d[:10].txt",
                     "root": "example.",
                     "start": None,
                     "stop": 10,
                     "step": 1,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)

        # test %d[5:10]      
        test_data = {"str": "example.%d[5:10].txt",
                     "root": "example.",
                     "start": 5,
                     "stop": 10,
                     "step": 1,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)

        # test %d[::-1]      
        test_data = {"str": "example.%d[::-1].txt",
                     "root": "example.",
                     "start": None,
                     "stop": None,
                     "step": -1,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)

        # test %d[3:20:5]      
        test_data = {"str": "example.%d[3:20:5].txt",
                     "root": "example.",
                     "start": 3,
                     "stop": 20,
                     "step": 5,
                     "ext": ".txt"}
        self.check_parse_d_format(log, test_data)

        # check for incorrect input options
        
        # check for failure because of two %d
        test_data = {"str": "example.%d%d.txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of two [
        test_data = {"str": "example.%d[[.txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of %dxx[]
        test_data = {"str": "example.%dxx[].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of missing ]
        test_data = {"str": "example.%d[:3.txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of two ]]
        test_data = {"str": "example.%d[:3]].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of missing non digits between []
        test_data = {"str": "example.%d[aa].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of not having two numbers []
        test_data = {"str": "example.%d[30].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of not having two numbers []
        test_data = {"str": "example.%d[3:3:33:3].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because of dash between []
        test_data = {"str": "example.%d[-:-:-].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because negative start between []
        test_data = {"str": "example.%d[-1:7].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

        # check for failure because negative stop between []
        test_data = {"str": "example.%d[1:-7].txt"}
        try:
            self.check_parse_d_format(log, test_data)
        except romans.ensemble.EnsembleSpecifierError:
            pass

if __name__ == "__main__":
    unittest.main()