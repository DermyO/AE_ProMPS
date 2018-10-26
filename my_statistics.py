# -*- coding: utf-8 -*-

import pdb
import yarp
import numpy as np
from klepto.archives import file_archive
from os import listdir
from os.path import isfile, join
import scipy.io as sio
import os
from Data_driver import Data_driver

class My_statistics():
    """Classe s'occupant des stats"""
    def __init__(
        self,
        data=[],
        data_inf=[],
        data_reconstr=[],
        data_shape=[10,70,70,69], #percentObservation, nbTest, nbFrame, nbMembers
        scaling_factors = []
        
    ):
        self.nbFrame = data_shape[2]
        self.nbTest = data_shape[1]
        self.nbPercentTest = data_shape[0]
        self.nbInput = data_shape[3]

        self.scaling_factors = scaling_factors
        
        if(scaling_factors == []):
                self.data = data
                self.data_inf = data_inf
                self.data_reconstr = data_reconstr
        else:
                data_driver = Data_driver()
                data_driver.scaling_factors = scaling_factors   
                self.data = data_driver.undo_unit_bounds_rescaling(data)
                self.data_inf = data_driver.undo_unit_bounds_rescaling(data_inf)
                self.data_reconstr = data_driver.undo_unit_bounds_rescaling(data_reconstr)
        self.data_shape = data_shape

        self.dist_real_reconstr = np.zeros([self.nbTest,self.nbFrame, self.nbInput], np.float32) # 70 70 69
        self.dist_real_inf = np.zeros(data_shape, np.float32) # 10 70 70 69
        self.dist_reconstr_inf = np.zeros(data_shape, np.float32) # 10 70 70 69

        #mean input dist for all trials for all timestep
        self.dist_real_reconstr_1D = np.zeros([self.nbTest,self.nbFrame], np.float32) #70 70
        self.dist_real_inf_1D = np.zeros([self.nbPercentTest,self.nbTest,self.nbFrame], np.float32) # 10 70 70        
        self.dist_reconstr_inf_1D = np.zeros([self.nbPercentTest,self.nbTest,self.nbFrame], np.float32) #10 70 70

        #mean timestep for all trials
        self.mean_dist_real_reconstr = np.zeros([self.nbTest], np.float32) # 70
        self.mean_dist_real_inf = np.zeros([self.nbPercentTest, self.nbTest], np.float32) # 10 70
        self.mean_dist_reconstr_inf = np.zeros([self.nbPercentTest, self.nbTest], np.float32) # 10 70

        self.var_dist_real_reconstr  = np.zeros([self.nbTest], np.float32)
        self.var_dist_real_inf = np.zeros([self.nbPercentTest, self.nbTest], np.float32)
        self.var_dist_reconstr_inf = np.zeros([self.nbPercentTest, self.nbTest], np.float32)
        
        self.varGlobalErr_real_reconstr = 0
        self.meanGlobalErr_real_reconstr  =0
        

    def def_scaling_factors(self, scaling_factors):
        

        if(self.scaling_factors == []):
                        self.scaling_factors = scaling_factors
                        data_driver = Data_driver()
                        data_driver.scaling_factors = scaling_factors   
                        self.data = data_driver.undo_unit_bounds_rescaling(self.data)
                        self.data_inf = data_driver.undo_unit_bounds_rescaling(self.data_inf)
                        self.data_reconstr = data_driver.undo_unit_bounds_rescaling(self.data_reconstr)                        
                        self.get_distance()
        else:
                print('error, the scaling factors has been already taken into account')
                        

    def get_distance(self):

        #real_reconstr
        self.dist_real_reconstr = np.abs(self.data - self.data_reconstr) #test Frame input

        self.dist_real_reconstr_1D = np.mean(self.dist_real_reconstr, axis=2) #mean inputs
        self.mean_dist_real_reconstr = np.mean(self.dist_real_reconstr_1D, axis=1) #mean frame and inputs per test
        self.var_dist_real_reconstr = np.var(self.dist_real_reconstr_1D, axis=1)
        self.meanGlobalErr_real_reconstr = np.mean(self.mean_dist_real_reconstr)
        self.varGlobalErr_real_reconstr = np.var(self.mean_dist_real_reconstr)
        if(self.data_inf!=[]):
                for i in range(self.data_shape[0]):
                    #real_inf
                    self.dist_real_inf[i] = np.abs(self.data - self.data_inf[i]) # test frame input
                    self.dist_real_inf_1D[i] = np.mean(self.dist_real_inf[i], axis=2) #mean inputs
                    self.mean_dist_real_inf[i] = np.mean(self.dist_real_inf_1D[i], axis=1) #mean frame and input per test
                    self.var_dist_real_inf[i] = np.var(self.dist_real_inf_1D[i], axis=1)            
                    #reconstr_inf
                    self.dist_reconstr_inf[i] = np.abs(self.data_reconstr - self.data_inf[i])
                    self.dist_reconstr_inf_1D[i] = np.mean(self.dist_reconstr_inf[i], axis=2)
                    self.mean_dist_reconstr_inf[i] = np.mean(self.dist_reconstr_inf_1D[i], axis=1)
                    self.var_dist_reconstr_inf[i] = np.var(self.dist_reconstr_inf_1D[i], axis=1)        
        
        
        
