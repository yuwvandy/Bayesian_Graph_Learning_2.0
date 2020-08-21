# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 13:58:07 2020

@author: 10624
"""

class sbm(object):
    def __init__(self, networks, internetworks):
        """Set up the class of stochastic block model with networks and internetworks as the subelement to comprise the whole system
        Input:
            networks: list of network objects
            internetworks: list of internetwork objects
        Output:
            class of the stochastic block model
        """
        self.nodenum = 0
        
        self.networks = networks
        self.internetworks = internetworks
        
        #base element setup
        temp = 0
        self.nodeseries = []
        for i in range(len(networks)):
            self.nodenum += networks[i].nodenum
            self.nodeseries.append(np.arange(temp, networks[i].nodenum, 1))
            temp += networks[i].nodenum
    
    def adj_matrix(self):
        """ Set up the whole adjacent matrix of the interdependent systems
        """
        import copy
        import numpy as np
        
        self.adjmatrix = np.zeros((self.nodenum, self.nodenum), dtype = int)
        
        for i in range(len(self.networks)):
            self.adjmatrix[self.nodeseries[i], self.nodeseries[i]] = copy.deepcopy(self.networks[i].adjmatrix)
        
        for i in range(len(self.internetworks)):
            self.adjmatrix[self.internetworks[i].start_number + self.internetworks[i].supplyseries, self.internetworks[i].start_number + self.internetworks[i].demandseries] = 1
            