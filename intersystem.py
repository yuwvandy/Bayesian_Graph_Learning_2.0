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
        import numpy as np
        
        self.nodenum = 0
        
        self.networks = networks
        self.internetworks = internetworks
        
        #base element setup
        temp = 0
        self.nodeseries = []
        for i in range(len(networks)):
            self.nodenum += networks[i].nodenum
            self.nodeseries.append(np.arange(temp, self.nodenum, 1))
            temp += networks[i].nodenum
    
    def adj_matrix(self):
        """ Set up the whole adjacent matrix of the interdependent systems
        """
        import copy
        import numpy as np
        
        self.adjmatrix = np.zeros((self.nodenum, self.nodenum), dtype = int)
        temp = 0
        for i in range(len(self.networks)):
            self.adjmatrix[self.nodeseries[i][0]:(self.nodeseries[i][-1] + 1), self.nodeseries[i][0]:(self.nodeseries[i][-1] + 1)] = copy.deepcopy(self.networks[i].adjmatrix)
            temp += np.sum(self.networks[i].adjmatrix)
            print(temp)
        for i in range(len(self.internetworks)):
            self.adjmatrix[(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[0]):(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[-1] + 1), :][:, (self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[0]):(self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[-1] + 1)] = copy.deepcopy(self.internetworks[i].adjmatrix)
            temp += np.sum(self.internetworks[i].adjmatrix)
            print(temp)