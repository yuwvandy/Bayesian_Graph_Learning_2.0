# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 13:06:54 2020

@author: 10624
"""

class interblock(object):
    def __init__(self, inter_block_data, networks):
        """ Set up the class of 2-partite interdependent networks: demand -> ...
        Input:
            inter_block_data: the data of interdependent blocks
            networks: the data of each individual block
        """
        import numpy as np
        
        self.name = inter_block_data["name"]
        
        self.network1_num = inter_block_data["network1"]
        self.network2_num = inter_block_data["network2"]
        
        self.node1_num = inter_block_data["from"]
        self.node2_num = inter_block_data["to"]
        
        self.edge_prob = inter_block_data["edge_prob"]
        
        self.network1 = networks[self.network1_num]
        self.network2 = networks[self.network2_num]
        
        self.supplyseries = self.network1.type[self.node1_num[0]]
        self.demandseries = []
        for i in range(len(self.node2_num)):
            self.demandseries = self.demandseries + list(self.network2.type[self.node2_num[i]])
        self.demandseries = np.array(self.demandseries, dtype = int)
        
        self.supplynum = len(self.supplyseries)
        self.demandnum = len(self.demandseries)
        
        self.supply_start_num = inter_block_data["supply_start_num"]
        self.demand_start_num = inter_block_data["demand_start_num"]
        
        self.fail_prop = inter_block_data["fail_prop"]
    
    def adj_matrix(self):
        """ Create the adjacent matrix of the internetworks
        Two properties:
            1. All supply nodes have at least one demand nodes for resources to go to
            2. All demand nodes have at least one supply nodes for resources obtaining
        """
        import numpy as np
        
        self.adjmatrix = np.zeros((len(self.supplyseries), len(self.demandseries)), dtype = int)
        
        for i in range(len(self.supplyseries)):
            for j in range(len(self.demandseries)):
                if(np.random.rand() <= self.edge_prob):
                    self.adjmatrix[i, j] = 1
        
        ##To ensure the property 1
        for i in range(len(self.supplyseries)):
            while(np.sum(self.adjmatrix[i, :]) == 0):
                for j in range(len(self.demandseries)):
                    if(np.random.rand() <= self.edge_prob):
                        self.adjmatrix[i, j] = 1
                        break
        
        ##To ensure the property 2
        for i in range(len(self.demandseries)):
            while(np.sum(self.adjmatrix[:, i]) == 0):
                for j in range(len(self.supplyseries)):
                    if(np.random.rand() <= self.edge_prob):
                        self.adjmatrix[j, i] = 1
                        break
    
    def failpropmatrix(self):
        """ Create the failure propagation matrix
        Output:
            self.fail_prop_matrix of N*N: fail_prop_matrix[i, j]: the failure propagation from i to j
        """
        import numpy as np
        
        self.fail_prop_matrix = np.zeros((self.supplynum, self.demandnum), dtype = float)
        
        for i in range(self.supplynum):
            for j in range(self.demandnum):
                self.fail_prop_matrix[i, j] = self.fail_prop
    
    def edgeprobmatrix(self):
        """Calculate the probability of edges
        Input:
            self.edge_prob: the probability of edges between certain types of nodes
        Output:
            self.edge_prob_matrix: the matrix of the edge probability
        """
        import numpy as np
        
        self.edge_prob_matrix = np.zeros((self.supplynum, self.demandnum), dtype = float)
        
        for i in range(self.supplynum):
            for j in range(self.demandnum):
                self.edge_prob_matrix[i, j] = self.edge_prob
                