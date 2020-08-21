# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 08:37:40 2020

@author: 10624
"""

class internetwork(object):
    def __init__(self, inter_data, sc_networks, start_number):
        """ Set up the class of 2-partite interdependent networks: demand -> ...
        Input:
            inter_data: the data of interdependent networks
        """
        
        self.name = inter_data["name"]
        
        self.network1_num = inter_data["network1"]
        self.network2_num = inter_data["network2"]
        
        self.node1_num = inter_data["from"]
        self.node2_num = inter_data["to"]
        
        self.edge_prob = inter_data["edge_prob"]
        
        self.network1 = sc_networks[self.network1_num]
        self.network2 = sc_networks[self.network2_num]
        
        self.supplyseries = self.network1.type[self.node1_num[0]]
        self.demandseries = []
        for i in range(len(self.node2_num)):
            self.demandseries = np.concatenate((self.demandseries, self.network2.type[self.node2_num[i]]))
            
        self.start_number
    
    def adjmatrix(self):
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
                
                