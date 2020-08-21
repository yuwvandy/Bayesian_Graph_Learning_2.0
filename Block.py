# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 15:19:44 2020

@author: 10624
"""

class block(object):
    def __init__(self, network_data):
        """
        Define the class of individual block within SBM
        Input:
            network_data

        Output: 
            the class of the block
        """
        
        import numpy as np

        self.name = network_data["name"]
        self.color = network_data["color"]
        self.fail_prop_matrix = network_data["fail_prop_matrix"]
        
        self.supplynum, self.trannum, self.demandnum = network_data["supplynum"], network_data["trannum"], network_data["demandnum"]
        self.supplyname, self.tranname, self.demandname = network_data["supplyname"], network_data["tranname"], network_data["demandname"]
        self.nodenum = self.supplynum + self.trannum + self.demandnum

        self.supplyseries = np.arange(0, self.supplynum, 1)
        self.transeries = np.arange(self.supplynum, self.supplynum + self.trannum, 1)
        self.demandseries = np.arange(self.supplynum + self.trannum, self.nodenum, 1)
        self.nodeseries = np.arange(0, self.nodenum, 1)
        
        self.type = [self.supplyseries, self.transeries, self.demandseries]

    def adj_matrix(self):
        """
        The topology of the block is specific to infrastructure network
        supply - transmission
        transmission - demand
        supply - demand
        transmission - transmission
        demand - demand
        
        The general idea to initialize the SBM is to:
        1) perform Bernoulli experiment to assign edges between:
            supply -> tran, supply -> demand, tran -> tran, tran -> demand, demand -> demand
        2) add extra edges to guarantee the following three properties:
            1. every supply node has at least one demand node for extracting resources, either directly or via a transmission node
            2. every demand node has at least one supply node for providing resources, either directly or via a transmission node
            3. every transmission node has at least one demand and at least one supply node for gaining and providing resources
        """
        import numpy as np
        import random as rd

        self.adjmatrix = np.zeros((self.nodenum, self.nodenum), dtype = int)

        #Performing Bernoulli experiment to assign edges
        for i in range(len(self.type)):
            for j in range(i, len(self.type)):
                if(i != j):
                    for m in self.type[i]:
                        for n in self.type[j]:
                            if(np.random.rand() < self.fail_prop_matrix [i, j]):
                                self.adjmatrix[m, n] = 1
                if(i == j):
                    for m in range(len(self.type[i])):
                        for n in range(m + 1, len(self.type[j])):
                            if(np.random.rand() < self.fail_prop_matrix[i, j]):
                                self.adjmatrix[self.type[i][m], self.type[j][n]] = 1
        
        #Add extra edges to ensure the above three properties
        ###Property 1: every supply node must have a demand for obtaining resources
        for i in self.supplyseries:
            if(self.DFS_supply2demand(i) == 0):
                self.adjmatrix[i, rd.sample(list(self.demandseries), 1)] = 1

        ###Property 2: every demand node must have a supply node for providing resources
        for i in self.demandseries:
            if(self.DFS_demand2supply(i) == 0):
                self.adjmatrix[rd.sample(list(self.supplyseries), 1), i] = 1
        
        ###Property 3: every transmission node must have a supply node and a demand node for providing and receiving resources
        for i in self.transeries:
            if(self.DFS_tran2supply(i) == 0):
                self.adjmatrix[rd.sample(list(self.supplyseries), 1), i] = 1
            if(self.DFS_tran2demand(i) == 0):
                self.adjmatrix[i, rd.sample(list(self.demandseries), 1)] = 1


    def DFS_supply2demand(self, i):
        """Perform DFS on self.adjmatrix starting from supply node finding demand node
        Input:
        i - the supply node i

        Output:
        0 - there are no demand nodes reachable from supply node i
        1 - there is at least one node reachable from supply node i
        """
        if(i in self.supplyseries):
            for j in self.transeries:
                if(self.adjmatrix[i, j] == 1):
                    flag, node = self.DFS_supply2demand(j)
                    if(flag == 1):
                        return 1, node
            
            for j in self.demandseries:
                if(self.adjmatrix[i, j] == 1):
                    return 1, j
            return 0, None

        if(i in self.transeries):
            for j in self.demandseries:
                if(self.adjmatrix[i, j] == 1):
                    flag, node = self.DFS_supply2demand(j)
                    if(flag == 1):
                        return 1, node
            return 0, None
        
        if(i in self.demandseries):
            return 1, i

        return 0, None

    def DFS_demand2supply(self, i):
        """Perform DFS on self.adjmatrix starting from demand node finding supply node
        Input:
        i - the demand node i

        Output:
        0 - there are no supply nodes reachable from demand node i
        1 - there is at least one node reachable from demand node i
        """
        if(i in self.demandseries):
            for j in self.transeries:
                if(self.adjmatrix[j, i] == 1):
                    flag = self.DFS_demand2supply(j)
                    if(flag == 1):
                        return 1
            
            for j in self.supplyseries:
                if(self.adjmatrix[j, i] == 1):
                    return 1
                
            return 0

        if(i in self.transeries):
            for j in self.supplyseries:
                if(self.adjmatrix[j, i] == 1):
                    flag = self.DFS_demand2supply(j)
                    if(flag == 1):
                        return 1
            return 0
        
        if(i in self.supplyseries):
            return 1

        return 0
    
    def DFS_tran2supply(self, i):
        """Perform DFS on self.adjmatrix starting from transmission node finding supply node
        Input:
        i - the transmission node i

        Output:
        0 - there are no supply nodes reachable from transmission node i
        1 - there is at least one node reachable from transmission node i
        """
        for j in self.supplyseries:
            if(self.adjmatrix[j, i] == 1):
                return 1
            return 0
        
    def DFS_tran2demand(self, i):
        """Perform DFS on self.adjmatrix starting from transmission node finding demand node
        Input:
        i - the transmission node i

        Output:
        0 - there are no demand nodes reachable from transmission node i
        1 - there is at least one demand node reachable from transmission node i
        """
        for j in self.demandseries:
            if(self.adjmatrix[i, j] == 1):
                return 1
            return 0
