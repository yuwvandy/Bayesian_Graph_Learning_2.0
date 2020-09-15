# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 13:45:33 2020

@author: 10624
"""

class block_intersystem(object):
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
        
        self.nodetypeseries = [self.networks[0].supplyseries, self.networks[0].transeries, self.networks[0].demandseries,\
                               self.networks[1].supplyseries + self.networks[0].nodenum, self.networks[1].transeries + self.networks[0].nodenum, self.networks[1].demandseries + self.networks[0].nodenum,\
                               self.networks[2].supplyseries + self.networks[0].nodenum + self.networks[1].nodenum, self.networks[2].transeries + self.networks[0].nodenum + self.networks[1].nodenum, self.networks[2].demandseries + self.networks[0].nodenum + self.networks[1].nodenum]
    
        self.nodesearchtable()
        
    def nodesearchtable(self):
        """Create a node search table where we can assess the class the node belongs to , the node subtype and the node while type
        0, 1, 2 columns correspond to network, subtype, nodenumber in the network
        """
        import numpy as np
        
        self.nodetable = np.zeros((self.nodenum, 3), dtype = int)
        
        for i in range(len(self.networks)):
            for j in range(len(self.networks[i].type)):
                for k in self.networks[i].type[j]:
                    self.nodetable[k + self.networks[i].start_num, 0] = i
                    self.nodetable[k + self.networks[i].start_num, 1] = j
                    self.nodetable[k + self.networks[i].start_num, 2] = k
                    

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

        for i in range(len(self.internetworks)):
            self.adjmatrix[(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[0]):(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[-1] + 1), :][:, (self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[0]):(self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[-1] + 1)] = copy.deepcopy(self.internetworks[i].adjmatrix)
            temp += np.sum(self.internetworks[i].adjmatrix)


            
    def failinitialize(self):
        """Initialize the failure scenario, specifically prepare all array for saving failure data
        """
        self.node_fail_prob = []
        #The difference between node_fail_sequence and node_fail_final is the former shows how node fails in time but the latter shows the final node failure scenario
        self.node_fail_sequence = []
        self.node_fail_final = []
        
    def edge_failure_matrix(self):
        """Calculate the probability of failure propagation within same type of nodes and different types of nodes
        Input:
            fail_prop: the failure probability matrix
        """
        import numpy as np
        import copy
        
        self.fail_prop_matrix = np.zeros((self.nodenum, self.nodenum), dtype = float)
        
        for i in range(len(self.networks)):
            self.fail_prop_matrix[self.nodeseries[i][0]:(self.nodeseries[i][-1] + 1), self.nodeseries[i][0]:(self.nodeseries[i][-1] + 1)] = copy.deepcopy(self.networks[i].fail_prop_matrix)
        
        for i in range(len(self.internetworks)):
            self.fail_prop_matrix[(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[0]):(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[-1] + 1), :][:, (self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[0]):(self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[-1] + 1)] = copy.deepcopy(self.internetworks[i].fail_prop_matrix)
            
    def edgeprobmatrix(self):
        """Calculate the probability matrix of edges
        """
        import numpy as np
        import copy
        
        self.edge_prob_matrix = np.zeros((self.nodenum, self.nodenum), dtype = float)
        temp = 0
        for i in range(len(self.networks)):
            self.edge_prob_matrix[self.nodeseries[i][0]:(self.nodeseries[i][-1] + 1), self.nodeseries[i][0]:(self.nodeseries[i][-1] + 1)] = copy.deepcopy(self.networks[i].edge_prob_matrix)

        for i in range(len(self.internetworks)):
            self.edge_prob_matrix[(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[0]):(self.internetworks[i].supply_start_num + self.internetworks[i].supplyseries[-1] + 1), :][:, (self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[0]):(self.internetworks[i].demand_start_num + self.internetworks[i].demandseries[-1] + 1)] = copy.deepcopy(self.internetworks[i].edge_prob_matrix)

        
    def failure_probability(self):
        """Calculate the node failure probability based on failure_matrix
        Input:
            failure probability of edge edge (self.fail_prop_matrix)
            failure_sequence: node failure sequence: 1 - failed, 2 - unfailed
        Output:
            failure_prob: the failure probability of each node
        """
        import numpy as np
        
        node_fail_prob = np.zeros(self.nodenum, dtype = float)
        node_fail_sequence = self.node_fail_sequence[-1] ##Key!!!!!: the effect of failure is caused by failure of node at last time step
        # node_fail_sequence = self.node_fail_final[-1] ##Key!!!!!: the effect of failure is caused by failure of nodes at all time step
        
        for i in range(self.nodenum):
            temp = 0
            for j in range(self.nodenum):
                if(i == j):
                    continue
                temp += self.adjmatrix[j, i]*np.log(1 - self.fail_prop_matrix[j, i])*node_fail_sequence[j]
            node_fail_prob[i] = 1 - np.exp(temp)
            
        self.node_fail_prob.append(node_fail_prob)
    
    def failure_sequence(self):
        """Simulate one further node failure sceneria based on MC simulation
        Input: the node failure sceneria at the previous step, the node_fail_probability at previous step
        Output: the node failure sceneria at the current step
        """
        import numpy as np
        
        node_fail_sequence = np.zeros(self.nodenum, dtype = int)
        node_fail_prob = self.node_fail_prob[-1]
        
        for i in range(self.nodenum):
            if(self.node_fail_final[-1][i] == 0):
                temp = np.random.rand()
                if(temp < node_fail_prob[i]):
                    node_fail_sequence[i] = 1
        
        self.node_fail_sequence.append(node_fail_sequence)
    
    def generate_initial_failure(self, initial_failure_num, seed):
        """Generate the initial node failure scenaria
        Input:
            initial_failure_num: the number of initial failed nodes
            seed: the variable controling the randomness of the initial failed nodes
        Output: the initial failure sequence
        """
        import numpy as np
        
        initial_node_failure = np.zeros(self.nodenum)
        temp = np.random.randint(self.nodenum, size = initial_failure_num)
        initial_node_failure[temp] = 1
        
        self.node_fail_sequence.append(initial_node_failure)
        self.node_fail_final.append(initial_node_failure)
    
    def failure_simulation(self):
        """ Simulate the node failure evoluation along the time
        Output:
            Node failure sequence, node failure probability
        """
        import numpy as np
        
        self.edge_failure_matrix()
        
        while(1):#keep evoluation until there are no newly failed nodes
            self.failure_probability()
            self.failure_sequence()
            
            node_fail_final = np.zeros(self.nodenum)
            
            for i in range(self.nodenum):
                if(self.node_fail_final[-1][i] == 1 or self.node_fail_sequence[-1][i] == 1):
                    node_fail_final[i] = 1
            self.node_fail_final.append(node_fail_final)
            
            if((self.node_fail_final[-1] == self.node_fail_final[-2]).all() or (np.sum(self.node_fail_final[-1]) == self.nodenum)):
                break