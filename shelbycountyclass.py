class shelbycounty(object):
    def __init__(self, network_data):
        """Set up the object of Shelby County
        Input:
            network_data: the data of the network
        """
        import numpy as np
        
        self.name = sc_network_data["name"]
        self.nodenum = sc_network_data["nodenum"]
        self.color = sc_network_data["color"]
        self.fail_prop = sc_network_data["fail_prop_matrix"]
        
        self.supplynum, self.trannum, self.demandnum = sc_network_data["supplynum"], sc_network_data["trannum"], sc_network_data["demandnum"]
        self.supplyname, self.tranname, self.demandname = sc_network_data["supplyname"], sc_network_data["tranname"], sc_network_data["demandname"]

        self.supplyseries = np.arange(0, self.supplynum, 1)
        self.transeries = np.arange(self.supplynum, self.supplynum + self.trannum, 1)
        self.demandseries = np.arange(self.supplynum + self.trannum, self.nodenum, 1)
        self.nodeseries = np.arange(0, self.nodenum, 1)
        
        self.type = [self.supply_series, self.tran_series, self.demand_series]

    def adjmatrix(self, edgepath):
        """Set up the adjacent matrix of the network
        Input:
            edgepath - the path of the edge list used to load the data
        Output:
            adjmatrix and edgelist
        """
        import sharefunction as sf

        self.edgelist = sf.readedge(edgepath)
        self.adjmatrix = sf.edgelist2matrix(self.edgelist, self.nodenum)

    def edgeprob(self):
        """Calculate the edge probability within single partite nodes and between different types of nodes
        Output:
            a 3*3 matrix
        """
        import numpy as np

        self.edge_prob = np.zeros((3,3), dtype = float)
        

        for i in range(len(self.edge_prob)):
            for j in range(i, len(self.edge_prob)):
                temp = 0
                for m in self.type[i]:
                    for n in self.type[j]:
                        if(self.adjmatrix[m, n] == 1):
                            temp += 1

                if(i != j):
                    self.edge_prob[i, j] = temp/(len(self.type[i])*len(self.type[j]))
                if(i == j):
                    self.edge_prob[i, j] = 2*temp/(len(self.type[i])*(len(self.type[j]) - 1))
    
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
        
        self.fail_prop_matrix = np.zeros((self.nodenum, self.nodenum), dtype = float)
        
        for i in range(len(self.fail_prop)):
            for j in range(len(self.fail_prop)):
                for m in self.type[i]:
                    for n in self.type[j]:
                        if(m == n):
                            self.fail_prop_matrix[m, n] = 1
                        else:
                            self.fail_prop_matrix[m, n] = self.fail_prop[i, j]
    
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
    
    
            
        




