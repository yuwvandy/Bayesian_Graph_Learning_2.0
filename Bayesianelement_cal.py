# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:16:06 2020

@author: 10624
"""
def prior1(fail_seq_data, system):
    """Come up with the 1st type of prior: no consideration of phyiscal constraints, just randomly pick a node pair
    Input:
        fail_seq_data: the failure sequence data, list array
        system: the object of the infrastructure system, object
    Output: the modified initial prior of system adjacent matrix where the initial likelihood is nonzero
    
    Basic idea: take the original adjacent matrix and add links between two failed nodes at last and current time steps
    """
    import copy
    
    for i in range(len(fail_seq_data)):
        for j in range(len(fail_seq_data[i]) - 1):
            node_failed = []
            
            for k in range(len(fail_seq_data[i][j])):
                if(fail_seq_data[i][j][k] == 1):
                    node_failed.append(k)
            
            for k in range(len(fail_seq_data[i][j + 1])):
                if(fail_seq_data[i][j + 1][k] == 1):
                    node = node_failed[np.random.randint(len(node_failed), size = 1)]
                    block_system.adjmatrix[node, k] = 1
    
    return copy.deepcopy(block_system.adjmatrix)
    
    
def prior2(fail_seq_data, system, network2internetwork):
    """Come up with the 2nd type of prior: considering physical constraints
    Input:
        fail_seq_data: the failure sequence data, list array
        system: the object of the infrastructure system
    Output: the modified initial prior of system adjacent matrix where the initial likelihood is nonzero
    
    Basic idea: take the original adjacent matrix and add links between two failed nodes at last and the current time step,
    !Meanwhile, the following properties should be guaranteed
    
    In each individual network:
        1) every supply node has a demand node for their resourecs to go
        2) every demand node has a supply node for providing resources
        3) every transmission node has a supply node and demand node for obtaining and distributing resources
        4) there is no cycle
    In each interdependent network:
        1) every supply node has at least one demand node
        2) every demand node has at least one supply node
    """
    import copy
    import numpy as np
    import sharefunction as sf
    
    for i in range(len(fail_seq_data)):
        for j in range(len(fail_seq_data[i]) - 1):
            node_failed = []
            
            for k in range(len(fail_seq_data[i][j])):
                if(fail_seq_data[i][j][k] == 1):
                    node_failed.append(k)
    
            for k in range(len(fail_seq_data[i][j + 1])):
                if(fail_seq_data[i][j + 1][k] == 1):
                    #If at time t + 1 node k fails, than it must be connected to at least one of the failed nodes at time t step, randomly choose one
                    for node in node_failed:
                        if(system.nodetable[node, 0] == system.nodetable[k, 0]):
                            if(system.nodetable[k, 1] == 0):
                                continue
                            else:
                                if(system.nodetable[node, 1] > system.nodetable[k, 1]):
                                    continue
                                else:
                                    network = system.networks[system.nodetable[node, 0]]
                                    node2, kk = node - network.start_num, k - network.start_num
                                    network.adjmatrix[node2, kk] = 1
                                    
                                    traverse = np.zeros(network.nodenum, dtype = int)
                                    stack = []
                                    flag = sf.check_cycle(network.adjmatrix, node2, traverse, stack)
                                    
                                    if(flag == 1):
                                        system.adjmatrix[node, k] = 1
                                        break
                                    else:
                                        network.adjmatrix[node2, kk] = 0
                                        continue
                        else:
                            internetwork = network2internetwork[system.nodetable[node, 0], system.nodetable[k, 0]]
                            
                            if(internetwork == None):
                                continue
                            else:
                                node_series = internetwork.network1.type[system.nodetable[node, 1]]
                                k_series = internetwork.network2.type[system.nodetable[k, 1]]
                                
                                if(np.isin(node_series, internetwork.supplyseries).all() and np.isin(k_series, internetwork.demandseries).all()):
                                    system.adjmatrix[node, k] = 1
                                    internetwork.adjmatrix[system.nodetable[node, 2] - node_series[0], system.nodetable[k, 2] - k_series[0]] = 1
                                    break
                                else:
                                    continue
    return copy.deepcopy(system.adjmatrix)

def likelihood(sigma, adjmatrix, fail_prop_matrix):
    """Calculate the likelihood of the certain failure sequence given the adjmatrix and the conditioanl failure probability
    When the data is too much, we use log to simplify the computation and avoid numerical error
    Input:
        sigma: the failure sequence data
        adjmatrix: the adjacent matrix of the network
        fail_prop_matrix: the conditional failure probability
    """
    import numpy as np
    
    like2 = 1
    log_like2 = 0
    
    for i in range(len(sigma) - 1):
        for j in range(len(adjmatrix)):
            fail_prob = 1
            flag = 1
            for k in range(len(adjmatrix)):
                if(adjmatrix[k, j] == 1 and sigma[i][k] == 1):
                    fail_prob = fail_prob*(1 - fail_prop_matrix[k, j])
                    flag = 0
            
            temp = ((1 - fail_prob)**sigma[i + 1][j]*fail_prob**(1 - sigma[i + 1][j]))**(1 - sigma[i][j])
            
            if(temp == 0):
                return None, None, False
            
            like2 = like2*temp
            log_like2 += np.log(temp)
    
            
    return like2, log_like2, True

def proposal1(adjmatrix, edge_prob_matrix):
    """Proposal the new edge randomly and calculate the ratio of the prior probability
    Input:
        adjmatrix: the adjmatrix of the current iteration
        edge_prob_matrix: the edge probability between multiple types of nodes
    Output:
        the prior probabillity ratio
    """
    import numpy as np
    import copy
    
    adjmatrix2 = copy.deepcopy(adjmatrix)
    
    while(1):
        i, j = np.random.randint(len(adjmatrix), size = 2)
        if(i != j and edge_prob_matrix[i, j] != 0):
            break
    
    if(adjmatrix2[i, j] == 1):
        adjmatrix2[i, j] = 0
        priorratio = (1 - edge_prob_matrix[i, j])/edge_prob_matrix[i, j]
    else:
        adjmatrix2[i, j] = 1
        priorratio = edge_prob_matrix[i, j]/(1 - edge_prob_matrix[i, j])
    
    return adjmatrix2, priorratio

def proposal2(adjmatrix, system, candidate_edge, network2internetwork):
    """Proposal a new edge by randomly selecting a node from a parition and another node from another partition and add the edge
    Notice: after add the edge, three properties are checked to see whether the proposal is accepted or rejected
    Input:
        system
        candidate_edge: the set where we sample the edge
        network2internetwork: internetwork dictionary
    Output:
        priorratio: the prior probabillity ratio - P(G)/P(G')
        adjmatrix2: the adjmatrix of the next iteration
        flag: binary variable deciding whether we accept the proposed edge
    """
    import numpy as np
    import copy
    import random
    import sharefunction as sf
    
    flag = 1 #If the proposal cannot satisfy the three properties and acyclic, the flag is 0
    adjmatrix2 = copy.deepcopy(adjmatrix)
    
    group = random.choice(candidate_edge)

    while(1):
        i, j = random.choice(list(system.nodetypeseries[group[0]])), random.choice(list(system.nodetypeseries[group[1]]))
        if(i != j):
            break
    
    if(adjmatrix2[i, j] == 1):
        adjmatrix2[i, j] = 0
        
        if(system.nodetable[i, 0] == system.nodetable[j, 0]):
            
            network = system.networks[system.nodetable[i, 0]]
            
            #update the adjacent matrix in each individual network
            ii, jj = i - network.start_num, j - network.start_num
            network.adjmatrix[ii, jj] = 0
            
            for m in network.supplyseries:
                if(network.DFS_supply2demand(m) == 0):
                    flag = 0
                    break
            
            if(flag == 1):
                for m in network.demandseries:
                    if(network.DFS_demand2supply(m) == 0):
                        flag = 0
                        break
            
            if(flag == 1):
                for m in network.transeries:
                    if(network.DFS_tran2supply(m) == 0):
                        flag = 0
                        break
                    if(network.DFS_tran2demand(m) == 0):
                        flag = 0
                        break

            if(flag == 0):
                network.adjmatrix[ii, jj] = 1 #read the edge we delete so the effect will not proceed to the next iteration
        else:
            internetwork = network2internetwork[system.nodetable[i, 0], system.nodetable[j, 0]]
            ii = i - system.nodetypeseries[group[0]][0] ##!!!! different when (i, j) in the same network
            jj = j - system.nodetypeseries[group[1]][0]
            internetwork.adjmatrix[ii, jj] = 0
            
            ##To ensure the property 1
            for m in range(len(internetwork.supplyseries)):
                if(np.sum(internetwork.adjmatrix[m, :]) == 0):
                    flag = 0
                    break
            
            if(flag == 1):
                ##To ensure the property 2
                for m in range(len(internetwork.demandseries)):
                    if(np.sum(internetwork.adjmatrix[:, m]) == 0):
                        flag = 0
                        break
            
            if(flag == 0):
                internetwork.adjmatrix[ii, jj] = 1 ##re-add the edge we delete so the effect will not proceed to next iteration
            
        if(flag == 1):
            priorratio = (1 - system.edge_prob_matrix[i, j])/system.edge_prob_matrix[i, j]
            return adjmatrix2, priorratio, flag, i, j
        else:
            adjmatrix2[i, j] = 1
            return None, None, flag, i, j
    else:
        adjmatrix2[i, j] = 1
        #check cycle
        if(system.nodetable[i, 0] == system.nodetable[j, 0] and system.nodetable[i, 1] == system.nodetable[j, 1]):
            network = system.networks[system.nodetable[i, 0]]
            traverse = np.zeros(network.nodenum, dtype = int)
            stack = []
            ii, jj = i - network.start_num, j - network.start_num
            network.adjmatrix[ii, jj] = 1
            flag = sf.check_cycle(network.adjmatrix, ii, traverse, stack)

            if(flag == 1):
                priorratio = system.edge_prob_matrix[i, j]/(1 - system.edge_prob_matrix[i, j])
                return adjmatrix2, priorratio, flag, i, j
            else:
                network.adjmatrix[ii, jj] = 0
                adjmatrix2[i, j] = 0
                return None, None, flag, i, j
        
        elif(system.nodetable[i, 0] == system.nodetable[j, 0] and system.nodetable[i, 1] != system.nodetable[j, 1]):
            network = system.networks[system.nodetable[i, 0]]
            ii, jj = i - network.start_num, j - network.start_num
            network.adjmatrix[ii, jj] = 1
            
            flag = 1
            priorratio = system.edge_prob_matrix[i, j]/(1 - system.edge_prob_matrix[i, j])
            return adjmatrix2, priorratio, flag, i, j
        
        else:
            internetwork = network2internetwork[system.nodetable[i, 0], system.nodetable[j, 0]]
            ii = i - system.nodetypeseries[group[0]][0] ##!!!! different when (i, j) in the same network
            jj = j - system.nodetypeseries[group[1]][0]
            internetwork.adjmatrix[ii, jj] = 1
            
            flag = 1
            priorratio = system.edge_prob_matrix[i, j]/(1 - system.edge_prob_matrix[i, j])
            return adjmatrix2, priorratio, flag, i, j
        

def MCMC_MH(experiment_num, prior_adjmatrix, num, system, fail_seq_data, network2internetwork, sc_system):
    """ Perform the Metropoli Hasting sampling to obtain the MCMC chain
    Input:
        prior_adjmatrix: the initial prior adjmatrix, note: for different initialization, we further randomly move from the initial prior adjmatrix
        num: the number of the MCMC chain
        system: the object of the interdependent system we want to infer
        fail_seq_data: the failure sequence data we use to update our prior
        network2internetwork: the mapping from network2internetwork
        experiment_num: the number of the experiment that is currently in proceeding
        sc_system: the target system we want to infer
    
    Output:
        adjlist: a single MCMC chain of adjacent matrix
    """
    import copy
    import numpy as np
    import data as dt
    
    adj_list = []
    adjmatrix = copy.deepcopy(prior_adjmatrix)
    
    adj_list.append(adjmatrix)
    plike2_1 = np.empty(len(fail_seq_data), dtype = float)
    plike2_2 = np.empty(len(fail_seq_data), dtype = float)
    
    log_plike2_1 = np.empty(len(fail_seq_data), dtype = float)
    log_plike2_2 = np.empty(len(fail_seq_data), dtype = float)
    for i in range(len(fail_seq_data)):
        plike2_1[i], log_plike2_1[i], accept_flag = likelihood(fail_seq_data[i], adjmatrix, system.fail_prop_matrix)

    while(len(adj_list) <= num):
        # adjmatrix2, priorratio = beycal.proposal1(adjmatrix, block_system.edge_prob_matrix)
        adjmatrix2, priorratio, flag, i, j = proposal2(adjmatrix, system, dt.candidate_edge, network2internetwork)
        # print(accept_ratio, i, j)
        if(flag == 0):
            continue
        
        

        accept_ratio = priorratio
        log_accept_ratio = np.log(priorratio)
        
        for i in range(len(fail_seq_data)):
            plike2_2[i], log_plike2_2[i], accept_flag = likelihood(fail_seq_data[i], adjmatrix2, system.fail_prop_matrix)
            if(accept_flag == False):
                break
#            accept_ratio = accept_ratio*plike2_2[i]/plike2_1[i]
            log_accept_ratio += log_plike2_2[i] - log_plike2_1[i]
#            print(plike2_2[i])
        
#        if(np.random.rand() < accept_ratio):
        if(accept_flag == True):
            if(np.log(np.random.rand()) < log_accept_ratio):
                plike2_1 = copy.deepcopy(plike2_2)
                log_plike2_1 = copy.deepcopy(log_plike2_2)
                
                adjmatrix = adjmatrix2
                adj_list.append(adjmatrix)
                print(np.sum(adjmatrix)/(system.nodenum**2), len(adj_list), np.sum(sc_system.adjmatrix)/(sc_system.nodenum**2), experiment_num)
            
        
    return adj_list