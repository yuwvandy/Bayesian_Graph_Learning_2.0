# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:16:06 2020

@author: 10624
"""

def likelihood(sigma, adjmatrix, fail_prop_matrix):
    """Calculate the likelihood of the certain failure sequence given the adjmatrix and the conditioanl failure probability
    Input:
        sigma: the failure sequence data
        adjmatrix: the adjacent matrix of the network
        fail_prop_matrix: the conditional failure probability
    """
    like2 = 1
    
    for i in range(len(sigma) - 1):
        for j in range(len(adjmatrix)):
            fail_prob = 1
            flag = 1
            for k in range(len(adjmatrix)):
                if(adjmatrix[k, j] == 1 and sigma[i][k] == 1):
                    fail_prob = fail_prob*(1 - fail_prop_matrix[k, j])
                    flag = 0
            
            temp = ((1 - fail_prob)**sigma[i + 1][j]*fail_prob**(1 - sigma[i + 1][j]))**(1 - sigma[i][j])
            
            like2 = like2*temp
            
    
    return like2

def proposal(adjmatrix, edge_prob_matrix):
    """Proposal the new point and calculate the ratio of the prior probability
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
        
    