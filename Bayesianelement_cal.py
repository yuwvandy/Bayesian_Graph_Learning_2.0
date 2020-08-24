# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 14:16:06 2020

@author: 10624
"""

def likelihood(sigma, adjmatrix, fail_prob_matrix):
    """Calculate the likelihood of the certain failure sequence given the adjmatrix and the conditioanl failure probability
    Input:
        sigma: the failure sequence data
        adjmatrix: the adjacent matrix of the network
        fail_prob_matrix: the conditional failure probability
    """
    like2 = 1
    
    for i in range(len(sigma) - 1):
        for j in range(len(adjmatrix)):
            fail_prob = 1
            flag = 1
            for k in range(len(adjmatrix)):
                if(adjmatrix[k, j] == 1 and sigma[i][k] == 1):
                    fail_prob = fail_prob*(1 - fail_prob_matrix[k, j])
                    flag = 0
            
            temp = ((1 - fail_prob)**sigma[i + 1][j]*fail_prob**(1 - sigma[i + 1][j]))**(1 - sigma[i][j])
            
            like2 = like2*temp
    
    return like2

def proposal