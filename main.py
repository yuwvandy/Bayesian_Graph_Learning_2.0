import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import data as dt
import sharefunction as sf
from shelbycountyclass import shelbycounty
from Block import block
from Interblock import interblock
from Sc_internetwork import sc_internetwork
from Sc_intersystem import sc_intersystem
from Block_intersystem import block_intersystem
import Bayesianelement_cal as beycal
import copy



#Initialize the three shelby county networks
sc_water = shelbycounty(dt.sc_water_data)
sc_power = shelbycounty(dt.sc_power_data)
sc_gas = shelbycounty(dt.sc_gas_data)

sc_networks = [sc_water, sc_power, sc_gas]
edge_prob = []
for i in range(len(sc_networks)):
    sc_networks[i].adj_matrix(dt.sc_data[i]["edge_path"])
    sc_networks[i].edgeprob()
    dt.sc_data[i]["edgeprob"] = sc_networks[i].edge_prob
    edge_prob.append(sc_networks[i].edge_prob)
    sc_networks[i].edge_failure_matrix()
    
    ##If we want to simulate the failure scenario in each individual network
    # #simulate the failure evoluation
    # sc_networks[i].failinitialize()
    # # initial_fail_num = np.random.randint(sc_networks[i].nodenum)
    # initial_fail_num = 8
    # sc_networks[i].generate_initial_failure(initial_fail_num, dt.seed)
    # sc_networks[i].failure_simulation()

#Initialize the four interdependent networks
sc_inter_wd2ps = sc_internetwork(dt.inter_wd2ps, sc_networks)
sc_inter_gd2ps = sc_internetwork(dt.inter_gd2ps, sc_networks)
sc_inter_pd2w = sc_internetwork(dt.inter_pd2w, sc_networks)
sc_inter_pd2g = sc_internetwork(dt.inter_pd2g, sc_networks)

sc_internetworks = [sc_inter_wd2ps, sc_inter_gd2ps, sc_inter_pd2w, sc_inter_pd2g]
for i in range(len(sc_internetworks)):
    sc_internetworks[i].adj_matrix()
    sc_internetworks[i].failpropmatrix()

#Initialize the system
sc_system = sc_intersystem(sc_networks, sc_internetworks)
sc_system.adj_matrix()
sc_system.edge_failure_matrix()

#Simulate the failure of the whole system several times
##The fail_seq_data is used to update the Bayesian posterior
def faildata_simulation(time, sc_system, seed):
    """Simulate the failure evolution data with given time
    Input:
        time - the amount of simulation
        sc_system - the object where we perform the failure simulation
        seed - the parameter control the randomness of the initial failure scenario
    Output:
        fail_seq_data
    """
    temp = 0
    fail_seq_data = []
    while(temp <= time):
        sc_system.failinitialize()
        initial_fail_num = np.random.randint(sc_system.nodenum)
        # initial_fail_num = 8
        sc_system.generate_initial_failure(initial_fail_num, seed)
        sc_system.failure_simulation()
        
        if(np.sum(sc_system.node_fail_sequence[0]) == 0):
            continue
        
        fail_seq_data.append(sc_system.node_fail_sequence)
        temp += 1
    
    return fail_seq_data


####----------------------------------------Bayesian inference for sampling prior graph
#Initialize the three self-defined networks
def initial_prior(network_data, internetwork_data, edge_prob):
    """ Generate the initial prior system to start the MCMC
    Input:
        network_data: the data of the three networks
        inter_network_data: the data of the four internetworks
        edge_prob: the edge probability to sample edges obtained from shelby county networks
    """
    block_water = block(network_data[0], edge_prob[0])
    block_power = block(network_data[1], edge_prob[1])
    block_gas = block(network_data[2], edge_prob[2])

    block_networks = [block_water, block_power, block_gas]
    for i in range(len(block_networks)):
        block_networks[i].adj_matrix()
        block_networks[i].edge_failure_matrix()
        block_networks[i].edgeprobmatrix()

    #Initialize the four interdependent networks
    block_inter_wd2ps = interblock(internetwork_data[0], block_networks)
    block_inter_gd2ps = interblock(internetwork_data[1], block_networks)
    block_inter_pd2w = interblock(internetwork_data[2], block_networks)
    block_inter_pd2g = interblock(internetwork_data[3], block_networks)

    block_internetworks = [block_inter_wd2ps, block_inter_gd2ps, block_inter_pd2w, block_inter_pd2g]
    for i in range(len(block_internetworks)):
        block_internetworks[i].adj_matrix()
        block_internetworks[i].failpropmatrix()
        block_internetworks[i].edgeprobmatrix()
            
    block_system = block_intersystem(block_networks, block_internetworks)
    block_system.adj_matrix()
    block_system.edge_failure_matrix()
    block_system.edgeprobmatrix()
    return block_system, block_networks, block_internetworks


##Generate failure sequence data based on simulation
data_num = 5
fail_seq_data = faildata_simulation(data_num, sc_system, seed = 1)

##MCMC: Metropolis hasting algorithm
experiment_num = 5
initial_random_num = 50
num = 5000

experiment = []
for i in range(experiment_num):
    ##Generate the initial graph topology by first samplying system and then adding edges based on failure sequence data
    block_system, block_networks, block_internetworks = initial_prior(dt.block_data, dt.block_inter_data, edge_prob)
    
    #network2internetwork dictionary
    network2internetwork = np.array([[None, block_internetworks[0], None], [block_internetworks[2], None, block_internetworks[3]], [None, block_internetworks[1], None]])
    
    #It should be noticed that the initial prior should guarantee the likelihood is not 0 so that MCMC chain can proceed to converge
    prior_adjmatrix = beycal.prior2(fail_seq_data, block_system, network2internetwork)
    
    #MCMC
    adj_list = beycal.MCMC_MH(i, prior_adjmatrix, initial_random_num, num, block_system, fail_seq_data, network2internetwork, sc_system)
    experiment.append(adj_list)
    print("experiment {} ends".format(i))




# experiment = []
# while(len(experiment) <= 10):
#     adj_list = []
#     adjmatrix = copy.deepcopy(block_system.adjmatrix)
#     #Randomly interference so that we have random starting points
#     temp = 0
#     Temp = np.random.randint(100)
#     while(temp <= Temp):
#         while(1):
#             i, j = np.random.randint(block_system.nodenum, size = 2)
#             if(i != j and adjmatrix[i, j] == 0):
#                 break
#         adjmatrix[i, j] = 1
#         temp += 1
    
#     adj_list.append(adjmatrix)
#     plike2_1 = np.empty(len(fail_seq_data), dtype = float)
#     plike2_2 = np.empty(len(fail_seq_data), dtype = float)
#     for i in range(len(fail_seq_data)):
#         plike2_1[i] = beycal.likelihood(fail_seq_data[i], adjmatrix, block_system.fail_prop_matrix)
    
#     while(len(adj_list) <= 3000):
#         # adjmatrix2, priorratio = beycal.proposal1(adjmatrix, block_system.edge_prob_matrix)
#         adjmatrix2, priorratio, flag, i, j = beycal.proposal2(adjmatrix, block_system, dt.candidate_edge, network2internetwork)
#         # print(accept_ratio, i, j)
#         if(flag == 0):
#             continue

#         accept_ratio = priorratio
#         for i in range(len(fail_seq_data)):
#             plike2_2[i] = beycal.likelihood(fail_seq_data[i], adjmatrix2, block_system.fail_prop_matrix)
#             accept_ratio = accept_ratio*plike2_2[i]/plike2_1[i]
        
        
#         if(np.random.rand() < accept_ratio):
#             plike2_1 = copy.deepcopy(plike2_2)
#             adjmatrix = adjmatrix2
#             adj_list.append(adjmatrix)
#             print(np.sum(adjmatrix)/(125*125), len(adj_list), np.sum(sc_system.adjmatrix)/(125*125))

#     experiment.append(adj_list)
#     # print("experiment one more")
    


##Given the MCMC chain - adj_list, post-procession
degree_list = []
for i in range(len(experiment[1])):
    degree_list.append(np.sum(experiment[1][i])/(sc_system.nodenum*sc_system.nodenum))

import matplotlib.pyplot as plt
plt.plot(np.arange(0, len(degree_list), 1), degree_list)
plt.xlabel("Iteration number", fontsize = 15, weight = "bold")
plt.ylabel("Degree value", fontsize = 15, weight = "bold")

adj_heatmap = sf.cal_adjmatrix_heatmap(experiment[0], sc_system.adjmatrix, warm_up = 2000)
import seaborn as sns
sns.heatmap(sc_system.adjmatrix, cmap = "cividis")
sns.heatmap(adj_heatmap)

#threshold to generate the eventual adjacent matrix given adj_heatmap
accuracylist = []
accuracylist = []
accuracylist = []
accuracylist = []
for threshold in np.arange(0, 1.01, 0.01):
    infer_adjmatrix = np.zeros((sc_system.nodenum, sc_system.nodenum), dtype = int)
    for i in range(sc_system.nodenum):
        for j in range(sc_system.nodenum):
            if(adj_heatmap[i, j] >= threshold):
                infer_adjmatrix[i, j] = 1
    
    accuracy, precision, recall, F1 = performance(infer_adjmatrix, sc_system.adjmatrix)
    




