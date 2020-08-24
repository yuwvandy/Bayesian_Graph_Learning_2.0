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

    #Initialize the four interdependent networks
    block_inter_wd2ps = interblock(internetwork_data[0], block_networks)
    block_inter_gd2ps = interblock(internetwork_data[1], block_networks)
    block_inter_pd2w = interblock(internetwork_data[2], block_networks)
    block_inter_pd2g = interblock(internetwork_data[3], block_networks)

    block_internetworks = [block_inter_wd2ps, block_inter_gd2ps, block_inter_pd2w, block_inter_pd2g]
    for i in range(len(block_internetworks)):
        block_internetworks[i].adj_matrix()
        block_internetworks[i].failpropmatrix()
            
    block_system = block_intersystem(block_networks, block_internetworks)
    block_system.adj_matrix()
    block_system.edge_failure_matrix()
    
    return block_system, block_networks, block_internetworks

block_system, block_networks, block_internetworks = initial_prior(dt.block_data, dt.block_inter_data, edge_prob)



