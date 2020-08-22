import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import data as dt
import sharefunction as sf
from shelbycountyclass import shelbycounty
from Block import block
from sc_internetwork import internetwork
from intersystem import sbm




#Initialize the three shelby county networks
sc_water = shelbycounty(dt.sc_water_data)
sc_power = shelbycounty(dt.sc_power_data)
sc_gas = shelbycounty(dt.sc_gas_data)

sc_networks = [sc_water, sc_power, sc_gas]
for i in range(len(sc_networks)):
    sc_networks[i].adj_matrix(dt.sc_data[i]["edge_path"])
    sc_networks[i].edgeprob()
    dt.sc_data[i]["edgeprob"] = sc_networks[i].edge_prob
    
    ##If we want to simulate the failure scenario in each individual network
    # #simulate the failure evoluation
    # sc_networks[i].failinitialize()
    # # initial_fail_num = np.random.randint(sc_networks[i].nodenum)
    # initial_fail_num = 8
    # sc_networks[i].generate_initial_failure(initial_fail_num, dt.seed)
    # sc_networks[i].failure_simulation()

#Initialize the four interdependent networks
sc_inter_wd2ps = internetwork(dt.inter_wd2ps, sc_networks, 0, sc_water.nodenum)
sc_inter_gd2ps = internetwork(dt.inter_gd2ps, sc_networks, sc_water.nodenum + sc_gas.nodenum, sc_water.nodenum)
sc_inter_pd2w = internetwork(dt.inter_pd2w, sc_networks, sc_water.nodenum, 0)
sc_inter_pd2g = internetwork(dt.inter_pd2g, sc_networks, sc_water.nodenum, sc_water.nodenum + sc_power.nodenum)

sc_internetworks = [sc_inter_wd2ps, sc_inter_gd2ps, sc_inter_pd2w, sc_inter_pd2g]
for i in range(len(sc_internetworks)):
    sc_internetworks[i].adj_matrix()

#Initialize the system
sc_system = sbm(sc_networks, sc_internetworks)
sc_system.adj_matrix()




####----------------------------------------Bayesian inference for sampling prior graph
#Initialize the three self-defined networks
block_water = block(dt.water_data["name"], dt.water_data["supplynum"], dt.water_data["trannum"], dt.water_data["demandnum"], sc_water.edge_prob)
block_power = block(dt.power_data["name"], dt.power_data["supplynum"], dt.power_data["trannum"], dt.power_data["demandnum"], sc_power.edge_prob)
block_gas = block(dt.gas_data["name"], dt.gas_data["supplynum"], dt.gas_data["trannum"], dt.gas_data["demandnum"], sc_gas.edge_prob)

networks = [block_water, block_power, block_gas]
for i in range(len(networks)):
    networks[i].adj_matrix()
