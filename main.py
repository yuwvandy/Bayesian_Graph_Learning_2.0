import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import data as dt
import sharefunction as sf
from shelbycountyclass import shelbycounty
from Block import block




#Initialize the three shelby county networks
sc_water = shelbycounty(dt.sc_water_data["name"], dt.sc_water_data["nodenum"], dt.sc_water_data["supplynum"], dt.sc_water_data["trannum"], dt.sc_water_data["demandnum"], dt.sc_water_data["color"], dt.sc_water_data["supplyname"], dt.sc_water_data["tranname"], dt.sc_water_data["demandname"], dt.sc_water_data["fail_prop_matrix"])
sc_power = shelbycounty(dt.sc_power_data["name"], dt.sc_power_data["nodenum"], dt.sc_power_data["supplynum"], dt.sc_power_data["trannum"], dt.sc_power_data["demandnum"], dt.sc_power_data["color"], dt.sc_power_data["supplyname"], dt.sc_power_data["tranname"], dt.sc_power_data["demandname"], dt.sc_power_data["fail_prop_matrix"])
sc_gas = shelbycounty(dt.sc_gas_data["name"], dt.sc_gas_data["nodenum"], dt.sc_gas_data["supplynum"], dt.sc_gas_data["trannum"], dt.sc_gas_data["demandnum"], dt.sc_gas_data["color"], dt.sc_gas_data["supplyname"], dt.sc_gas_data["tranname"], dt.sc_gas_data["demandname"], dt.sc_gas_data["fail_prop_matrix"])

sc_networks = [sc_water, sc_power, sc_gas]
for i in range(len(sc_networks)):
    sc_networks[i].adjmatrix(dt.sc_data[i]["edge_path"])
    sc_networks[i].edgeprob()
    dt.sc_data[i]["edgeprob"] = sc_networks[i].edge_prob
    
    #simulate the failure evoluation
    sc_networks[i].failinitialize()
    # initial_fail_num = np.random.randint(sc_networks[i].nodenum)
    initial_fail_num = 8
    sc_networks[i].generate_initial_failure(initial_fail_num, dt.seed)
    sc_networks[i].failure_simulation()
    

#Initialize the three self-defined networks
block_water = block(dt.water_data["name"], dt.water_data["supplynum"], dt.water_data["trannum"], dt.water_data["demandnum"], sc_water.edge_prob)
block_power = block(dt.power_data["name"], dt.power_data["supplynum"], dt.power_data["trannum"], dt.power_data["demandnum"], sc_power.edge_prob)
block_gas = block(dt.gas_data["name"], dt.gas_data["supplynum"], dt.gas_data["trannum"], dt.gas_data["demandnum"], sc_gas.edge_prob)

networks = [block_water, block_power, block_gas]
for i in range(len(networks)):
    networks[i].adj_matrix()
