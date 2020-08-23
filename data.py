"""This code file specifies all data used in the Bayesian learning for inferrning network 
"""
import numpy as np

###########-----------------------------Data for modeling Shelby County network
sc_water_data = {"name": "sc_water",
            "supplyname": "Pumping station",
            "tranname": "Storage tank",
            "demandname": "Delivery station",
            "nodenum": 49,
            "supplynum": 9,
            "trannum": 6,
            "demandnum": 34,
            "color": "blue",
            "edge_path": "./data/wateredges.xlsx",
            "fail_prop_matrix": np.array([[0.3, 0.3, 0.3],
                                          [0, 0.3, 0.3],
                                          [0, 0, 0.3]])
            }

sc_power_data = {"name": "sc_power",
            "supplyname": "Power gate station",
            "tranname": "23kv substation",
            "demandname": "12kv substation",
            "nodenum": 60,
            "supplynum": 9,
            "trannum": 14,
            "demandnum": 37,
            "color": "red",
            "edge_path": "./data/poweredges.xlsx",
            "fail_prop_matrix": np.array([[0.3, 0.3, 0.3],
                                          [0, 0.3, 0.3],
                                          [0, 0, 0.3]])
            }

sc_gas_data = {"name": "sc_gas",
            "supplyname": "Gas gate station",
            "tranname": "Regulator station",
            "demandname": "Other",
            "nodenum": 16,
            "supplynum": 3,
            "trannum": 7,
            "demandnum": 6,
            "color": "green",
            "edge_path": "./data/gasedges.xlsx",
            "fail_prop_matrix": np.array([[0.3, 0.3, 0.3],
                                          [0, 0.3, 0.3],
                                          [0, 0, 0.3]])
            }

sc_data = [sc_water_data, sc_power_data, sc_gas_data]

##########----------------------------Data for interdependent links between the above three networks
##Water demand -> power supply
inter_wd2ps = {"name": "wdemand2psupply",
               "network1": 0, #which network in networks object list, starting from 0
               "network2": 1,
               "from": [2], #which type of nodes in network1 that nodes in network2 depend on
               "to": [0],
               "supply_start_num": 0,
               "demand_start_num": sc_water_data["nodenum"],
               "edge_prob": 0.1,
               "fail_prop": 0.08
               }

##Gas demand -> power supply
inter_gd2ps = {"name": "gdemand2psupply",
               "network1": 2,
               "network2": 1,
               "from": [2],
               "to": [0],
               "supply_start_num": sc_water_data["nodenum"] + sc_power_data["nodenum"],
               "demand_start_num": sc_water_data["nodenum"],
               "edge_prob": 0.1,
               "fail_prop": 0.08
               }

##Power demand -> all water nodes
inter_pd2w = {"name": "pdemand2water",
               "network1": 1,
               "network2": 0,
               "from": [2],
               "to": [0, 1, 2],
               "supply_start_num": sc_water_data["nodenum"],
               "demand_start_num": 0,
               "edge_prob": 0.1,
               "fail_prop": 0.08
               }

##Power demand -> all gas nodes
inter_pd2g = {"name": "pdemand2gas",
               "network1": 1,
               "network2": 2,
               "from": [2],
               "to": [0, 1, 2],
               "supply_start_num": sc_water_data["nodenum"],
               "demand_start_num": sc_water_data["nodenum"] + sc_power_data["nodenum"],
               "edge_prob": 0.1,
               "fail_prop": 0.08
               }

sc_inter_data = [inter_wd2ps, inter_gd2ps, inter_pd2w, inter_pd2w]



#####data for failure simulation
seed = 1

###########-----------------------------Data for modeling self-defined block network
water_data = {"name": "water",
            "supplyname": "Pumping station",
            "tranname": "Storage tank",
            "demandname": "Delivery station",
            "nodenum": 49,
            "supplynum": 9,
            "trannum": 6,
            "demandnum": 34,
            "color": "blue",
            "fail_prop_matrix": np.array([[0.08, 0.08, 0.08],
                                          [0, 0.08, 0.08],
                                          [0, 0, 0.08]])
            }

power_data = {"name": "power",
            "supplyname": "Power gate station",
            "tranname": "23kv substation",
            "demandname": "12kv substation",
            "nodenum": 60,
            "supplynum": 9,
            "trannum": 14,
            "demandnum": 37,
            "color": "red",
            "fail_prop_matrix": np.array([[0.08, 0.08, 0.08],
                                          [0, 0.08, 0.08],
                                          [0, 0, 0.08]])
            }

gas_data = {"name": "gas",
            "supplyname": "Gas gate station",
            "tranname": "Regulator station",
            "demandname": "Other",
            "nodenum": 16,
            "supplynum": 3,
            "trannum": 7,
            "demandnum": 6,
            "color": "green",
            "fail_prop_matrix": np.array([[0.08, 0.08, 0.08],
                                          [0, 0.08, 0.08],
                                          [0, 0, 0.08]])
            }

block_data = [water_data, power_data, gas_data]