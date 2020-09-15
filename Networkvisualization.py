# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 23:42:03 2020

@author: 10624
"""

from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

def Plot3d1(Networks):
    fig = plt.figure(figsize = (20, 15))
    ax = fig.add_subplot(111, projection = '3d')
    ZZ = [400, 200, 0]
    
    temp = 0
    for network in Networks:
        x, y = np.arange(0, 70, 1), np.arange(0, 70, 1)
        x, y = np.meshgrid(x, y)
        z = np.array([[ZZ[temp]]*len(x)]*len(y), dtype = float)
        
        temp += 1
        
        X = network.x/1000
        Y = network.y/1000
        if(network.name == 'Gas'):
            Z = np.array([ZZ[0]]*network.nodenum, dtype = float)
        if(network.name == 'Power'):
            Z = np.array([ZZ[1]]*network.nodenum, dtype = float)
        if(network.name == 'Water'):
            Z = np.array([ZZ[2]]*network.nodenum, dtype = float)
        
        network.Z = Z
        network.X = X
        network.Y = Y
        
        #Network nodes plots
        ax.scatter3D(X[network.supplyseries], Y[network.supplyseries], Z[network.supplyseries], \
                     depthshade = False, zdir = 'z', marker = '+', color = network.color, \
                         label = network.supplyname, s = 80)
        ax.scatter3D(X[network.transeries], Y[network.transeries], Z[network.transeries], \
                     depthshade = False, zdir = 'z', marker = '*', color = network.color, \
                         label = network.tranname, s = 60)
        ax.scatter3D(X[network.demandseries], Y[network.demandseries], Z[network.demandseries], \
                     depthshade = False, zdir = 'z', marker = 'o', color = network.color, \
                         label = network.demandname, s = 40)
            
        ax.plot_surface(x, y, z, linewidth=0, antialiased=False, alpha=0.05, color = network.color)
        
        #Link plots
        ##link in the network
        for i in range(len(network.Adjmatrix)):
            for j in range(len(network.Adjmatrix)):
                if(network.Adjmatrix[i, j] == 1):
                    ax.plot([X[i], X[j]], [Y[i], Y[j]], [Z[i], Z[j]], 'black', lw = 1)
    
    # for network in InterNetworks:
    #     for i in range(network.nodenum1):
    #         if(network.__class__.__name__ == 'phynode2node'):
    #             for j in range(network.nodenum2):
    #                 if(network.adjmatrix[i, j] == 1):
    #                         ax.plot([network.network1.X[network.network1.demandseries[i]], network.network2.X[network.network2.supplyseries[j]]], \
    #                                 [network.network1.Y[network.network1.demandseries[i]], network.network2.Y[network.network2.supplyseries[j]]], \
    #                                 [network.network1.Z[0], network.network2.Z[0]], 'purple', linestyle = "--", lw = 1)
            
            # if(network.__class__.__name__ == 'phynode2link'):
            #     for j in range(network.linknum2):
            #             ax.plot([network.network1.X[network.network1.demandseries[i]], network.network2.edgelist[j]["middlex"]/1000], \
            #                     [network.network1.Y[network.network1.demandseries[i]], network.network2.edgelist[j]["middley"]/1000], \
            #                     [network.network1.Z[0], network.network2.Z[0]], 'purple')
                        
                    # if(network.__class__.__name__ == 'phynode2interlink'):
                    #     ax.plot([network.network3.X[network.network3.demandseries[i]], network.network2.edgelist[j]["middlex"]/1000, \
                    #             [network.network3.Y[network.network3.demandseries[i]], network.network2.edgelist[j]["middley"]/1000, \
                    #             [network.network3.Z[0], network.network2.Z[0], 'purple')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.85), ncol=3, shadow=False, frameon = 0)
    
    
import graphviz
def GraphvizGraph(network, color, shape):
    """ Visualize the networks using graphviz package
    Input:
        network - object
        color - the color of facilities in the network
        shape - the shape of different type of nodes(facilities) in the network
    
    Output: the plot of the network in PDF file saved in the default path
    """
    G = graphviz.Digraph()
    
    #Add nodes to the G
    for i in range(network.nodenum):
        if(i in network.supplyseries):
            G.node(str(i + 1), color = color, fillcolor = color, shape = shape[0], style = "filled")
        if(i in network.transeries):
            G.node(str(i + 1), color = color, fillcolor = color, shape = shape[1], style = 'filled')
        if(i in network.demandseries):
            G.node(str(i + 1), color = color, fillcolor = color, shape = shape[2], style = 'filled')
        
    for i in range(network.nodenum):
        for j in range(i, network.nodenum):
            if(network.adjmatrix[i, j] == 1):
                G.edge(str(i + 1), str(j + 1))
    
    G.render("./result/" + network.name, view = True)
    
def topologysort(network):
    """ Perform topological sort for visulization
    Input:
        network - the network on which the topological sort is performed
    """
    v_visited = np.zeros(network.nodenum, dtype = int)
    stack = []
    
    for i in range(network.nodenum):
        if(v_visited[i] == 0):
            stack, v_visited = DFS(network.adjmatrix, i, stack, v_visited)
    
    stack.reverse()
    network.topoorder = stack
    return stack

def DFS(adjmatrix, node, stack, v_visited):
    """ Perform DFS search
    """
    v_visited[node] = 1
    
    for i in range(len(adjmatrix)):
        if(adjmatrix[node, i] == 1 and v_visited[i] == 0):
            stack, v_visited = DFS(adjmatrix, i, stack, v_visited)
    
    stack.append(node)
    
    return stack, v_visited

def nodecoordinate(network):
    """ Assign the coordinates so that the nodes can be visualized organized
    """
    network.nodelocation = np.empty((network.nodenum, 2), dtype = float)
    
    s, t, d = 0, 0, 0
    distance = network.demandnum*40
    sdistance = distance/network.supplynum
    tdistance = distance/network.trannum
    
    for i in range(len(network.topoorder)):
        if(network.topoorder[i] in network.supplyseries):
            network.nodelocation[network.topoorder[i], 0] = s
            network.nodelocation[network.topoorder[i], 1] = 150
            s += sdistance
            
        if(network.topoorder[i] in network.transeries):
            network.nodelocation[network.topoorder[i], 0] = t
            network.nodelocation[network.topoorder[i], 1] = 100
            t += tdistance
            
        if(network.topoorder[i] in network.demandseries):
            network.nodelocation[network.topoorder[i], 0] = d
            network.nodelocation[network.topoorder[i], 1] = 50
            d += 40
        

GraphvizGraph(sc_water, color = "skyblue", shape = ["square", "circle", "triangle"])
GraphvizGraph(sc_power, color = "skyblue", shape = ["square", "circle", "triangle"])
GraphvizGraph(sc_gas, color = "skyblue", shape = ["square", "circle", "triangle"])

watersort = topologysort(sc_water)
powersort = topologysort(sc_power)
gassort = topologysort(sc_gas)

nodecoordinate(sc_water)
for i in range(len(sc_water.nodelocation)):
    plt.scatter(sc_water.nodelocation[i,0], sc_water.nodelocation[i,1])

for i in range(len(sc_water.edgelist)):
    plt.plot([sc_water.nodelocation[sc_water.edgelist[i, 0] - 1, 0], sc_water.nodelocation[sc_water.edgelist[i, 1] - 1, 0]], [sc_water.nodelocation[sc_water.edgelist[i, 0] - 1, 1], sc_water.nodelocation[sc_water.edgelist[i, 1] - 1, 1]])


Water = graphviz.Digraph('Water', filename = 'water.gv')
Water.edge("a","b")
Water.view(quiet = True)
