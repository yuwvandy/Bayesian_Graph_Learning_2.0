def readedge(edgepath):
    """Read the edge file output the 2D array
    Input: edgepath
    Output: the 2D array E where the edge points from E[i, 0] -> E[i, 1]
    """
    import pandas as pd
    import numpy as np

    E = pd.read_excel(edgepath)

    return np.array(E.iloc[:, 1:3])

def edgelist2matrix(edgelist, nodenum):
    """Read the edgelist file and construct the corresponding adjmatrix
    Input: edgelist - 2D array
    Output: adjmatrix
    """
    import numpy as np

    adjmatrix = np.zeros((nodenum, nodenum), dtype = int)

    for i in range(len(edgelist)):
        adjmatrix[edgelist[i, 0] - 1, edgelist[i, 1] - 1] = 1 #-1 because of the starting value from 0

    return adjmatrix

def cal_score(adjlist, target_adj, warm_up_num):
    """Calculate the similarity score of the target and inferenced adjmatrix
    Input:
        adjlist: the list of inferenced adjacent matrix
        target_adj: the targeted adjacent matrix
    Output:
        Accuracy, precision, recall, F1score
    """
    import numpy as np
    
    adjlist = adjlist[warm_up_num:]

    Temp = 0
    
    for i in range(len(target_adj)):
        for j in range(len(target_adj)):
            temp = 0
            for k in range(len(adjlist)):
                temp += adjlist[k][i, j]
            Temp += np.abs(target_adj[i, j] - temp/len(adjlist))
    
    return (1 - Temp/len(target_adj)**2)

def performance(adjmatrix, target_adjmatrix):
    """Calculate the precision, accuracy, recall and F1-score of the adjmatrix given the target_adjmatrix
    Input:
        adjmatrix
        target_adjmatrix
    Output:
        precision, accuracy, recall and F1-score: all are basically scalars
        Notice: the positive or negative is whether there is an edge or not
    """
    TP, TN, FP, FN = 0, 0, 0, 0
    
    for i in range(len(adjmatrix)):
        for j in range(len(adjmatrix)):
            if(adjmatrix[i, j] == 0):
                if(adjmatrix[i, j] == target_adjmatrix[i, j]):
                    TN += 1
                else:
                    FP += 1
            else:
                if(adjmatrix[i, j] == target_adjmatrix[i, j]):
                    TP += 1
                else:
                    FN += 1
    
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    F1 = 2*precision*recall/(precision + recall)
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    
    return accuracy, precision, recall, F1

def cal_adjmatrix_heatmap(adj_list, target_adjmatrix, warm_up):
    """Calculate the heatmap of the adjacent matrix
    Input:
        adj_list: the list of the adjmatrix
        target_adjmatrix: the target adjmatrix
    Output:
        heat_adj: the heatmap of the adjmatrix
    """
    import numpy as np
    
    adj_list = adj_list[warm_up:]
    heat_adj = np.zeros((len(target_adjmatrix), len(target_adjmatrix)), dtype = float)
    
    for i in range(len(target_adjmatrix)):
        for j in range(len(target_adjmatrix)):
            temp = 0
            for k in range(len(adj_list)):
                if(adj_list[k][i, j] == 1):
                    temp += 1
            heat_adj[i, j] = temp/len(adj_list)
    
    return heat_adj

def check_cycle(adjmatrix, root, traverse, stack):
    """ Check the existence of cycle in a graph
    Input:
        adjmatrix: the adjacent matrix of the graph to be checked
        root: the starting node
        traverse
    Output:
        0 - cycle
        1 - no cycle
    """
    stack.append(root)
    traverse[root] = 1
    
    for i in range(len(adjmatrix)):
        if(adjmatrix[root, i] == 1):
            if(traverse[i] == 0):
                if(check_cycle(adjmatrix, i, traverse, stack) == 0):
                    return 0
            elif(i in stack):
                return 0
    
    stack.remove(root)
    return 1

    

                