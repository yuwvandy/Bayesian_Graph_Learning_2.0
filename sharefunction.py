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