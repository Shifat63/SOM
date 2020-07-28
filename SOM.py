import numpy as np
import networkx as nx
import numpy.random as rnd
import csv
import matplotlib.pyplot as plt

# "name" - graph name
# read graph from a file
def readGraph(name):
    return nx.read_gpickle(name + '.gpickle')

# "G" - graph
# "name" - graph name
# write graph to a file
def writeGraph(G, name):
    nx.write_gpickle(G, name + '.gpickle')

# "matX" - input space points matrix
# "G" - graph
# "name" - figure name
# save and show plotted figure
def plotFigure(matX, G, name):
    fig = plt.figure(figsize=(40,20))
    ax = fig.add_subplot(111, projection='3d')
    # plotting input space points
    ax.scatter3D(np.array(matX[0]).astype(np.float), np.array(matX[1]).astype(np.float),np.array(matX[2]).astype(np.float), color='black', alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    matW = np.vstack([G.nodes[v]['w'] for v in G.nodes()]).T
    # plotting graph nodes
    ax.scatter3D(np.array(matW[0]).astype(np.float), np.array(matW[1]).astype(np.float),np.array(matW[2]).astype(np.float), color='red')

    # plotting graph edges
    for v in G:
        point1 = np.array(G.nodes[v]['w']).astype(np.float)
        for adj in G.neighbors(v):
            point2 = np.array(G.nodes[adj]['w']).astype(np.float)
            matW = []
            for i in range(len(point1)):
                matW.append([point1[i], point2[i]])
            ax.plot3D(np.array(matW[0]).astype(np.float), np.array(matW[1]).astype(np.float), np.array(matW[2]).astype(np.float), color='blue')

    plt.title(str(name))
    plt.savefig('animation/' + str(name) + ".png")
    plt.show()
    plt.close(fig)

# "matX" - input space points matrix
# "kx" - number of grid nodes in x axis
# "ky" - number of grid nodes in y axis
# returns a kx*ky grid graph
def initGridSOM(matX, kx, ky):
    G = nx.generators.lattice.grid_2d_graph(kx, ky)
    G = nx.convert_node_labels_to_integers(G)
    m, n = matX.shape # m = number of rows, n = number of columns
    smpl = rnd.choice(n, kx * ky, replace=False) # Generate a 1D array of size kx*ky with random numbers bounded by n
    for v in G:
        G.nodes[v]['w'] = None

    for v in G:
        if G.nodes[v]['w'] is None:
            G.nodes[v]['w'] = np.array(matX[:, smpl[0]]).astype(np.float)
            smpl = np.delete(smpl, 0)

        for adj in G.neighbors(v):
            if G.nodes[adj]['w'] is None:
                point1 = np.array(G.nodes[v]['w']).astype(np.float)
                minDistance = float("inf")
                minPointLocation = -1
                closestPoint = np.array((0,0,0)).astype(np.float)
                for i in range(len(smpl)):
                    point2 = np.array(matX[:, smpl[i]]).astype(np.float)
                    distance = (np.linalg.norm(point1 - point2))**2
                    if distance < minDistance:
                        minDistance = distance
                        closestPoint = point2
                        minPointLocation = i

                if minPointLocation != -1:
                    G.nodes[adj]['w'] = closestPoint
                    smpl = np.delete(smpl, minPointLocation)
    return G

# "matX" - input space points matrix
# "k" - number of nodes
# returns a ring graph having k nodes
def initRingSOM(matX, k):
    G = nx.generators.lattice.grid_2d_graph(k, 1, periodic=True)
    G = nx.convert_node_labels_to_integers(G)
    m, n = matX.shape
    smpl = rnd.choice(n, k, replace=False)
    for v in G:
        G.nodes[v]['w'] = None

    for v in G:
        if G.nodes[v]['w'] is None:
            G.nodes[v]['w'] = np.array(matX[:, smpl[0]]).astype(np.float)
            smpl = np.delete(smpl, 0)

        for adj in G.neighbors(v):
            if G.nodes[adj]['w'] is None:
                point1 = np.array(G.nodes[v]['w']).astype(np.float)
                minDistance = float("inf")
                minPointLocation = -1
                closestPoint = np.array((0, 0, 0)).astype(np.float)
                for i in range(len(smpl)):
                    point2 = np.array(matX[:, smpl[i]]).astype(np.float)
                    distance = (np.linalg.norm(point1 - point2)) ** 2
                    if distance < minDistance:
                        minDistance = distance
                        closestPoint = point2
                        minPointLocation = i

                if minPointLocation != -1:
                    G.nodes[adj]['w'] = closestPoint
                    smpl = np.delete(smpl, minPointLocation)
    return G

# "matX" - input space points matrix
# "k" - number of nodes(must be even)
# returns a graph having k nodes. Graph has 2 rings. Each node of a ring has a connecting edge with exactly one node of another ring.
def initInnerOuterRingSOM(matX, k):
    # First Ring
    G1 = nx.generators.lattice.cycle_graph(int(k/2))
    # Second Ring
    G2 = nx.generators.lattice.cycle_graph(int(k / 2))
    G = nx.disjoint_union(G1,G2)
    G = nx.convert_node_labels_to_integers(G)
    for i in range(int(k / 2)):
        G.add_edge(i, (i+int(k/2)))
    m, n = matX.shape
    smpl = rnd.choice(n, k, replace=False)
    for v in G:
        G.nodes[v]['w'] = None

    for v in G:
        if G.nodes[v]['w'] is None:
            G.nodes[v]['w'] = np.array(matX[:, smpl[0]]).astype(np.float)
            smpl = np.delete(smpl, 0)

        for adj in G.neighbors(v):
            if G.nodes[adj]['w'] is None:
                point1 = np.array(G.nodes[v]['w']).astype(np.float)
                minDistance = float("inf")
                minPointLocation = -1
                closestPoint = np.array((0, 0, 0)).astype(np.float)
                for i in range(len(smpl)):
                    point2 = np.array(matX[:, smpl[i]]).astype(np.float)
                    distance = (np.linalg.norm(point1 - point2)) ** 2
                    if distance < minDistance:
                        minDistance = distance
                        closestPoint = point2
                        minPointLocation = i

                if minPointLocation != -1:
                    G.nodes[adj]['w'] = closestPoint
                    smpl = np.delete(smpl, minPointLocation)
    return G

# "matX" - input space points matrix
# "k" - number of nodes
# returns a graph having k nodes. Graph has 2 rings. There is exactly one connecting edge between these 2 rings
def initDualRingSOM(matX, k):
    # First Ring
    G1 = nx.generators.lattice.cycle_graph(int(k/2))
    # Second Ring
    G2 = nx.generators.lattice.cycle_graph(k - int(k / 2))
    G = nx.disjoint_union(G1,G2)
    G = nx.convert_node_labels_to_integers(G)
    connectingNodeFromG1 = int(k/2)-1
    connectingNodeFromG2 = int(k / 2)
    G.add_edge(connectingNodeFromG1, connectingNodeFromG2)
    m, n = matX.shape
    smpl = rnd.choice(n, k, replace=False)
    for v in G:
        G.nodes[v]['w'] = None

    for v in G:
        if G.nodes[v]['w'] is None:
            G.nodes[v]['w'] = np.array(matX[:, smpl[0]]).astype(np.float)
            smpl = np.delete(smpl, 0)

        for adj in G.neighbors(v):
            if G.nodes[adj]['w'] is None:
                point1 = np.array(G.nodes[v]['w']).astype(np.float)
                minDistance = float("inf")
                minPointLocation = -1
                closestPoint = np.array((0, 0, 0)).astype(np.float)
                for i in range(len(smpl)):
                    point2 = np.array(matX[:, smpl[i]]).astype(np.float)
                    distance = (np.linalg.norm(point1 - point2)) ** 2
                    if distance < minDistance:
                        minDistance = distance
                        closestPoint = point2
                        minPointLocation = i

                if minPointLocation != -1:
                    G.nodes[adj]['w'] = closestPoint
                    smpl = np.delete(smpl, minPointLocation)
    return G

# "G" - graph
# xt - coordinate of a point
# returns nearest node of xt in G
def getBestMatchingUnit(G, xt):
    minDistance = float("inf")
    b = -1
    for v in G:
        vecG = np.array(G.nodes[v]['w']).astype(np.float)
        distance = (np.linalg.norm(xt - vecG)) ** 2
        if distance < minDistance:
            minDistance = distance
            b = v
    return b

def calculate_h(distanceMatrix, b, i, sigma):
    return np.exp(-0.5 * (distanceMatrix[b,i] / sigma**2))

# "matX" - input space points matrix
# "G" - graph
# "tmax" - number of iterations
# sigma - <=1
# eta0 - <=1
# returns trained graph
def trainSOMOnline(matX, G, tmax=1000, sigma0=1.0, eta0=1.0):
    m, n = matX.shape

    # compute matrix of squared path length distances between neurons
    # NOTE: networkx returns a numpy matrix, but we want a numpy array
    # because this allows for easy squaring of its entries
    matD = np.asarray(nx.floyd_warshall_numpy(G))**2

    # a list of tmax random indices into the columns of matrix X
    smpl = rnd.randint(0, n, size=tmax) # 1D Array of size=tmax. Min integer = 0, max integer = n

    for t in range(tmax):
        # sample a point x, i.e. a column of matrix X
        vecX = np.array(matX[:, smpl[t]]).astype(np.float)

        # determine the best matching unit
        b = getBestMatchingUnit(G, vecX)

        # update the learning rate
        eta = eta0 * (1.0 - t/tmax)

        # update the topological adaption rate
        sigma = sigma0 * np.exp(-t/tmax)

        # update all weights
        for i, v in enumerate(G):
            # evaluate neighborhood function
            h = calculate_h(matD, b, i, sigma)
            vecG = np.array(G.nodes[v]['w']).astype(np.float)
            G.nodes[v]['w'] += eta * h * (vecX - vecG)
        if t==(tmax-1):
            plotFigure(matX, G, "Online "+str(tmax))

    return G

# "matX" - input space points matrix
# "G" - graph
# "tmax" - number of iterations
# sigma - <=1
# returns trained graph
def trainSOMOBatch(matX, G, tmax=1000, sigma0=1.0):
    m, n = matX.shape

    # Distance among nodes calculation
    matD = np.asarray(nx.floyd_warshall_numpy(G))**2

    for t in range(tmax):
        # For each point of input space get the best matching
        b = [0 for x in range(n)]
        for j in range(n):
            xj = np.array(matX[:, j]).astype(np.float)
            b[j] = getBestMatchingUnit(G, xj)

        # update the topological adaption rate
        sigma = sigma0 * np.exp(-t/tmax)

        # update all weights
        for i in G:
            summation_xj_multipliedby_h = np.array((0, 0, 0)).astype(np.float)
            summation_h = 0
            for j in range(n):
                xj = np.array(matX[:, j]).astype(np.float)
                summation_xj_multipliedby_h = summation_xj_multipliedby_h + (calculate_h(matD, b[j], i, sigma) * xj)
                summation_h = summation_h + calculate_h(matD, b[j], i, sigma)
            G.nodes[i]['w'] = summation_xj_multipliedby_h / summation_h

        if t == (tmax - 1):
            plotFigure(matX, G, "Batch " + str(tmax))

    return G

# read data from csv
with open('q3dm1-path2.csv', 'r') as f:
    reader = csv.reader(f)
    data_as_list = list(reader)

matX = np.array(data_as_list)
matX = matX.transpose()

# Plotting Grid Online
SOMGrid = initGridSOM(matX, 7, 7)
SOMGrid = trainSOMOnline(matX, SOMGrid, tmax=8000)
writeGraph(SOMGrid, 'SOMGridOnline')
print("Online Grid Done")

# Plotting Ring Online
SOMRing = initRingSOM(matX, 30)
SOMRing = trainSOMOnline(matX, SOMRing, tmax=8000)
writeGraph(SOMRing, 'SOMRingOnline')
print("Online Ring Done")

# Plotting Dual ring Online
SOMDualRing = initDualRingSOM(matX, 24)
SOMDualRing = trainSOMOnline(matX, SOMDualRing, tmax=8000)
writeGraph(SOMDualRing, 'SOMDualRingOnline')
print("Online Dual Ring Done")

# Plotting Inner outer Ring Online
SOMInnerOuterRing = initInnerOuterRingSOM(matX, 24)
SOMInnerOuterRing = trainSOMOnline(matX, SOMInnerOuterRing, tmax=8000)
writeGraph(SOMInnerOuterRing, 'SOMInnerOuterRingOnline')
print("Online Inner Outer Ring Done")

# Plotting Grid Batch
SOMGrid = initGridSOM(matX, 7, 7)
SOMGrid = trainSOMOBatch(matX, SOMGrid, tmax=1000)
writeGraph(SOMGrid, 'SOMGridBatch')
print("Batch Grid Done")

# Plotting Ring Batch
SOMRing = initRingSOM(matX, 30)
SOMRing = trainSOMOBatch(matX, SOMRing, tmax=1000)
writeGraph(SOMRing, 'SOMRingBatch')
print("Batch Ring Done")

# Plotting Dual ring Batch
SOMDualRing = initDualRingSOM(matX, 24)
SOMDualRing = trainSOMOBatch(matX, SOMDualRing, tmax=1000)
writeGraph(SOMDualRing, 'SOMDualRingBatch')
print("Batch Dual Ring Done")

# Plotting Inner outer Ring Batch
SOMInnerOuterRing = initInnerOuterRingSOM(matX, 24)
SOMInnerOuterRing = trainSOMOBatch(matX, SOMInnerOuterRing, tmax=1000)
writeGraph(SOMInnerOuterRing, 'SOMInnerOuterRingBatch')
print("Batch Inner Outer Ring Done")
