import networkx as nx
import numpy as np
from simulation import simulationIC, simulationLT
from score import SobolT
import ndlib
import ndlib.models.epidemics as ep
import ndlib.models.ModelConfig as mc
import statistics as s
import heapdict as hd
import random

def eigen(g, config, budget):

    g_eig = g.__class__()
    g_eig.add_nodes_from(g)
    g_eig.add_edges_from(g.edges)
    for a, b in g_eig.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_eig[a][b]['weight'] = weight

    eig = []

    for k in range(budget):

        eigen = nx.eigenvector_centrality_numpy(g_eig)
        selected = sorted(eigen, key=eigen.get, reverse=True)[0]
        eig.append(selected)
        g_eig.remove_node(selected)

    return eig

def degree(g, config, budget):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(budget):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)

    return deg

def pi(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    result = []

    for k in range(budget):

        n = g_greedy.number_of_nodes()

        I = np.ones((n, 1))

        C = np.ones((n, n))
        N = np.ones((n, n))

        A = nx.to_numpy_matrix(g_greedy, nodelist=list(g_greedy.nodes()))

        for i in range(5):
            B = np.power(A, i + 1)
            D = C - B
            N = np.multiply(N, D)

        P = C - N

        pi = np.matmul(P, I)

        value = {}

        for i in range(n):
            value[list(g_greedy.nodes())[i]] = pi[i, 0]

        selected = sorted(value, key=value.get, reverse=True)[0]

        result.append(selected)

        g_greedy.remove_node(selected)

    return result

def sigma(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    result = []

    for k in range(budget):

        n = g_greedy.number_of_nodes()

        I = np.ones((n, 1))

        F = np.ones((n, n))
        N = np.ones((n, n))

        A = nx.to_numpy_matrix(g, nodelist=g_greedy.nodes())

        sigma = I
        for i in range(5):
            B = np.power(A, i + 1)
            C = np.matmul(B, I)
            sigma += C

        value = {}

        for i in range(n):
            value[list(g_greedy.nodes())[i]] = sigma[i, 0]

        selected = sorted(value, key=value.get, reverse=True)[0]

        result.append(selected)

        g_greedy.remove_node(selected)

    return result

def Netshield(g, config, budget):

    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    A = nx.adjacency_matrix(g_greedy)

    lam, u = np.linalg.eigh(A.toarray())
    lam = list(lam)
    lam = lam[-1]

    u = u[:, -1]

    u = np.abs(np.real(u).flatten())
    v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

    nodes = []
    for i in range(budget):
        B = A[:, nodes]
        b = B * u[nodes]

        score = v - 2 * b * u
        score[nodes] = -1

        nodes.append(np.argmax(score))

    return nodes

def Soboldeg(g, config, budget):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(2*budget):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)


    for j in range(budget):
        df = simulationIC(1, g, deg, config)
        ST = SobolT(df, deg)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        deg.remove((rem))

    return deg

def degreeDis(g, config, budget):

    selected = []
    d = {}
    t = {}
    dd = hd.heapdict()

    for node in g.nodes():
        d[node] = sum([g[node][v]['weight'] for v in g[node]])
        dd[node] = -d[node]
        t[node] = 0

    for i in range(budget):
        seed, _ = dd.popitem()
        selected.append(seed)
        for v in g.neighbors(seed):
            if v not in selected:
                t[v] += g[seed][v]['weight']
                discount = d[v] - 2*t[v] - (d[v] - t[v])*t[v]
                dd[v] = -discount

    return selected

def SoboldegreeDis(g, config, budget):

    selected = []
    d = {}
    t = {}
    dd = hd.heapdict()

    for node in g.nodes():
        d[node] = sum([g[node][v]['weight'] for v in g[node]])
        dd[node] = -d[node]
        t[node] = 0

    for i in range(2*budget):
        seed, _ = dd.popitem()
        selected.append(seed)
        for v in g.neighbors(seed):
            if v not in selected:
                t[v] += g[seed][v]['weight']
                discount = d[v] - 2*t[v] - (d[v] - t[v])*t[v]
                dd[v] = -discount

    for j in range(budget):
        df = simulationIC(1, g, selected, config)
        ST = SobolT(df, selected)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        selected.remove((rem))

    return selected

def greedyIC(g, config, budget):

    selected = []
    candidates = []

    for node in g.nodes():
        candidates.append(node)

    for i in range(budget):
        max = 0
        index = -1
        for node in candidates:
            seed = []
            for item in selected:
                seed.append(item)
            seed.append(node)

            # g_temp = g.__class__()
            # g_temp.add_nodes_from(g)
            # g_temp.add_edges_from(g.edges)
            result = []

            for iter in range(100):

                model_temp = ep.IndependentCascadesModel(g) # _temp
                config_temp = mc.Configuration()
                config_temp.add_model_initial_configuration('Infected', seed)

                for a, b in g.edges(): # _temp
                    weight = config.config["edges"]['threshold'][(a, b)]
                    # g_temp[a][b]['weight'] = weight
                    config_temp.add_edge_configuration('threshold', (a, b), weight)

                model_temp.set_initial_status(config_temp)

                iterations = model_temp.iteration_bunch(5)

                total_no = 0

                for j in range(5):
                    a = iterations[j]['node_count'][1]
                    total_no += a

                result.append(total_no)

            if s.mean(result) > max:
                max = s.mean(result)
                index = node

        selected.append(index)
        candidates.remove(index)

    return selected

def greedyLT(g, config, budget):

    selected = []
    candidates = []

    for node in g.nodes():
        candidates.append(node)

    for i in range(budget):
        max = 0
        index = -1
        for node in candidates:
            seed = []
            for item in selected:
                seed.append(item)
            seed.append(node)

            # g_temp = g.__class__()
            # g_temp.add_nodes_from(g)
            # g_temp.add_edges_from(g.edges)
            result = []

            for iter in range(100):

                model_temp = ep.ThresholdModel(g) # _temp
                config_temp = mc.Configuration()
                config_temp.add_model_initial_configuration('Infected', seed)

                for a, b in g.edges(): # _temp
                    weight = config.config["edges"]['threshold'][(a, b)]
                    # g_temp[a][b]['weight'] = weight
                    config_temp.add_edge_configuration('threshold', (a, b), weight)

                for i in g.nodes():
                    threshold = random.randrange(1, 20)
                    threshold = round(threshold / 100, 2)
                    config_temp.add_node_configuration("threshold", i, threshold)

                model_temp.set_initial_status(config_temp)

                iterations = model_temp.iteration_bunch(5)

                total_no = iterations[4]['node_count'][1]
                result.append(total_no)

            if s.mean(result) > max:
                max = s.mean(result)
                index = node

        selected.append(index)
        candidates.remove(index)

    return selected

'''
def SobolIM2(g, config):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(10):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)


    df = simulationIC(1, g, deg, config)
    ST = STS(df, deg)
    rank = []
    for node in sorted(ST, key=ST.get, reverse=True):
        rank.append(node)
    rem = rank.pop()
    nodes = rem.split('.')
    k = len(nodes)
    for i in range(k-1):
        deg.remove(int(nodes[i]))

    return deg

def SobolIM(g, config):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(10):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)


    for j in range(5):
        df = simulationIC(1, g, deg, config)
        ST = SobolT(df, deg)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        deg.remove((rem))

    return deg
'''
def Soboleigen(g, config, budget):
    g_eig = g.__class__()
    g_eig.add_nodes_from(g)
    g_eig.add_edges_from(g.edges)
    for a, b in g_eig.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_eig[a][b]['weight'] = weight

    eig = []

    for k in range(2*budget):

        eigen = nx.eigenvector_centrality_numpy(g_eig)
        selected = sorted(eigen, key=eigen.get, reverse=True)[0]
        eig.append(selected)
        g_eig.remove_node(selected)


    for j in range(budget):
        df = simulationIC(1, g, eig, config)
        ST = SobolT(df, eig)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        eig.remove((rem))

    return eig

def SobolPi(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    result = []

    for k in range(2*budget):

        n = g_greedy.number_of_nodes()

        I = np.ones((n, 1))

        C = np.ones((n, n))
        N = np.ones((n, n))

        A = nx.to_numpy_matrix(g_greedy, nodelist=list(g_greedy.nodes()))

        for i in range(5):
            B = np.power(A, i + 1)
            D = C - B
            N = np.multiply(N, D)

        P = C - N

        pi = np.matmul(P, I)

        value = {}

        for i in range(n):
            value[list(g_greedy.nodes())[i]] = pi[i, 0]

        selected = sorted(value, key=value.get, reverse=True)[0]

        result.append(selected)

        g_greedy.remove_node(selected)


    for j in range(budget):
        df = simulationIC(1, g, result, config)
        ST = SobolT(df, result)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        result.remove((rem))

    return result

def SobolSigma(g, config, budget):
    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    result = []

    for k in range(2*budget):

        n = g_greedy.number_of_nodes()

        I = np.ones((n, 1))

        F = np.ones((n, n))
        N = np.ones((n, n))

        A = nx.to_numpy_matrix(g, nodelist=g_greedy.nodes())

        sigma = I
        for i in range(5):
            B = np.power(A, i + 1)
            C = np.matmul(B, I)
            sigma += C

        value = {}

        for i in range(n):
            value[list(g_greedy.nodes())[i]] = sigma[i, 0]

        selected = sorted(value, key=value.get, reverse=True)[0]

        result.append(selected)

        g_greedy.remove_node(selected)


    for j in range(budget):
        df = simulationIC(1, g, result, config)
        ST = SobolT(df, result)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        result.remove((rem))

    return result

def SobolNS(g, config, budget):

    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    A = nx.adjacency_matrix(g_greedy)

    lam, u = np.linalg.eigh(A.toarray())
    lam = list(lam)
    lam = lam[-1]

    u = u[:, -1]

    u = np.abs(np.real(u).flatten())
    v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

    nodes = []
    for i in range(2*budget):
        B = A[:, nodes]
        b = B * u[nodes]

        score = v - 2 * b * u
        score[nodes] = -1

        nodes.append(np.argmax(score))


    for j in range(budget):
        df = simulationIC(1, g, nodes, config)
        ST = SobolT(df, nodes)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        nodes.remove((rem))

    return nodes

def SoboldegLT(g, config, budget):
    g_deg = g.__class__()
    g_deg.add_nodes_from(g)
    g_deg.add_edges_from(g.edges)
    for a, b in g_deg.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_deg[a][b]['weight'] = weight

    deg = []

    for k in range(2*budget):
        degree = nx.centrality.degree_centrality(g_deg)
        selected = sorted(degree, key=degree.get, reverse=True)[0]
        deg.append(selected)
        g_deg.remove_node(selected)

    for j in range(budget):
        df = simulationLT(10, g, deg, config)
        ST = SobolT(df, deg)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        deg.remove((rem))

    return deg

def SoboleigenLT(g, config,budget):
    g_eig = g.__class__()
    g_eig.add_nodes_from(g)
    g_eig.add_edges_from(g.edges)
    for a, b in g_eig.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_eig[a][b]['weight'] = weight

    eig = []

    for k in range(2*budget):

        eigen = nx.eigenvector_centrality_numpy(g_eig)
        selected = sorted(eigen, key=eigen.get, reverse=True)[0]
        eig.append(selected)
        g_eig.remove_node(selected)


    for j in range(budget):
        df = simulationLT(10, g, eig, config)
        ST = SobolT(df, eig)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        eig.remove((rem))

    return eig

def SobolNSLT(g, config,budget):

    g_greedy = g.__class__()
    g_greedy.add_nodes_from(g)
    g_greedy.add_edges_from(g.edges)

    for a, b in g_greedy.edges():
        weight = config.config["edges"]['threshold'][(a, b)]
        g_greedy[a][b]['weight'] = weight

    A = nx.adjacency_matrix(g_greedy)

    lam, u = np.linalg.eigh(A.toarray())
    lam = list(lam)
    lam = lam[-1]

    u = u[:, -1]

    u = np.abs(np.real(u).flatten())
    v = (2 * lam * np.ones(len(u))) * np.power(u, 2)

    nodes = []
    for i in range(2*budget):
        B = A[:, nodes]
        b = B * u[nodes]

        score = v - 2 * b * u
        score[nodes] = -1

        nodes.append(np.argmax(score))

    for j in range(budget):
        df = simulationLT(1, g, nodes, config)
        ST = SobolT(df, nodes)
        rank = []
        for node in sorted(ST, key=ST.get, reverse=True):
            rank.append(node)
        rem = rank.pop()
        nodes.remove((rem))

    return nodes