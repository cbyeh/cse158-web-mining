import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import sys

print("Reading data...")
f = open('data/egonet.txt', 'r')
data_list = [line.strip() for line in f]
data = np.array(data_list)
print("Done")


"""Begin question 6"""


edges = set()
nodes = set()
G = nx.Graph
for edge in data:
    x, y = edge.split()
    x, y = int(x), int(y)
    edges.add((x, y))
    edges.add((y, x))
    nodes.add(x)
    nodes.add(y)

G = nx.Graph()
for e in edges:
    G.add_edge(e[0], e[1])
# nx.draw(G)
# plt.show()
# plt.clf()

ccs_num_nodes = [len(c) for c in sorted(
    nx.connected_components(G), key=len, reverse=True)]
print(len(ccs_num_nodes))  # 3 connected components

largest_cc = max(nx.connected_components(G), key=len)
print(len(largest_cc))  # 40 nodes in largest cc


"""Begin question 7"""


def normalized_cut_cost(c1, c2, G):
    """Get normalized cut cost for C=2
    """
    cut = 0
    for e in edges:
        if e[0] in c1 and e[1] in c2:
            cut += 1
    dc1 = sum([G.degree(node) for node in c1])
    dc2 = sum([G.degree(node) for node in c2])
    return (1/2) * (cut / dc1 + cut / dc2)


# Split largest CC in half by smallest and greatest
sorted_largest_cc = sorted(largest_cc)
c1 = list(sorted_largest_cc[:len(sorted_largest_cc) // 2])
c2 = list(sorted_largest_cc[len(sorted_largest_cc) // 2:])

# Find and print normalized cut cost
cost = normalized_cut_cost(c1, c2, G)
print(cost)  # 0.42240587695133147


"""Begin question 8"""


def greedy_normalized_cut_cost(c1, c2, G):

    def find_cut_cost(c1, c2, node):
        c1_temp = [node for node in c1]
        c2_temp = [node for node in c2]
        cut = 0
        if node in c1:
            c1_temp.remove(node)
            c2_temp.append(node)
        else:
            c2_temp.remove(node)
            c1_temp.append(node)
        for e in edges:
            if e[0] in c1_temp and e[1] in c2_temp:
                cut += 1
        dc1 = sum([G.degree(node) for node in c1_temp])
        dc2 = sum([G.degree(node) for node in c2_temp])
        return (1/2) * (cut / dc1 + cut / dc2)

    cost = normalized_cut_cost(c1, c2, G)
    while True:
        min_cost_and_node = sys.maxsize, None
        for node in sorted_largest_cc:
            cost_and_node = find_cut_cost(c1, c2, node), node
            if cost_and_node[0] < min_cost_and_node[0]:
                min_cost_and_node = cost_and_node
        if min_cost_and_node[0] <= cost:
            cost = min_cost_and_node[0]
            if min_cost_and_node[1] in c1:
                c1.remove(min_cost_and_node[1])
                c2.append(min_cost_and_node[1])
            else:
                c2.remove(min_cost_and_node[1])
                c1.append(min_cost_and_node[1])
        else:
            return c1, c2, cost


sol = greedy_normalized_cut_cost(c1, c2, G)
print(sol[0])  # c1
# [697, 703, 708, 713, 719, 745, 747, 753, 769, 772, 774, 798, 800, 803, 805, 810, 811, 819, 828, 823, 830, 840, 880, 890, 869, 856]
print(sol[1])  # c2
# [825, 861, 863, 864, 876, 878, 882, 884, 886, 888, 889, 893, 729, 804]
print(sol[2])  # cost
# 0.09817045961624274
