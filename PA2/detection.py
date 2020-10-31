import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


print("Reading data...")
f = open('data/egonet.txt', 'r')
data_list = [line.strip() for line in f]
data = np.array(data_list)
print("Done")


"""Begin question 6"""


edges = set()
nodes = set()
for edge in data:
    x, y = edge.split()
    x, y = int(x), int(y)
    edges.add((x, y))
    edges.add((y, x))
    nodes.add(x)
    nodes.add(y)

print(len(edges))
print(len(nodes))

G = nx.Graph()
for e in edges:
    G.add_edge(e[0], e[1])
nx.draw(G)
plt.show()
plt.clf()

# From the graph we can see there are 3 connected components,
# and can count 40 nodes in the largest connected component


"""Begin question 7"""
