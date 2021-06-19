import pickle
from collections import Mapping

import igraph as ig
import matplotlib.pyplot as plt
import networkx as nx
import plotly.graph_objs as go
import tensorflow as tf
from plotly.offline import iplot

input_size = (512, 512)
nb_samples = 1005
prediction = [1513, 1696, 1923, 1928, 1949, 1977]

source = "C:/Users/s4625266/PycharmProjects/coral/processed_image/"
path = 'C:/Users/s4625266/PycharmProjects/coral/pickled_data'


def read(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def myprint(hierarchy):
    for k, v in hierarchy.items():
        if not v == {}:
            myprint(v)
        else:
            hierarchy[k] = cluster[k]


cluster, hierarchy = read(f'{path}/clusters.pkl'), read(f'{path}/nested_dict.pkl')
print(cluster)
myprint(hierarchy)

fig = plt.figure(figsize=(10, 10))
G = nx.DiGraph()

# Iterate through the layers
qi = list(hierarchy.items())
while qi:
    v, d = qi.pop()
    for nv, nd in d.items():
        G.add_edge(v, nv)
        if isinstance(nd, Mapping):
            qi.append((nv, nd))
            if len(cluster[v]) - len(cluster[nv]) == 1:
                item = (list(list(set(cluster[v]) - set(cluster[nv])) + list(set(cluster[nv]) - set(cluster[v]))))
                G.add_edge(v, item[0])
        if isinstance(nd, list):
            if len(cluster[v]) - len(cluster[nv]) == 1:
                item = (list(list(set(cluster[v]) - set(cluster[nv])) + list(set(cluster[nv]) - set(cluster[v]))))
                G.add_edge(v, item[0])
            if len(nd) == 1:
                G.add_edge(nv, nd[0])
            else:
                for i in nd:
                    G.add_edge(nv, i)

# train_ds = tf.keras.preprocessing.image_dataset_from_directory(
#     source,
#     seed=42,
#     image_size=input_size,
#     shuffle=False
# )

# for i in cluster[prediction[0]]:
#     image_no = i
#     j = 0
#     if i <= 32:
#         j = 0
#     else:
#         j = i // 32
#         image_no = int(((i / j) - 32) * j)
#     plt.imshow([i[0] for i in train_ds.take(j+1)][0][image_no].numpy().astype("uint8"))
#     plt.axis("off")
#     plt.show()


nodes = G.nodes
G = ig.Graph.TupleList(G.edges, directed=True)
layout = G.layout('kk', dim=3)

color_map, labels = [], []
for node in G.vs():
    labels.append(node["name"])
    if int(node['name']) in prediction:
        color_map.append('blue')
    if int(node['name']) not in prediction:
        if int(node['name']) >= nb_samples:
            color_map.append('red')
        else:
            color_map.append('green')

Xn = [layout[k][0] for k in range(len(nodes))]  # x-coordinates of nodes
Yn = [layout[k][1] for k in range(len(nodes))]  # y-coordinates
Zn = [layout[k][2] for k in range(len(nodes))]  # z-coordinates

Xe, Ye, Ze = [], [], []
for e in G.get_edgelist():
    Xe += [layout[e[0]][0], layout[e[1]][0], None]  # x-coordinates of edge ends
    Ye += [layout[e[0]][1], layout[e[1]][1], None]
    Ze += [layout[e[0]][2], layout[e[1]][2], None]

trace1 = go.Scatter3d(x=Xe,
                      y=Ye,
                      z=Ze,
                      mode='lines',
                      line=dict(color='rgb(125,125,125)', width=1),
                      hoverinfo='none'
                      )

trace2 = go.Scatter3d(x=Xn,
                      y=Yn,
                      z=Zn,
                      mode='markers',
                      name='actors',
                      marker=dict(symbol='circle',
                                  size=3,
                                  color=color_map,
                                  colorscale='Viridis',
                                  line=dict(color='rgb(50,50,50)', width=0.5)
                                  ),
                      text=labels,
                      hoverinfo='text'
                      )

axis = dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )

layout = go.Layout(
    title="HIERARCHY OF CORALS",
    width=1750,
    height=1000,
    showlegend=False,
    scene=dict(
        xaxis=dict(axis),
        yaxis=dict(axis),
        zaxis=dict(axis),
    ),
    margin=dict(
        t=100
    ),
    hovermode='closest',
    annotations=[
        dict(
            showarrow=False,
            text="CORALS TAXONOMY",
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dict(
                size=14
            )
        )
    ], )

data = [trace1, trace2]
fig = go.Figure(data=data, layout=layout)


# iplot(fig, filename='corals')


def find_all_paths2(graph, start, end, vn=[]):
    """
    Finds all paths between nodes start and end in graph.
    If any node on such a path is within vn, the path is not
    returned.
    !! start and end node can't be in the vn list !!

    Params:
    --------

    G : igraph graph

    start: start node index

    end : end node index

    vn : list of via- or stop-nodes indices

    Returns:
    --------

    A list of paths (node index lists) between start and end node
    """

    vn = vn if type(vn) is list else [vn]
    path = []
    paths = []
    queue = [(start, end, path)]
    while queue:
        start, end, path = queue.pop()
        path = path + [start]

        if start in vn:
            pass

        if start == end:
            paths.append(path)

        if start not in vn:
            for node in set(graph.neighbors(start)).difference(path):
                queue.append((node, end, path))
    return paths


for p in find_all_paths2(G, 1977, 1513, []):
    print('path: ', p)
