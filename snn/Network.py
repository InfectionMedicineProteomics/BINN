
import networkx as nx
import pandas as pd
import itertools
import numpy as np
import re

""" 
Original file from https://github.com/marakeby/pnet_prostate_paper/blob/master/data/pathways/reactome.py
Modified to fit our repo.

"""


def get_map_from_layer(layer_dict):
    pathways = layer_dict.keys()
    inputs = list(itertools.chain.from_iterable(layer_dict.values()))
    inputs = list(np.unique(inputs))
    df = pd.DataFrame(index=pathways, columns=inputs)
    for k, v in layer_dict.items():
        df.loc[k, v] = 1
    df = df.fillna(0)
    return df.T


def add_edges(G, node, n_levels):
    edges = []
    source = node
    for l in range(n_levels):
        target = node + '_copy' + str(l + 1)
        edge = (source, target)
        source = target
        edges.append(edge)

    G.add_edges_from(edges) #this adds n_levels of "copies" to node.
    return G


def complete_network(G, n_levels=4):
    nr_copies = 0
    sub_graph = nx.ego_graph(G, 'root', radius=n_levels)
    terminal_nodes = [n for n, d in sub_graph.out_degree() if d == 0]
    for node in terminal_nodes:
        distance = len(nx.shortest_path(sub_graph, source='root', target=node))
        if distance <= n_levels:
            nr_copies = nr_copies +  n_levels - distance
            diff = n_levels - distance + 1
            sub_graph = add_edges(sub_graph, node, diff) # This just adds nodes if n_levels is longer than distance. It creates node_last_copy1, node_last_copy2 etc.
            
    # Basically, this methods adds "copy layers" if a path is shorter than n_levels. 
    print(f"Number of copies made for {n_levels} layers: {nr_copies}")
    return sub_graph


def get_nodes_at_level(net, distance):
    # This methods just gets node at a certain distance from root.
    # get all nodes within distance around the query node
    nodes = set(nx.ego_graph(net, 'root', radius=distance))
    # remove nodes that are not **at** the specified distance but closer
    if distance >= 1.:
        nodes -= set(nx.ego_graph(net, 'root', radius=distance - 1))

    return list(nodes)


def get_layers_from_net(net, n_levels):
    layers = []
    for i in range(n_levels):
        nodes = get_nodes_at_level(net, i) # returns all the nodes at a specific Level
        dict = {}
        for n in nodes:
            n_name = re.sub('_copy.*', '', n) #removes the "_copy" in the node name (see complete_network())
            next = net.successors(n) #gets all the succesor of nodes at level
            dict[n_name] = [re.sub('_copy.*', '', nex) for nex in next] #for each node, it adds a list of succesors
            #dict[n_name] = [nex for nex in next]
        layers.append(dict) #this will then return a list where all layers have a dict of origin node and successors.
    return layers


class Network():

    def __init__(self,  
                 inputs = [], 
                 pathways = pd.DataFrame(),  
                 mapping = pd.DataFrame()):
        self.mapping = mapping
        self.ms_hierarchy = pathways
        self.inputs = inputs
        self.netx = self.get_reactome_networkx()
        
    
    def get_terminals(self):
        return [n for n, d in self.netx.out_degree() if d == 0]

    def get_roots(self):
        return get_nodes_at_level(self.netx, distance=1)

    # get a DiGraph representation of the Reactome hierarchy
    def get_reactome_networkx(self):
        if hasattr(self, 'netx'):
            return self.netx
        
        human_hierarchy = self.ms_hierarchy
        net = nx.from_pandas_edgelist(human_hierarchy, 'parent','child', create_using=nx.DiGraph())
        # add root node
        roots = [n for n, d in net.in_degree() if d == 0]
        root_node = 'root'
        edges = [(root_node, n) for n in roots]
        net.add_edges_from(edges)

        return net

    def info(self):
        return nx.info(self.netx)

    def get_tree(self):
        # convert to tree
        return nx.bfs_tree(self.netx, 'root') #breadth first to remove weird connections

    def get_completed_network(self, n_levels):
        return complete_network(self.netx, n_levels = n_levels)

    def get_completed_tree(self, n_levels):
        G = self.get_tree()
        return complete_network(G, n_levels = n_levels)
    
    def get_layers(self, n_levels, direction='root_to_leaf'):
        if direction == 'root_to_leaf':
            net = self.get_completed_network(n_levels)
            layers = get_layers_from_net(net, n_levels)


        terminal_nodes = [n for n, d in net.out_degree() if d == 0]  # set of terminal pathways
        # we need to find inputs belonging to these pathways
        mapping_df = self.mapping
        dict = {}
        missing_pathways = []
        for p in terminal_nodes:
            pathway_name = re.sub('_copy.*', '', p)
            inputs = mapping_df[mapping_df['connections'] == pathway_name]['input'].unique()
            if len(inputs) == 0:
                missing_pathways.append(pathway_name)
            dict[pathway_name] = inputs
        layers.append(dict)
        return layers
    
    def get_connectivity_matrices(self, n_levels, direction = "root_to_leaf"):
        connectivity_matrices = []
        layers = self.get_layers(n_levels, direction)
        for i, layer in enumerate(layers[::-1]):
            mapp = get_map_from_layer(layer)
            if i == 0:
                inputs = list(mapp.index)
                self.inputs = inputs
            filter_df = pd.DataFrame(index=inputs)
            all = filter_df.merge(mapp, right_index=True, left_index=True, how='inner')
            inputs = list(mapp.columns)
            connectivity_matrices.append(all)
        return connectivity_matrices
