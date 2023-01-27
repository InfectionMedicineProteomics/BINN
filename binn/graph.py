
import networkx as nx
from binn.ExplainerPlot import shap_sankey
import numpy as np


def create_graph(df, val_col="value"):
    df['source'] = df['source'].apply(lambda x: x.split('_')[0])
    df['target'] = df['target'].apply(lambda x: x.split('_')[0])
    G = nx.DiGraph()
    for k in df.iterrows():
        source = k[1]['source']
        value = k[1][val_col]
        source_layer = k[1]['source layer']+1
        G.add_node(source, weight=value, layer=source_layer)
    for k in df.iterrows():
        source = k[1]['source']
        target = k[1]['target']
        G.add_edge(source, target)
    root_layer = max(df['target layer'])+1
    G.add_node('root', weight=0, layer=root_layer)
    return G


def get_downstream_subgraph(G, query_node, depth_limit=None):
    SG = nx.DiGraph()
    nodes = [n for n in nx.traversal.bfs_successors(
        G, query_node, depth_limit=depth_limit) if n != query_node]
    for source, targets in nodes:
        SG.add_node(source, **G.nodes()[source])
        for t in targets:
            SG.add_node(t, **G.nodes()[t])
    for node1 in SG.nodes():
        for node2 in SG.nodes():
            if G.has_edge(node1, node2):
                SG.add_edge(node1, node2)

    SG.add_node(query_node, **G.nodes()[query_node])
    return SG


def get_upstream_subgraph(G, query_node, depth_limit=None):
    G = G.reverse()
    SG = get_downstream_subgraph(G, query_node, depth_limit=depth_limit)
    return SG


def get_complete_subgraph(G, query_node, depth_limit=None):
    SG = get_downstream_subgraph(G, query_node)
    G = G.reverse()
    nodes = [n for n in nx.traversal.bfs_successors(
        G, query_node, depth_limit=depth_limit) if n != query_node]
    for source, targets in nodes:
        SG.add_node(source, **G.nodes()[source])
        for t in targets:
            SG.add_node(t, **G.nodes()[t])
    for node1 in SG.nodes():
        for node2 in SG.nodes():
            if G.has_edge(node1, node2):
                SG.add_edge(node1, node2)

    SG.add_node(query_node, **G.nodes()[query_node])
    return SG


def get_nr_nodes_in_upstream_SG(G, query_node):
    nr_nodes = 0
    SG = get_upstream_subgraph(G, query_node, depth_limit=None)
    for node in SG.nodes():
        nr_nodes += 1
    return nr_nodes


def get_nr_nodes_in_downstream_SG(G, query_node):
    nr_nodes = 0
    SG = get_downstream_subgraph(G, query_node, depth_limit=None)
    for node in SG.nodes():
        nr_nodes += 1
    return nr_nodes


def get_fan_in(G, query_node):
    return len([n for n in G.in_edges(query_node)])


def get_fan_out(G, query_node):
    return len([n for n in G.out_edges(query_node)])


def add_normalization(df, G):
    df['fan_in'] = df.apply(lambda x: get_fan_in(G, x['source']), axis=1)
    df['fan_out'] = df.apply(lambda x: get_fan_out(G, x['source']), axis=1)
    df['fan_tot'] = df.apply(lambda x: x['fan_in'] + x['fan_out'], axis=1)
    df['nodes_in_upstream'] = df.apply(
        lambda x: get_nr_nodes_in_upstream_SG(G, x['source']), axis=1)
    df['nodes_in_downstream'] = df.apply(
        lambda x: get_nr_nodes_in_downstream_SG(G, x['source']), axis=1)
    df['nodes_in_SG'] = df['nodes_in_downstream'] + df['nodes_in_upstream']
    df['log(nodes_in_SG)'] = np.log2(df['nodes_in_SG'])
    df['weighted_val_log'] = df.apply(
        lambda x: x["value"]/(np.log2(x['nodes_in_SG'])), axis=1)
    return df


def generate_sankey(df, G, query_node, upstream=False, save_name='sankey.png', val_col='value', cmap_name='coolwarm'):
    if upstream == False:
        final_node = "root"
        SG = get_downstream_subgraph(G, query_node, depth_limit=None)
        source_or_target = "source"
    else:
        final_node = query_node
        SG = get_upstream_subgraph(G, query_node, depth_limit=None)
        source_or_target = "target"
    nodes_in_SG = [n for n in SG.nodes]
    df = df[df[source_or_target].isin(nodes_in_SG)]
    fig = shap_sankey(df, final_node=final_node,
                      val_col=val_col, cmap_name=cmap_name)

    fig.write_image(
        f'{save_name}', width=1200, scale=2.5, height=500)
