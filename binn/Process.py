import pandas as pd
import networkx as nx

#TODO: Make it so that one method spits out:
################################ proteins, connection, protein TO ALL connection
class ProcessData():
    """ 
    A class to subset and get reactome data.
    """
    
    def __init__(self,
                 pathways : str, 
                 input_data : str , 
                 translation_mapping : str,
                 verbose=False):
        self.verbose=verbose
        self.input_df = pd.read_csv(input_data, sep='\t')
        self.translation_df = pd.read_csv(translation_mapping, index_col=False,
                        names = ['UniProt_id', 'Reactome_id', 'URL', 'Description','Evidence Code','Species'], sep="\t")
        self.path_df = pd.read_csv(pathways, names=['parent','child'], sep="\t", index_col=False)
        
    def subset_species(self, species = 'Homo sapiens'):
        self.translation_df = self.translation_df[self.translation_df['Species'] == species]
        return self.translation_df
        
    def subset_on_proteins_in_ms_data(self):
        proteins_in_ms_data = self.input_df['Protein'].unique()
       
        self.translation_df = self.translation_df[self.translation_df['UniProt_id'].isin(proteins_in_ms_data)]
        if self.verbose:
            print(f'Number of reactome ids before subsetting: {len(self.translation_df.index)}')
            print(f"Unique proteins in reactome df: {len(list(self.translation_df['UniProt_id'].unique()))}")
        return self.translation_df
        
    def subset_pathways_on_idx(self):
        """
        Recursive method to add parents and children to pathway_df based on filtered translation_df.
        """
        def add_pathways(counter, idx_list, parent):
            counter += 1
            if self.verbose:
                print(f"Function called {counter} times.")
                print(f'Values in idx_list: {len(idx_list)}')
            if len(parent) == 0:
                print('Base case reached')
                return idx_list
            else:
                idx_list = idx_list + parent
                subsetted_pathway = self.path_df[self.path_df['child'].isin(parent)]

                new_parent = list(subsetted_pathway['parent'].unique())
                return add_pathways(counter, idx_list, new_parent)
                
        counter = 0    
        original_parent = list(self.translation_df['Reactome_id'].unique()) 
        idx_list = []
        idx_list = add_pathways(counter, idx_list, original_parent)
        self.path_df = self.path_df[self.path_df['parent'].isin(idx_list)]
        print("Final number of unique connections in pathway: ", len(self.path_df.index))
        return self.path_df

def get_mapping_to_all_layers(path_df, translation_df):
    G = nx.from_pandas_edgelist(path_df,source='child',target='parent',create_using=nx.DiGraph())
    components = {"input":[],"connections":[]}
    for protein in translation_df['UniProt_id']:
        ids = translation_df[translation_df['UniProt_id'] == protein]['Reactome_id']
        for id in ids:
            connections = G.subgraph(nx.single_source_shortest_path(G,id).keys()).nodes
            for connection in connections:
                components["input"].append(protein)
                components["connections"].append(connection)
    components = pd.DataFrame(components)
    components.drop_duplicates(inplace=True)
    return components
  
        
def generate_pathway_file(
                          species = 'Homo sapiens',
                          pathways = 'data/ReactomePathwaysRelation.txt',
                          input_data = 'data/TestQM.tsv' ,
                          translation_mapping = "data/UniProt2Reactome.txt"):
    """_summary_

    Args:
        species (str): e.g. Homo sapiens. If none won't subset.
        pathways (str): _description_. Defaults to 'data/reactome/ReactomePathwaysRelation.txt'.
        input_data (str, optional): _description_. Defaults to 'data/ms/inner.tsv'.
        translation_mapping (str, optional): _description_. Defaults to "data/reactome/UniProt2Reactome.txt".

    Returns:
        pd.DataFrame : DataFrame containing the subsettet paths
    """
    RD = ProcessData(pathways, input_data, translation_mapping)
    if species is not None: 
        RD.subset_species(species)
    RD.subset_on_proteins_in_ms_data()
    RD.subset_pathways_on_idx()
    mapping_to_all_layers = get_mapping_to_all_layers(RD.path_df, RD.translation_df)
    proteins = RD.translation_df['UniProt_id'].unique()
    
    return RD.path_df, proteins, mapping_to_all_layers
    
    