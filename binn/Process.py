import pandas as pd
import networkx as nx


def get_separation(pathways, input_data, translation_mapping):
    sep_path = ','
    sep_input = ','
    sep_transl = ','
    if pathways.endswith('tsv'):
        sep_path = '\t'
    if input_data.endswith('tsv'):
        sep_input = '\t'
    if translation_mapping.endswith('tsv'):
        sep_transl = '\t'
    return sep_path, sep_input, sep_transl

class ProcessData():
    def __init__(self,
                 pathways : str, 
                 input_data : str , 
                 translation_mapping,
                 verbose : bool = False,
                 input_data_column : str = 'Protein'):
        """   
        A class to generate the DataFrames needed to create the network. 
        """
        self.verbose=verbose
        self.input_data_column=input_data_column
        sep_path, sep_input, sep_transl = get_separation(pathways, input_data, translation_mapping)
        self.input_df = pd.read_csv(input_data, sep=sep_input)
        if isinstance(translation_mapping, str):
            self.translation_df = pd.read_csv(translation_mapping, index_col=False, sep=sep_transl)
        elif isinstance(translation_mapping,pd.DataFrame):
            self.translation_df = translation_mapping
        self.path_df = pd.read_csv(pathways, sep=sep_path, index_col=False)
        
    def subset_on_proteins_in_ms_data(self):
        proteins_in_ms_data = self.input_df[self.input_data_column].unique()
        self.translation_df = self.translation_df[self.translation_df['input'].isin(proteins_in_ms_data)]
        if self.verbose:
            print(f'Number of reactome ids before subsetting: {len(self.translation_df.index)}')
            print(f"Unique proteins in reactome df: {len(list(self.translation_df['input'].unique()))}")
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
        original_parent = list(self.translation_df['translation'].unique()) 
        idx_list = []
        idx_list = add_pathways(counter, idx_list, original_parent)
        self.path_df = self.path_df[self.path_df['child'].isin(idx_list)]
        print("Final number of unique connections in pathway: ", len(self.path_df.index))
        return self.path_df

def get_mapping_to_all_layers(path_df,  translation_df):
    G = nx.from_pandas_edgelist(path_df,source='child',target='parent',create_using=nx.DiGraph())
    components = {"input":[],"connections":[]}
    for input in translation_df['input']:
        ids = translation_df[translation_df['input'] == input]['translation']
        for id in ids:
            connections = G.subgraph(nx.single_source_shortest_path(G,id).keys()).nodes
            for connection in connections:
                components["input"].append(input)
                components["connections"].append(connection)
    components = pd.DataFrame(components)
    components.drop_duplicates(inplace=True)
    return components
  
        
def generate_pathway_file(
                        pathways = 'data/pathways.tsv',
                        input_data = 'data/TestQM.tsv' ,
                        input_data_column = "Protein",
                        translation_mapping = None,):
    """_summary_
    Args:
        pathways (str): Path to edge list (*.tsv)
        input_data (str): Path to input data (*.tsv)
        translation_mapping (str, None): Translation mapping between input names and edge list names. This might be necessary
        input_data_column (str): specify which column in input_data contains the data
    Returns:
        (pathways, input, mapping) 
    """
    if translation_mapping == None:
        # Create mapping to itself
        translation_mapping = pd.DataFrame({
            'input':pd.read_csv(input_data, sep='\t')[input_data_column].values,
            'translation':pd.read_csv(input_data, sep='\t')[input_data_column].values,
            })
    RD = ProcessData(pathways, input_data, translation_mapping, input_data_column)
    RD.subset_on_proteins_in_ms_data()
    RD.subset_pathways_on_idx()
    mapping_to_all_layers = get_mapping_to_all_layers(RD.path_df, RD.translation_df)
    proteins = RD.translation_df['input'].unique()
    return RD.path_df, proteins, mapping_to_all_layers
    
    