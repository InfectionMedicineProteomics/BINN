import pandas as pd

#TODO: Make it so that one method spits out:
################################ proteins, connection, protein TO ALL connection
class ReactomeData():
    """ 
    A class to subset and get reactome data.
    """
    
    def __init__(self,
                 hierarchy_file_path : str, 
                 ms_data_file_path : str , 
                 reactome_all_data_file_path : str ):
          
        self.ms_df = pd.read_csv(ms_data_file_path, sep='\t')
        self.reactome_df = pd.read_csv(reactome_all_data_file_path, index_col=False,
                        names = ['UniProt_id', 'Reactome_id', 'URL', 'Description','Evidence Code','Species'], sep="\t")
        self.path_df = pd.read_csv(hierarchy_file_path, names=['parent','child'], sep="\t", index_col=False)
        
    def subset_species_reactome_data(self, species = 'Homo sapiens'):
        df_species = self.reactome_df[self.reactome_df['Species'] == species]
        print(f"Number of rows of {species}: {len(df_species.index)}")
        self.reactome_df = df_species
        return self.reactome_df
        
    def subset_on_proteins_in_ms_data(self):
        proteins_in_ms_data = self.ms_df['Protein'].unique()
        print(f'Number of reactome ids before subsetting: {len(self.reactome_df.index)}')
        self.reactome_df = self.reactome_df[self.reactome_df['UniProt_id'].isin(proteins_in_ms_data)]
        print(f"Number of reactome ids after subsetting: {len(self.reactome_df.index)}")
        print(f"Unique proteins in reactome df: {len(list(self.reactome_df['UniProt_id'].unique()))}")
        return self.reactome_df
        
    def subset_pathways_on_reactome_idx(self):
        """
        Recursive method to add parents and children to pathway_df based on filtered reactome_df.
        """
        def add_pathways(counter, reactome_idx_list, parent):
            counter += 1
            print(f"Function called {counter} times.")
            print(f'Values in reactome_idx_list: {len(reactome_idx_list)}')
            if len(parent) == 0:
                print('Base case reached')
                return reactome_idx_list
            else:
                reactome_idx_list = reactome_idx_list + parent
                subsetted_pathway = self.path_df[self.path_df['child'].isin(parent)]
                new_parent = list(subsetted_pathway['parent'].unique())
                print(f"Values in new_parent: {len(new_parent)}")
                return add_pathways(counter, reactome_idx_list, new_parent)
                
        counter = 0    
        original_parent = list(self.reactome_df['Reactome_id'].unique()) 
        print(f"Length of original parent: {len(original_parent)}")
        reactome_idx_list = []
        reactome_idx_list = add_pathways(counter, reactome_idx_list, original_parent)
        print(f"Final number of values in reactome list: {len(reactome_idx_list)}")
        print(f"Number of unique values in list: {len(list(dict.fromkeys(reactome_idx_list)))}")
        # the reactome_idx_list now contains all reactome idx which we want to subset the path_df on.
        self.path_df = self.path_df[self.path_df['parent'].isin(reactome_idx_list)]
        print("Final number of unique connections in pathway: ", len(self.path_df.index))
        return self.path_df
        
 
    def save_df(self, df_id, save_path):
        if df_id == 'reactome':
            self.reactome_df.to_csv(save_path, index=False)
        if df_id == 'path':
            self.path_df.to_csv(save_path, index=False)
            
  
        
def generate_pathway_file(
                          species = 'Homo sapiens',
                          hierarchy_file_path = 'data/ReactomePathwaysRelation.txt',
                          ms_data_file_path = 'data/TestQM.tsv' ,
                          reactome_all_data_file_path = "data/UniProt2Reactome.txt"):
    """_summary_

    Args:
        species (str): e.g. Homo sapiens. If none won't subset.
        hierarchy_file_path (str): _description_. Defaults to 'data/reactome/ReactomePathwaysRelation.txt'.
        ms_data_file_path (str, optional): _description_. Defaults to 'data/ms/inner.tsv'.
        reactome_all_data_file_path (str, optional): _description_. Defaults to "data/reactome/UniProt2Reactome.txt".

    Returns:
        pd.DataFrame : DataFrame containing the subsettet paths
    """
    RD = ReactomeData(hierarchy_file_path, ms_data_file_path, reactome_all_data_file_path)
    if species is not None: 
        RD.subset_species_reactome_data(species)
    RD.subset_on_proteins_in_ms_data()
    RD.subset_pathways_on_reactome_idx()
    proteins = RD.reactome_df['UniProt_id'].unique()
    return RD.path_df, proteins
    
    