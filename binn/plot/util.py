import os
import pandas as pd

def load_default_mapping(mapping_name: str) -> pd.DataFrame:
    """
    Load a default DataFrame based on a known string key (e.g. 'reactome').
    You can add more options if you have multiple defaults.

    Example usage:
        df = load_default_mapping('reactome')
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    if mapping_name.lower() == "reactome":
        mapping_path = os.path.join(base_dir, "..", "data", "downloads", "reactome_pathways_names_2025_01_21.txt")
        mapping_df = pd.read_csv(mapping_path, sep="\t", header=None,
                                 names=["pathway_id", "pathway_name", "species"])
        return mapping_df
    elif mapping_name.lower() == "uniprot":
        mapping_path = os.path.join(base_dir, "..", "data", "downloads", "uniprotkb_human_2025_01_21.tsv")
        mapping_df = pd.read_csv(mapping_path, sep="\t", header=None,
                                 names=["input_id", "input_name", "input_longer_name"])
        return mapping_df

    else:
        raise ValueError(f"Unknown default mapping requested: {mapping_name}")


def build_mapping_dict(mapping_df: pd.DataFrame,
                       key_col: str,
                       val_col: str) -> dict:
    """
    Convert a pandas DataFrame into a dictionary {key_col -> val_col}.

    - Drops rows with NaN in key_col or val_col.
    - Drops duplicates on the key_col, keeping the first encountered row.

    Example usage:
        dct = build_mapping_dict(mapping_df, key_col="translation", val_col="name")
    """
    tmp = mapping_df.dropna(subset=[key_col, val_col]).drop_duplicates(subset=[key_col])
    return dict(zip(tmp[key_col], tmp[val_col]))


def rename_node_by_layer(node_tuple, input_entity_dict=None, pathways_dict=None):
    """
    Given a node tuple (node_name, layer) and two optional dictionaries:
      - input_entity_dict: used if layer == 1
      - pathways_dict: used if layer != 1
    Return the renamed node if found, else the original node_name.

    Adjust this logic if you have more complicated layer->dictionary rules.
    """
    node_name, layer_str = node_tuple
    layer_int = int(layer_str)

    # If it's layer == 1, attempt input_entity_dict if provided:
    if layer_int == 0 and input_entity_dict is not None:
        return input_entity_dict.get(node_name, node_name)
    # Otherwise, attempt the pathways_dict if provided:
    elif pathways_dict is not None:
        return pathways_dict.get(node_name, node_name)

    # Fallback if no relevant dictionary or not found in dictionary
    return node_name
