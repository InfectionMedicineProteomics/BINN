import os
import pandas as pd

def load_reactome_db():
    """
    Load default Reactome mapping and pathways files.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))

    mapping_path = os.path.join(base_dir, "../../data/uniprot_2_reactome_2025_01_14.txt")
    pathways_path = os.path.join(base_dir, "../../data/reactome_pathways_relation_2025_01_14.txt")

    try:
        mapping = pd.read_csv(mapping_path, sep="\t", header=None, names=["input", "translation", "url", "name", "x", "species"])
        pathways = pd.read_csv(pathways_path, sep="\t", header=None, names=["target", "source"])
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Unable to find file: {e.filename}")

    return {"mapping": mapping, "pathways": pathways}
