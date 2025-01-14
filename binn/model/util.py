import pandas as pd


def load_reactome_db():
    mapping = pd.read_csv("../../data/uniprot_2_reactome_2025_01_14.txt", sep="\t")
    pathways = pd.read_csv(
        "../../data/reactome_pathways_relation_2025_01_14.txt", sep="\t"
    )
    return {"mapping": mapping, "pathways": pathways}
