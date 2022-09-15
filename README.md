# BINN

Generating neural network from graph files and MS data

```py
from dpks.quant_matrix import QuantMatrix
from binn.NN import BINN

qm = QuantMatrix(...)

model = BINN(
    qm,  #dpks.QuantMatrix (alternatively QuantMatrix.csv)
    PathwaysFile, #(All connections in pathway database)
    TranslationFile,  #Translate the input QuantMatrix proteins to PathwayFile. In case of Reactome: UniProt ID --> Reactome ID.
    )
```
