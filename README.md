# SparseNN

```py
from snn.NN import SparseNN

model = SparseNN(proteins = proteins,
            pathways = pathways,
            connections_to_all_layers=components,
            activation ='tanh',
            learning_rate  = 1e-4,
            sparse = True,
            n_layers  = 4,
            scheduler = 'plateau',
            validate  =True,
            n_outputs =2)

```

---

#### Contact: Erik Hartman - erik.hartman@hotmail.com
