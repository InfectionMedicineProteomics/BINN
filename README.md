# BINN

```py
from binn.NN import BINN

model = BINN(proteins = proteins,
             pathways = pathways,
             activation = 'tanh',
             learning_rate  = 1e-4,
             sparse = True,
             n_layers  = 4,
             scheduler = 'plateau',
             validate  = True,
             n_outputs = 2)

```

---

#### Contact: Erik Hartman - erik.hartman@hotmail.com
