# Biologically Informed Neural Network (BINN)

Generates a sparse network and translates it into a pytorch lightning architecture.

```py
from binn.NN import BINN

model = BINN(
            input_data  = 'data/TestQM.tsv', # the data containing the input column
            input_data_column = 'Protein', # specify input column
            pathways = 'data/pathways.tsv', # datafile containing the pathways
            translation_mapping  = 'data/translation.tsv', # translation between input and pathways (can be None)
            activation ='tanh',
            learning_rate  = 1e-4,
            sparse = True,
            n_layers  = 4,
            scheduler = 'plateau',
            validate  =True,
            n_outputs = 2,
            dropout = 0.2)
model.report_layer_structure(verbose=True)

```

Generates a model:

```
Sequential(
  (Layer_0): Linear(in_features=446, out_features=953, bias=True)
  (BatchNorm_0): BatchNorm1d(953, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (Dropout_0): Dropout(p=0.2, inplace=False)
  (Tanh 0): Tanh()
  (Layer_1): Linear(in_features=953, out_features=455, bias=True)
  (BatchNorm_1): BatchNorm1d(455, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (Dropout_1): Dropout(p=0.2, inplace=False)
  (Tanh 1): Tanh()
  (Layer_2): Linear(in_features=455, out_features=162, bias=True)
  (BatchNorm_2): BatchNorm1d(162, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (Dropout_2): Dropout(p=0.2, inplace=False)
  (Tanh 2): Tanh()
  (Layer_3): Linear(in_features=162, out_features=28, bias=True)
  (BatchNorm_3): BatchNorm1d(28, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (Dropout_3): Dropout(p=0.2, inplace=False)
  (Tanh 3): Tanh()
  (Output layer): Linear(in_features=28, out_features=2, bias=True)
)

Layer 0
Number of nonzero weights: 2509
Number biases: 2509
Total number of elements: 425991
Layer 4
Number of nonzero weights: 955
Number biases: 955
Total number of elements: 434070
Layer 8
Number of nonzero weights: 455
Number biases: 455
Total number of elements: 73872
Layer 12
Number of nonzero weights: 163
Number biases: 163
Total number of elements: 4564
Layer 16
Number of nonzero weights: 56
Number biases: 56
Total number of elements: 58
```

---

#### Contact: Erik Hartman - erik.hartman@hotmail.com
