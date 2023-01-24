# Biologically Informed Neural Network (BINN)

Generates a sparse network and translates it into a pytorch lightning architecture sequential neural network. Read
more [here](test.ipynb) for code examples.

```py
from binn.NN import BINN

model = BINN(
            input_data  = 'data/TestQM.tsv', # the data containing the input column
            input_data_column = 'Protein', # specify input column
            pathways = 'data/pathways.tsv', # datafile containing the pathways
            translation_mapping  = 'data/translation.tsv', # translation between input and pathways (can be None)
            n_layers  = 4,
            activation ='tanh',
            learning_rate  = 1e-4, # initial learning rate
            scheduler = 'plateau', # can pass own scheduler
            optimizer = 'adam', # can pass own optimizer
            n_outputs = 2, # 2 for binary classification
            dropout = 0.2, # dropout rate. After every hidden layer.
            validate  = True
            )


import torch.nn as nn
# we can also pass a list of activations
activations = [nn.Sigmoid(), nn.Tanh(), nn.ReLU(), nn.ReLU()]
# and a list of dropout ratios
dropouts = [0.5, 0.3, 0.1, 0.1]

```

Generates a model:

```py
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

### Example input

```py

#Test data (quantmatrix or some matrix containing input column - in this case "Protein")
#                        PeptideSequence Protein
# 0  VDRDVAPGTLC(UniMod:4)DVAGWGIVNHAGR  P00746
# 1  VDRDVAPGTLC(UniMod:4)DVAGWGIVNHAGR  P00746
# 2                          VDTVDPPYPR  P04004
# 3                      AVTEQGAELSNEER  P27348
# 4                     VDVIPVNLPGEHGQR  P02751
...
#Pathways file
#           parent          child
# 0    R-BTA-109581   R-BTA-109606
# 1    R-BTA-109581   R-BTA-169911
# 2    R-BTA-109581  R-BTA-5357769
# 3    R-BTA-109581    R-BTA-75153
# 4    R-BTA-109582   R-BTA-140877
...
#Translation file
#           input    translation
# 0    A0A075B6P5   R-HSA-166663
# 1    A0A075B6P5   R-HSA-173623
# 2    A0A075B6P5   R-HSA-198933
# 3    A0A075B6P5   R-HSA-202733
# 4    A0A075B6P5  R-HSA-2029481
...
```

---

#### Contact: Erik Hartman - erik.hartman@hotmail.com
