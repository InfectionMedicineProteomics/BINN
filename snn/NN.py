
import torch.nn as nn
from snn.Network import Network
from pytorch_lightning import LightningModule
import torch
from snn.NNUtils import generate_sequential
from snn.Process import generate_pathway_file


class SparseNN(LightningModule):
    def __init__(self, 
                 input_data : str = 'data/TestQM.tsv', 
                 pathways : str = 'data/pathways.tsv',
                 translation_mapping : str = None,
                 input_data_column :str = 'Protein',
                 activation : str = 'tanh', 
                 weight = torch.Tensor([1,1]),
                 learning_rate : float = 1e-4, 
                 n_layers : int = 4, 
                 scheduler : str = 'plateau',
                 validate : bool =True,
                 n_outputs : int = 2, 
                 dropout : float = 0):
        super().__init__()
        pathways, inputs, mapping_to_all_layers = generate_pathway_file(pathways = pathways,
                          input_data = input_data ,
                          translation_mapping = translation_mapping,
                          input_data_column = input_data_column)
        self.RN = Network(inputs = inputs, pathways=pathways, mapping=mapping_to_all_layers)
        print("Network: ", self.RN.info())
        self.n_layers = n_layers
        connectivity_matrices = self.RN.get_connectivity_matrices(n_layers)
        layer_sizes = []
        self.column_names = []
        for matrix in connectivity_matrices:
            i,_ = matrix.shape
            layer_sizes.append(i)
            self.column_names.append(matrix.index)

        self.layers = generate_sequential(layer_sizes, 
                                        connectivity_matrices = connectivity_matrices, 
                                        activation=activation, 
                                        bias=True, 
                                        n_outputs=n_outputs,
                                        dropout=dropout)
        init_weights(self.layers)   
        self.loss = nn.CrossEntropyLoss(weight=weight) 
        self.learning_rate = learning_rate 
        self.scheduler = scheduler
        self.validate=validate
        self.save_hyperparameters()
    
    def forward(self, x):
        return self.layers(x) 
        
        
    def training_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat,y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
    def test_step(self, batch, batch_nb):
        x, y = batch
        y_hat = self(x)
        loss = self.loss(y_hat, y)
        prediction = torch.argmax(y_hat, dim=1)
        accuracy = self.calculate_accuracy(y, prediction)
        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("test_acc", accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
    def report_layer_structure(self, verbose=False):
        for i, l in enumerate(self.layers):
            if isinstance(l,nn.Linear):
                nz_weights = torch.count_nonzero(l.weight)
                weights = torch.numel(l.weight)
                biases = torch.numel(l.bias)
                if verbose:
                    print(f"Layer {i}")
                    print(f"Number of nonzero weights: {nz_weights} ")
                    print(f"Number biases: {nz_weights} ")
                    print(f"Total number of elements: {weights+biases} ")
                
    def configure_optimizers(self):
        if self.validate==True:
            monitor='val_loss'
        else:
            monitor = 'train_loss'
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        if self.scheduler == 'plateau':
            scheduler = {"scheduler": 
                        torch.optim.lr_scheduler.ReduceLROnPlateau(
                            optimizer, patience=5, 
                            threshold = 0.00001, 
                            mode='min', verbose=True),
                        "interval": "epoch",
                        "monitor": monitor}
        elif self.scheduler == 'step':
            scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1, verbose=True)}
        return [optimizer], [scheduler]
    
    def calculate_accuracy(self, y, prediction):
        return torch.sum(y == prediction).item() / (float(len(y)))
    
    def get_connectivity_matrices(self):
        return self.RN.get_connectivity_matrices(self.n_layers)
        
def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        
def reset_params(m):
    if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.Linear):
        m.reset_parameters()