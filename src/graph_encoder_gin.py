"""
Graph Encoder for Quantum Multi-Target Drug Discovery
Converts SMILES strings into molecular graphs and implements Graph Convolutional Networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, GraphNorm, global_mean_pool, global_max_pool
from torch_geometric.data import Data, Batch
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen
import warnings
warnings.filterwarnings('ignore')

import datetime
DEBUG_LOG_PATH = "amplitude_debug.log"

def log_debug(msg: str):
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {msg}\n")



class MolecularGraph:
    """
    Class for the convertion of SMILES into molecular graphs.
    """
    
    def __init__(self):
        # Atom features we are going to use
        self.atom_features = [
            'atomic_num', 'degree', 'formal_charge', 'chiral_tag',
            'num_hs', 'hybridization', 'aromatic', 'mass'
        ]
        
        # Bond features
        self.bond_features = [
            'bond_type', 'stereo', 'is_conjugated', 'is_aromatic'
        ]
    
    def get_atom_features(self, atom):
        """
        Extracts features from one molecule
        """
        features = []
        
        # Atomic number (one-hot encoded for main features)
        atomic_num = atom.GetAtomicNum()
        common_atoms = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]  # H, C, N, O, F, P, S, Cl, Br, I
        atom_encoding = [int(atomic_num == x) for x in common_atoms]
        features.extend(atom_encoding)
        
        # Degree (connections)
        degree = atom.GetDegree()
        degree_encoding = [int(degree == x) for x in range(6)]  # 0-5 connections
        features.extend(degree_encoding)
        
        # Formal charge
        formal_charge = atom.GetFormalCharge()
        charge_encoding = [int(formal_charge == x) for x in [-2, -1, 0, 1, 2]]
        features.extend(charge_encoding)
        
        # Hybridization
        hybridization = atom.GetHybridization()
        hybrid_types = [Chem.HybridizationType.SP, Chem.HybridizationType.SP2, 
                       Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D,
                       Chem.HybridizationType.SP3D2]
        hybrid_encoding = [int(hybridization == x) for x in hybrid_types]
        features.extend(hybrid_encoding)
        
        # Aromaticity
        features.append(int(atom.GetIsAromatic()))
        
        # Number of hydrogens
        num_hs = atom.GetTotalNumHs()
        hs_encoding = [int(num_hs == x) for x in range(5)]  # 0-4 hydrogens
        features.extend(hs_encoding)
        
        return features
    
    def get_bond_features(self, bond):
        """
        Extracts features from bonds
        """
        features = []
        
        # Bond type
        bond_type = bond.GetBondType()
        bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, 
                     Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
        bond_encoding = [int(bond_type == x) for x in bond_types]
        features.extend(bond_encoding)
        
        # Stereo
        stereo = bond.GetStereo()
        stereo_types = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY,
                       Chem.BondStereo.STEREOZ, Chem.BondStereo.STEREOE]
        stereo_encoding = [int(stereo == x) for x in stereo_types]
        features.extend(stereo_encoding)
        
        # Conjugated and aromatic
        features.append(int(bond.GetIsConjugated()))
        features.append(int(bond.GetIsAromatic()))
        
        return features
    
    def smiles_to_graph(self, smiles):
        """
        Converts SMILES string to PyTorch Geometric Data object
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            
            # Removal of hydrogens for simplicity
            mol = Chem.RemoveHs(mol)
            
            # Node features (atoms)
            node_features = []
            for atom in mol.GetAtoms():
                features = self.get_atom_features(atom)
                node_features.append(features)
            
            # Edge features and connections
            edge_indices = []
            edge_features = []
            
            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                
                #Undirected graph
                edge_indices.extend([[i, j], [j, i]])
                
                bond_feat = self.get_bond_features(bond)
                edge_features.extend([bond_feat, bond_feat])
            
            # Convert to tensors
            x = torch.tensor(node_features, dtype=torch.float)
            
            if len(edge_indices) > 0:
                edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
                edge_attr = torch.tensor(edge_features, dtype=torch.float)
            else:
                # Molecule without bonds (single atom)
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, len(self.get_bond_features(mol.GetBonds()[0])) if mol.GetNumBonds() > 0 else 10), dtype=torch.float)
            
            # Creation of Data object
            data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
            
            return data
            
        except Exception as e:
            print(f"Error converting SMILES {smiles}: {e}")
            return None

class GraphConvNet(nn.Module):
    """
    Graph Convolutional Network for molecular representation
    """
    
    def __init__(self, node_input_dim=32, edge_input_dim=10, hidden_dim=96, 
                 output_dim=256, num_layers=3, dropout=0.1):
        super(GraphConvNet, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        
        # Input transformation
        self.node_embedding = nn.Linear(node_input_dim, hidden_dim)
        self.edge_embedding = nn.Linear(edge_input_dim, hidden_dim)
        
        # GINE layers
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        #CHANGE
        for _ in range(num_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.conv_layers.append(GINEConv(mlp, train_eps=True))
            self.norm_layers.append(GraphNorm(hidden_dim))
        #END_CHANGE
        
        # Graph-level representation
        self.graph_norm = nn.LayerNorm(hidden_dim * 2)  # Mean + Max pooling
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def debug_weights(self):
        w = self.node_embedding.weight.data
        b = self.node_embedding.bias.data
        print(
            "node_embedding weight:",
            "min", float(w.min()), "max", float(w.max()),
            "any_nan", torch.isnan(w).any().item(),
            "any_inf", torch.isinf(w).any().item(),
        )
        print(
            "node_embedding bias:",
            "min", float(b.min()), "max", float(b.max()),
            "any_nan", torch.isnan(b).any().item(),
            "any_inf", torch.isinf(b).any().item(),
        )
    
    def forward(self, batch):
        """
        Forward pass
        """
        x, edge_index, edge_attr, batch_idx = batch.x, batch.edge_index, batch.edge_attr, batch.batch
        
        #DEBUG 1: raw node features
        if not torch.isfinite(x).all():
            log_debug("[GIN_X_INPUT_NAN] node features contain NaNs or Infs")
            raise RuntimeError("GCN input x has NaNs/Inf")
        
        # Node and edge embeddings
        x = self.node_embedding(x)
        x = F.relu(x)

        #DEBUG 2: after node embedding
        if not torch.isfinite(x).all():
            log_debug("[GIN_X_EMBED_NAN] after node_embedding+ReLU got NaNs")
            raise RuntimeError("GCN node embedding output has NaNs")
        
        #CHANGE
        # Edge embedding (so edges have same dim as nodes)
        edge_attr = self.edge_embedding(edge_attr)
        edge_attr = F.relu(edge_attr)
        #END_CHANGE

        # GINE layers
        for i in range(self.num_layers):
            x_residual = x
            #CHANGE
            x = self.conv_layers[i](x, edge_index, edge_attr)
            #END_CHANGE
            x = self.norm_layers[i](x,batch_idx)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            
            if not torch.isfinite(x).all():
                log_debug(f"[GCN_LAYER_NAN] layer={i}")
                raise RuntimeError(f"GCN layer {i} produced NaNs")
            
            # Residual connection
            if x.shape == x_residual.shape:
                x = x + x_residual
        
        # Graph-level pooling
        graph_mean = global_mean_pool(x, batch_idx)
        graph_max = global_max_pool(x, batch_idx)
        graph_repr = torch.cat([graph_mean, graph_max], dim=1)
        
        # Normalization
        graph_repr = self.graph_norm(graph_repr)

        if not torch.isfinite(graph_repr).all():
            log_debug("[GIN_GRAPH_REPR_NAN] after graph_norm got NaNs")
            raise RuntimeError("graph_repr has NaNs")
        
        # Final projection
        output = self.output_projection(graph_repr)

        if not torch.isfinite(output).all():
            log_debug("[GIN_OUTPUT_NAN] final GCN output has NaNs")
            raise RuntimeError("GIN output has NaNs")
        
        return output

class MolecularGraphEncoder(nn.Module):
    """
    Complete encoder that combines SMILES parsing with GINe
    """
    
    def __init__(self, output_dim=256, hidden_dim=96, num_layers=3, dropout=0.1):
        super(MolecularGraphEncoder, self).__init__()
        
        self.graph_converter = MolecularGraph()
        
        # Calculation of features
        # Atom features: 10 (atomic_num) + 6 (degree) + 5 (charge) + 5 (hybrid) + 1 (aromatic) + 5 (hydrogens) = 32
        # Bond features: 4 (bond_type) + 4 (stereo) + 1 (conjugated) + 1 (aromatic) = 10
        node_dim = 32
        edge_dim = 10
        
        self.gcn = GraphConvNet(
            node_input_dim=node_dim,
            edge_input_dim=edge_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def forward(self, smiles_list):
        """
        Processes a list of SMILES strings
        """
        # Convertion of SMILES into graphs
        graphs = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            graph = self.graph_converter.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)
        
        if len(graphs) == 0:
            # Fallback 
            batch_size = len(smiles_list)
            output_dim = self.gcn.output_dim
            return torch.zeros(batch_size, output_dim), list(range(batch_size))
        
        #  batch
        batch = Batch.from_data_list(graphs)
        device = next(self.gcn.parameters()).device
        batch = batch.to(device)
        # Forward pass
        molecular_representations = self.gcn(batch)


        # DEBUG: check raw GCN output before re-indexing
        if not torch.isfinite(molecular_representations).all():
            log_debug(
                f"[GIN_INTERNAL_NAN] "
                f"min={molecular_representations.min().item()} "
                f"max={molecular_representations.max().item()} "
                f"any_nan={torch.isnan(molecular_representations).any().item()} "
                f"any_inf={torch.isinf(molecular_representations).any().item()}"
            )
            raise RuntimeError("GIN internal output (per-graph) has NaNs")

        
        # batch with zeros for invalid molecules
        batch_size = len(smiles_list)
        output_dim = molecular_representations.shape[1]
        full_output = torch.zeros(batch_size, output_dim, device=molecular_representations.device)
        
        for i, valid_idx in enumerate(valid_indices):
            full_output[valid_idx] = molecular_representations[i]
        
        return full_output, valid_indices

def test_graph_encoder():
    """
    Test function 
    """
    print("🧪 Testing Molecular Graph Encoder...")
    
    # Test SMILES
    test_smiles = [
        "CC(C)Cc1ccc(C(C)C(=O)O)cc1",  # Ibuprofen
        "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",  # Benzophenone-like
        "CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN3C4=CC=CC=C4F)C#N",  # Complex drug-like
        "INVALID_SMILES",  # Invalid for testing
        "CCO"  # Ethanol (simple)
    ]
    
    try:
        # Creation of encoder
        encoder = MolecularGraphEncoder(output_dim=256, hidden_dim=128)
        encoder.eval()
        
        # Test encoding
        with torch.no_grad():
            molecular_features, valid_indices = encoder(test_smiles)
        
        print(f"✓ Input SMILES: {len(test_smiles)}")
        print(f"✓ Valid molecules: {len(valid_indices)}")
        print(f"✓ Output shape: {molecular_features.shape}")
        print(f"✓ Valid indices: {valid_indices}")
        
        # Test individual graph conversion
        converter = MolecularGraph()
        for i, smiles in enumerate(test_smiles):
            graph = converter.smiles_to_graph(smiles)
            if graph is not None:
                print(f"✓ SMILES {i}: {graph.x.shape[0]} atoms, {graph.edge_index.shape[1]} edges")
            else:
                print(f"✗ SMILES {i}: Invalid")
        
        print("✅ Graph Encoder test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Error in the Graph Encoder test: {e}")
        import traceback
        traceback.print_exc()
        return False


def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")

    # Optional: print layer-wise breakdown
    print("\nLayer-wise parameter count:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:<50} {param.numel():,}")

    return trainable_params


if __name__ == "__main__":
    test_graph_encoder()
    encoder = MolecularGraphEncoder(output_dim=256, hidden_dim=96)
    n_params = count_parameters(encoder)
    print(f"\nTotal trainable parameters in MolecularGraphEncoder: {n_params:,}")