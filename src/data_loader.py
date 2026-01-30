import os

# Change to project root directory (parent of src)
script_dir = os.path.dirname(os.path.abspath(__file__))
project_dir = os.path.dirname(script_dir)
os.chdir(project_dir)
print(f"Working directory: {os.getcwd()}")
"""
Data Loader for Quantum Multi-Target Drug Discovery
Creates a synthetic datasets for 5 protein targets
"""

import torch
import pandas as pd
import numpy as np
import pickle
import json
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from rdkit import Chem
from rdkit.Chem import Descriptors,rdMolDescriptors
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from collections import Counter,defaultdict
from rdkit.Chem import AllChem

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')




class MultiTargetDrugDataset(Dataset):
    """
    Dataset class for multi-target drug discovery
    """
    
    def __init__(self, data_list=None):
        """
        Args:
            data_list: List of dictionaries with data
        """
        self.data = data_list if data_list is not None else []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Convert to tensors
        result = {
            'molecule_smiles': sample['molecule_smiles'],
            'target_proteins': sample['target_proteins'],
            'target_names': sample['target_names'],
            'main_affinity': torch.tensor(sample['main_affinity'], dtype=torch.float32),
            'individual_affinities': torch.tensor(sample['individual_affinities'], dtype=torch.float32),
            'molecular_properties': torch.tensor(sample['molecular_properties'], dtype=torch.float32)
        }
        
        return result

class SyntheticDataGenerator:
    """
    Generator for synthetic multi-target drug data
    """
    
    def __init__(self, config_path="config.json"):
        """
        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Target proteins and their names
        self.target_names = list(self.config['targets'].keys())
        self.target_descriptions = list(self.config['targets'].values())
        
        # Sample SMILES strings (drug-like molecules)
        self.sample_smiles = [
            "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",      # Ibuprofen-like
            "CC1=CC=C(C=C1)C(=O)C2=CC=CC=C2",      # Benzophenone-like  
            "CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O",    # Salbutamol-like
            "CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN3C4=CC=CC=C4F)C#N",  # Complex drug-like
            "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C", # Another complex
            "C1=CC=C(C=C1)C2=CC=C(C=C2)C3=CC=CC=C3",           # Triphenyl-like
            "CN(C)CCN1C2=CC=CC=C2SC3=C1C=C(C=C3)C(F)(F)F",     # Antipsychotic-like
            "CC(C)NCC(C1=CC(=C(C=C1)O)CO)O",                   # Beta-agonist like
            "COC1=C(C=C2C(=C1)C(=CN2)CCN)OC",                  # Serotonin-like
            "CC1=CN=C(C=C1)C2=CC=CC=C2",                       # Pyridine derivative
        ]
        
        # Sample protein sequences (simplified for development)
        self.sample_proteins = {
            'CDK2': "MENFQKVEKIGEGTYGVVYKARNKLTGEVVALKKIRLDTETEGVPSTAIREISLLKELKDDNIVRLYDIVHSDAHKLYLVFEFLDLDLKRYMEGIPKDQPLGADIVKKFMMQLCKGIAYCHSHRILHRDLKPQNLLINTEGAIKLADFGLARAFGVPVRTYTHEVVTLWYRAPEILLGCKYYSTAVDIWSLGCIFAEMVTRRALFPGDSEIDQLFRIFRTLGTPDEVVWPGVTSMPDYKPSFPKWARQDFSKVVPPLDEDGRSLLSQMLHYDPNKRISAKAALAHPFFQDVTKPVPHLRL",
            'DRD2': "MDPLNLSWYDDDLERQNWSRPFNGSDGKADRPHYNYYATLLTLLIAVIVFGNVLVCWAVWLNSNLQNVTNYFVVSLAAADLLLALLVLPLFVYSEVQGGVWLSQVLCDLWLAIDYVASNASVMNLLIISFDRYFCVTKPLTYPVKRTTKMAGMMIAAAWVLSFILWAPAILFWQFIVGVRTVEDGECYIQFFSNAAVTFGTAIAAFYLPVIIMTVLYWHISRASKSRIKKDKKEPVANVTQILAFITVWTPYNIMVLVNTFCAPCIIPNTVWTIGYWLCYINSTINPACYALCNATFKKTFKHLLMCHYKNIGATMSQVSSTESVSFYVPFGVTVLVYARIYVVLKQIRRKRIN",
            'HIV1_PR': "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNFPISPIETVPVKLKPGMDGPKVKQWPLTEEKIKALVEICTEMEKEGKISKIGPENPYNTPVFAIKKKDSTKWRKLVDFRELNKRTQDFWEVQLGIPHPAGLKKKKSVTVLDVGDAYFSVPLDEDFRKYTAFTIPSINNETPGIRYQYNVLPQGWKGSPAIFQSSMTKILEPFRKQNPDIVIYQYMDDLYVGSDLEIGQHRTKIEELRQHLLRWGLTTPDKKHQKEPPFLWMGYELHPDKWTVQPIVLPEKDSWTVNDIQKLVGKLNWASQIYPGIKVRELCKLLRGTKALTEVIPLTEEAELELAENREILKAPVHGVYYDPSKDLIAEIQKQGQGQWTYQIYQEPFKNLKTGKYARMRGAHTNDVKQLTEAVQKITTESIVIWGKTPKFKLPIQKETWETWWTEYWQATWIPEWEFVNTPPLVKLWYQLEKEPIVGAETFYVDGAANRETKLGKAGYVTNRGRQKVVTLTDTTNQKTELQAIYLALQDSGLEVNIVTDSQYALGIIQAQPDQSESELVNQIIEQLIKKEKVYLAWVPAHKGIGGNEQVDKLVSAGIRKVLFLDGIDKAQEEHEKYHSNWRAMASDFNLPPVVAKEIVASCDKCQLKGEAMHGQVDCSPGIWQLDCTHLEGKVILVAVHVASGYIEAEVIPAETGQETAYFLLKLAGRWPVKTIHTDNGSNFTGATVRAACWWAGIKQEFGIPYNPQSQGVVESMNKELKKIIGQVRDQAEHLKTAVQMAVFIHNFKRKGGIGGYSAGERIVDIIATDIQTKELQKQITKIQNFRVYYRDSRNPLWKGPAKLLWKGEGAVVIQDNSDIKVVPRRKAKIIRDYGKQMAGDDCVASRQDED",
            'hERG': "MEAVKLLDSHHGRLREAGPLGPGAPPSPPRPQVYLQMLDHLDGNPYISRRLFQAFSAYEDLQKLMKDDYKKELPPPGGPARGFPPMPWQVLCSRLQQAYVFMGDSLKSRMCLLWDDDMYGLQKLGKALQHILNHIPQNEIVIQGMQNPDADFKCLYGDIIYHDLYNQYNQIHCQFKFIGMKYRAEEVAHHVIMEYCDPKAKLMFMFSQHMSAALDYCHHFAWPCFRLLRKEVGDQEAADGRLLKALAELIKKHGQPVAHHGVDCSLFGCQRLMQMPQALVLLITFLLTGILAAGAGAMGPLDYGGLSFDDLRSYMKEADPSGRLDMLHSLRSLGKDVGITGDATSGWLGQFGGTGLLGDMMSPGALSFLQLQVLILPVHVVLVSKLWSGLRRGDPAFLLRLVQGMYSYYLSASVTPAVVQIVGMFGLLVSTLFANSMPRFTQTYAILVLLPYSLLVVPSQRTPPYHMHSYKYFLGCPLLRMTLLPQDLSEVTMTTHAGKDLHIYPGCVELARIQRMNGDPGPPIPFPVHHTGYMGAALKQGPGPPPDVHYAARHTVYNSRLHQINRSWLGTLSHSDVPLEFPSCSDQRLGQRLLLLAAGASSLPAFEFLAQFPEPGIAGQFLNLLGADLGYHMTYRTADFQELPPQFGSVAGLLLQYPRLGLAAPKFPGYLRGLRKHMIVLMRRHDDRSKTQVNRSDSRLSFCRLAKQHAWGLLLLNFFRPFGCRVMQLAIHGFMGPEKKLQLLHYAARLSFEGKLLVGACPGGPDGMRFLLRNPLFLLLLSYLSLKGLTSAVVQLFLSTDGGQRVLIYGRDSSGRSLGGLLLSEEQPLLNLLLRLSLRKRLFRWSPGECSAAQRPAQRGRSSLLLSRSRKSASDQHSHLDYEPRLSNLLNLLVFPQKPAAEQENQLHRGRLRSWSAQLDLRNQLLQLSVLLPFNQATRRQEHLWLQLPQLSLGDMDLDLLPLAALHRGHSVLGDGSRPCDVLGLVVLYYLSPSASFVLHLQALLAYKDQRLSLAATLHAGYVLTTQTLVLIVAVSAGLAASVALVLVSQLMKLEKEWDNFLQALRPHQRLQVRMSLLQRRGYQYSLTSVGLAIFCLPTLLLMLEFLNRQMSMSQSKRWGDRAGTLIAALLADLHIYRYPDLLRFQASDGRKSPLGSNLQLDGSLVGRSLMYLQYLDNSRLVGDQFLLTAQHLRGLLRAHLLLLLSRRRSCRAVTTGYQLLSLSPTLDDHLGFGDQLGLLRLGLLNHKGRKTGPLNSLDISLPLYLQATLLLQLSQLLAEEQELLDTLVVSGTDLQARQHQREGSPSQARLFTPNQLTMGTPDLGRYGHRRSKTPQRSPPPQPPPQALLRLSQPVLARWSLGGSVPPLEHQESLPTSRHGVRAPHLALPRPRPSGSQRYRGVSESRRRPPLLLLDLLASGRQDLHGGAAGVGAVGRLDLERRPPGPGLGQPGRASRLPGPLLLLLGQRSARQGGAAEPAAARGWGGWEPLGLFPADQRQPPQPPGERLRGRPGQLPQAPPAAVGSVSRGGLALFAPPPPGPPLDSAGDPQPRPRTGLLLSSPQCQDTLLGTPGWDGEAKISQLGLMLGLLQLGQQLQHQPHRGRHSFAGLMRDGSVQLGSRQAMRGRWALDLRGRLLPWRRLHLGLSGARTVGAGAVGRVLTDRHWQHRGQLFALPPHGGLRYRRLSLLLQQLLQQLLRQQRRPAPQKPRHGAPACGPQTRGARQRGRSGAPSYCCYQ",
            'ESR1': "MTMTLHTKASGMALLHQIQGNELEPLNRPQLKIPLERPLGEVYLDSSKPAVYNYPEGAAYEFNAAAAANAQVYGQTGLPYGPGSEAAAFGSNGLGGFPPLNSVSPSPLMLLHPPPQLSPFLQPHGQQVPYYLENEPSGYTVREAGPPAFYRPNSDNRRQGGRERLASTNDKGSMAMESAKETRYCAVCNDYASGYHYGVWSCEGCKAFFKRSIQGHNDYMCPATNQCTIDKNRRKSCQACRLRKCYEVGMMKGGIRKDRRGGRMLKHKRQRDDGEGRGEVGSAGDMRAANLWPSPLMIKRSKKNSLALSLTADQMVSALLDAEPPILYSEYDPTRPFSEASMMGLLTNLADRELVHMINWAKRVPGFVDLTLHDQVHLLECAWLEILMIGLVWRSMEHPGKLLFAPNLLLDRNQGKCVEGMVEIFDMLLATSSRFRMMNLQGEEFVCLKSIILLNSGVYTFLSSTLKSLEEKDHIHRVLDKITDTLIHLMAKAGLTLQQQHQRLAQLLLILSHIRHMSNKGMEHLYSMKCKNVVPLYDLLLEMLDAHRLHAPTSRGGASVEETDQSHLATAGSTSSHSLQKYYITGEAEGFPATV"
        }
        
        print(f"✓ SyntheticDataGenerator created with {len(self.target_names)} targets")


    def canonicalize_smiles(self, smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return smiles
        return Chem.MolToSmiles(mol)

    def _add_alkyl_substituent(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Add a small alkyl chain (CH3 / CH2-CH3) to a carbon atom.
        """
        em = Chem.RWMol(mol)
        num_atoms = em.GetNumAtoms()
        candidate_idxs = [
            i for i in range(num_atoms)
            if em.GetAtomWithIdx(i).GetAtomicNum() == 6  # carbon
        ]
        if not candidate_idxs:
            return mol

        idx = random.choice(candidate_idxs)

        # create a CH3 or CH2-CH3 fragment
        frag_smiles = random.choice(["C", "CC"])
        frag = Chem.MolFromSmiles(frag_smiles)
        if frag is None:
            return mol

        combined = Chem.CombineMols(em, frag)
        em2 = Chem.RWMol(combined)

        # new atoms start after original count
        base_n = num_atoms
        # connect first atom of fragment to chosen carbon
        em2.AddBond(idx, base_n, Chem.BondType.SINGLE)

        new_mol = em2.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol

    def _add_halogen(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Attach a halogen (F / Cl) to a carbon atom.
        """
        em = Chem.RWMol(mol)
        num_atoms = em.GetNumAtoms()
        candidate_idxs = [
            i for i in range(num_atoms)
            if em.GetAtomWithIdx(i).GetAtomicNum() == 6
        ]
        if not candidate_idxs:
            return mol

        idx = random.choice(candidate_idxs)

        halogen_symbol = random.choice(["F", "Cl"])
        halogen = Chem.Atom(9 if halogen_symbol == "F" else 17)
        h_idx = em.AddAtom(halogen)
        em.AddBond(idx, h_idx, Chem.BondType.SINGLE)

        new_mol = em.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol

    def _heteroatom_substitution(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Replace an aromatic carbon with a nitrogen.
        """
        em = Chem.RWMol(mol)
        aromatic_c = [
            i for i in range(em.GetNumAtoms())
            if em.GetAtomWithIdx(i).GetAtomicNum() == 6
            and em.GetAtomWithIdx(i).GetIsAromatic()
        ]
        if not aromatic_c:
            return mol

        idx = random.choice(aromatic_c)
        atom = em.GetAtomWithIdx(idx)
        atom.SetAtomicNum(7)  # C → N

        new_mol = em.GetMol()
        Chem.SanitizeMol(new_mol)
        return new_mol

    def modify_molecule(self, base_smiles: str, max_tries: int = 10) -> str:
        """
        Apply a random medchem-like modification and return a valid SMILES.
        Falls back to canonical base SMILES if all attempts fail.
        """
        base_mol = Chem.MolFromSmiles(base_smiles)
        if base_mol is None:
            return base_smiles

        for _ in range(max_tries):
            try:
                # start from canonicalised version to keep things consistent
                mol = Chem.MolFromSmiles(Chem.MolToSmiles(base_mol))
                if mol is None:
                    break

                choice = random.random()
                if choice < 0.4:
                    new_mol = self._add_alkyl_substituent(mol)
                elif choice < 0.8:
                    new_mol = self._add_halogen(mol)
                else:
                    new_mol = self._heteroatom_substitution(mol)

                Chem.SanitizeMol(new_mol)
                new_smiles = Chem.MolToSmiles(new_mol)
                return new_smiles
            except Exception:
                # try another modification strategy / atom
                continue

        # fallback: just return canonical base SMILES
        return Chem.MolToSmiles(base_mol)

    
    def calculate_molecular_properties(self, smiles):
        """
        Calculation of molecular properties from SMILES
        """
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Return default values if the SMILES are invalid
                return [300.0, 2.0, 3.0, 2.0, 80.0, 5.0, 1.0, 0.5]
            
            properties = [
                Descriptors.MolWt(mol),                    # Molecular weight
                Descriptors.MolLogP(mol),                  # LogP  
                Descriptors.NumHDonors(mol),               # H donors
                Descriptors.NumHAcceptors(mol),            # H acceptors
                Descriptors.TPSA(mol),                     # Polar surface area
                Descriptors.NumRotatableBonds(mol),        # Rotatable bonds
                Descriptors.NumAromaticRings(mol),         # Aromatic rings
                rdMolDescriptors.CalcFractionCSP3(mol)     # FractionCsp3
            ]
            
            return properties
            
        except Exception as e:
            print(f"Error in the calculation of the properties for {smiles}: {e}")
            return [300.0, 2.0, 3.0, 2.0, 80.0, 5.0, 1.0, 0.5]
    
    def generate_binding_affinities(self, mol_properties):
        """
        Generation of realistic binding affinities based on molecular properties
        """
        # Base affinity based on the molecular weight and LogP
        mol_weight, logp = mol_properties[0], mol_properties[1]
        
        # Ideal ranges for drug-like molecules
        weight_factor = 1.0 - abs(mol_weight - 350) / 200.0  # Optimal ~350 Da
        logp_factor = 1.0 - abs(logp - 2.5) / 3.0           # Optimal LogP ~2.5
        
        # Base affinity (pKd scale: 5-10)
        base_affinity = 6.0 + 2.0 * max(0, weight_factor) + 1.5 * max(0, logp_factor)
        base_affinity += np.random.normal(0, 0.5)  # Add noise
        base_affinity = np.clip(base_affinity, 4.0, 10.0)
        
        # Individual target affinities with target-specific variations
        target_factors = {
            'CDK2': 1.0 + 0.2 * (mol_properties[6] > 1),      # Likes aromatic rings
            'DRD2': 1.0 + 0.3 * (mol_properties[1] > 2),      # Likes lipophilic
            'HIV1_PR': 1.0 + 0.25 * (mol_weight > 400),       # Likes larger molecules
            'hERG': 1.0 - 0.4 * (mol_properties[1] > 4),      # Dislikes very lipophilic
            'ESR1': 1.0 + 0.2 * (mol_properties[3] > 3)       # Likes H-bond acceptors
        }
        
        individual_affinities = []
        for target in self.target_names:
            target_affinity = base_affinity * target_factors[target]
            target_affinity += np.random.normal(0, 0.3)  # Target-specific noise
            target_affinity = np.clip(target_affinity, 4.0, 10.0)
            individual_affinities.append(target_affinity)
        
        return base_affinity, individual_affinities
    
    def generate_dataset(self, n_samples=1000):
        """
        Gemeration of a complete synthetic dataset
        """
        print(f"🧪 Creating synthetic dataset with {n_samples} samples...")
        
        dataset = []
        modification_prob = 0.6
        
        for i in range(n_samples):
            # Random molecule selection and modification
            base_smiles = random.choice(self.sample_smiles)

            if random.random()< modification_prob:
                smiles = self.modify_molecule(base_smiles)
            else:
                mol = Chem.MolFromSmiles(base_smiles)
                smiles = Chem.MolToSmiles(mol) if mol is not None else base_smiles

            # Calculate molecular properties
            mol_properties = self.calculate_molecular_properties(smiles)
            
            # Generate binding affinities
            main_affinity, individual_affinities = self.generate_binding_affinities(mol_properties)
            
            # Create sample
            sample = {
                'molecule_smiles': smiles,
                'target_proteins': [self.sample_proteins[target] for target in self.target_names],
                'target_names': self.target_names.copy(),
                'main_affinity': main_affinity,
                'individual_affinities': individual_affinities,
                'molecular_properties': mol_properties
            }
            
            dataset.append(sample)
            
            if (i + 1) % 200 == 0:
                print(f"{i + 1}/{n_samples} samples created")
        
        print(f"✅ The dataset has been created successfully! ({len(dataset)} samples)")
        return dataset




def assert_no_sample_overlap(train_data, val_data, test_data):
    # Use Python's id() to distinguish objects by identity
    train_ids = {id(x) for x in train_data}
    val_ids   = {id(x) for x in val_data}
    test_ids  = {id(x) for x in test_data}
    
    assert train_ids.isdisjoint(val_ids), "Leakage: samples in both TRAIN and VAL"
    assert train_ids.isdisjoint(test_ids), "Leakage: samples in both TRAIN and TEST"
    assert val_ids.isdisjoint(test_ids),   "Leakage: samples in both VAL and TEST"
    
    print("✓ No sample-level overlaps between train/val/test")

def canonicalize_smiles(s):
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return s  # fallback: just use the string as-is
    return Chem.MolToSmiles(mol)  # RDKit canonical


def split_by_molecule(full_dataset, test_size, val_size, random_state=42):
    # 1) Group samples by canonical molecule
    groups = defaultdict(list)
    for sample in full_dataset:
        key = canonicalize_smiles(sample['molecule_smiles'])
        groups[key].append(sample)

    molecule_keys = list(groups.keys())

    # 2) Split molecule groups into train / test
    train_keys, test_keys = train_test_split(
        molecule_keys,
        test_size=test_size,
        random_state=random_state
    )

    # 3) Split remaining train_keys into train / val
    train_keys, val_keys = train_test_split(
        train_keys,
        test_size=val_size / (1.0 - test_size),
        random_state=random_state
    )

    train_keys = set(train_keys)
    val_keys   = set(val_keys)
    test_keys  = set(test_keys)

    # 4) Build final sample lists
    train_data = [s for k in train_keys for s in groups[k]]
    val_data   = [s for k in val_keys   for s in groups[k]]
    test_data  = [s for k in test_keys  for s in groups[k]]

    print(f"Canonical molecules: {len(molecule_keys)}")
    print(f"Train mols: {len(train_keys)}, Val mols: {len(val_keys)}, Test mols: {len(test_keys)}")
    print(f"Train samples: {len(train_data)}, Val samples: {len(val_data)}, Test samples: {len(test_data)}")

    return train_data, val_data, test_data

def create_data_loaders(batch_size: int = 32, num_workers: int = 0,config_path="config.json"):
    """
    Creation of train/validation/test data loaders
    """
    # Load config
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    data_config = config['data']
    cache_path = Path(project_dir) / "data" / "synthetic_dataset.pkl"
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    # Generate synthetic data
    generator = SyntheticDataGenerator(config_path)
    if cache_path.exists():
        print(f"📁 Loading cached dataset from {cache_path}")
        full_dataset = load_dataset(cache_path)
        # Optional: if you want to enforce size, subsample or regenerate when size mismatch
        if len(full_dataset) != data_config['synthetic_samples']:
            print("⚠️ Cached dataset size mismatch. Regenerating...")
            full_dataset = generator.generate_dataset(data_config['synthetic_samples'])
            save_dataset(full_dataset, cache_path)
    else:
        print(f"🧪 No cached dataset found. Generating {data_config['synthetic_samples']} samples...")
        full_dataset = generator.generate_dataset(data_config['synthetic_samples'])
        save_dataset(full_dataset, cache_path)

    all_smiles = [sample['molecule_smiles'] for sample in full_dataset]
    smiles_counts = Counter(all_smiles)

    print(f"Total samples: {len(all_smiles)}")
    print(f"Unique SMILES: {len(smiles_counts)}")
    print("\nExamples:")
    for s, c in list(smiles_counts.items())[:20]:
        print(f"{s}  -->  {c} samples")
    
    # Split data
    """
    train_data, test_data = train_test_split(
        full_dataset, 
        test_size=data_config['test_split'], 
        random_state=42
    )
    
    train_data, val_data = train_test_split(
        train_data,
        test_size=data_config['val_split'] / (1 - data_config['test_split']),
        random_state=42
    )
    """
    # SMILES-level split
    train_data, val_data, test_data = split_by_molecule(
        full_dataset,
        test_size=data_config['test_split'],
        val_size=data_config['val_split'],
        random_state=42,
    )
    
    assert_no_sample_overlap(train_data,val_data,test_data)
    # Create datasets
    train_dataset = MultiTargetDrugDataset(train_data)
    val_dataset = MultiTargetDrugDataset(val_data)
    test_dataset = MultiTargetDrugDataset(test_data)
    
    # Create data loaders
    
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    print(f"📊 Data loaders created:")
    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val:   {len(val_dataset)} samples") 
    print(f"  Test:  {len(test_dataset)} samples")
    
    return train_loader, val_loader, test_loader

def custom_collate_fn(batch):
    """
    Custom collate function for batching
    """
    molecules = [item['molecule_smiles'] for item in batch]
    target_proteins = [item['target_proteins'] for item in batch]
    target_names = [item['target_names'] for item in batch]
    
    main_affinities = torch.stack([item['main_affinity'] for item in batch])
    individual_affinities = torch.stack([item['individual_affinities'] for item in batch])
    molecular_properties = torch.stack([item['molecular_properties'] for item in batch])
    
    return {
        'molecules': molecules,
        'target_proteins': target_proteins,
        'target_names': target_names,
        'main_affinity': main_affinities.unsqueeze(1),
        'individual_affinities': individual_affinities,
        'molecular_properties': molecular_properties
    }

def save_dataset(dataset, filepath):
    """Save dataset"""
    with open(filepath, 'wb') as f:
        pickle.dump(dataset, f)
    print(f"💾 Dataset saved: {filepath}")

def load_dataset(filepath):
    """Load dataset"""
    with open(filepath, 'rb') as f:
        dataset = pickle.load(f)
    print(f"📁 Dataset loaded: {filepath} ({len(dataset)} samples)")
    return dataset

# Test function
def test_data_loader():
    """Test function for the data loader"""
    print("🧪 Testing Data Loader...")
    
    try:
        # Create data loaders
        train_loader, val_loader, test_loader = create_data_loaders()
        
        # Test one batch
        batch = next(iter(train_loader))
        print(f"\n📋 Test batch info:")
        print(f"  Molecules: {len(batch['molecules'])}")
        print(f"  Target proteins: {len(batch['target_proteins'][0])} targets")
        print(f"  Main affinity shape: {batch['main_affinity'].shape}")
        print(f"  Individual affinities shape: {batch['individual_affinities'].shape}")
        print(f"  Molecular properties shape: {batch['molecular_properties'].shape}")
        
        # Sample molecule
        print(f"\n🧬 Sample molecule: {batch['molecules'][0][:50]}...")
        print(f"🎯 Sample affinities: {batch['individual_affinities'][0].tolist()}")
        
        print("✅ Data Loader test successful!")
        return True
        
    except Exception as e:
        print(f"❌ Erron in the Data Loader test: {e}")
        return False

if __name__ == "__main__":
    test_data_loader()
