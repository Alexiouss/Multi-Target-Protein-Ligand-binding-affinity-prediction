{
  "model": {
    "n_targets": 3,
    "feature_dim": 128,
    "pca_path": "pca_amplitude.npz",
    "n_qubits": 4,
    "encoding_type": "angle",
    "use_data_reuploading": true,
    "num_layers": 3,
    "angle_pca_path": "pca_angle_4_gcn.npz"
  },
  "training": {
    "batch_size": 16,
    "epochs": 10,
    "learning_rate": 0.001,
    "num_workers": 0
  },
  "targets": {
    "CHEMBL203": "CHEMBL203",
    "CHEMBL206": "CHEMBL206",
    "CHEMBL279": "CHEMBL279"
  },
  "chembl_target_map": {
    "CHEMBL203": "Epidermal growth factor receptor",
    "CHEMBL206": "Estrogen receptor",
    "CHEMBL279": "Vascular endothelial growth factor receptor 2"
  },
  "data": {
    "synthetic_samples": 5000,
    "test_split": 0.2,
    "val_split": 0.2,
    "use_chembl": true,
    "csv_path": "C:\\Users\\Administrator\\Desktop\\Thesis_final\\scripts\\data\\chembl\\chembl_affinity_dataset.csv"
  },
  "targets_order": [
    "CHEMBL203",
    "CHEMBL206",
    "CHEMBL279"
  ]
}
