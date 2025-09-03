# Stellar Classification with Gaia DR3 Using Variational Quantum Classifiers (VQC)

## Description
This project uses **Gaia DR3 data** to classify stars into nearby and distant categories using **Variational Quantum Classifiers (VQC)**.  
It includes:  
- Chunked and robust Gaia data fetching  
- Feature selection pipeline  
- VQC training using PennyLane and PyTorch  
- Logging of jobs and results  

---

## Requirements / Dependencies

This project requires the following Python libraries:

- `torch` (PyTorch)  
- `torchvision`  
- `torchaudio`  
- `torchviz` (for visualizing computational graphs)  
- `pennylane` (Quantum machine learning)  
- `pandas`  
- `scikit-learn`  
- `matplotlib`  
- `seaborn`  
- `astroquery` (for Gaia DR3 data access)  
- `requests` (for connectivity checks)  

> **Optional:** Graphviz system installation is required for `torchviz`. Download from [Graphviz](https://graphviz.org/download/) and add `bin/` to your PATH.  

---
```powershell
python -m pip install torch torchvision torchaudio torchviz pennylane pandas scikit-learn matplotlib seaborn astroquery requests
