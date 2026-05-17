# HMS-PLI: Hierarchical Multi-Scale Protein-Ligand Interaction Prediction

This repository contains the official TensorFlow/Keras implementation of the **HMS-PLI** framework, a dual-task architecture for predicting protein-ligand interaction probabilities and binding affinities.

## 🚀 Environment Setup
To guarantee complete reproducibility of the manuscript results, please configure your Python environment using the exact versions listed below:

```bash
# Recommended: Create a clean conda environment
conda create -n hms_pli python=3.10
conda activate hms_pli

# Install exact dependencies to avoid version conflicts
pip install tensorflow==2.15.0
pip install numpy==1.24.3
pip install pandas scikit-learn matplotlib seaborn