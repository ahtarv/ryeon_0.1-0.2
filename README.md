# Ryeon: Architectural Co-Optimization for Property-Aware Molecular Embeddings

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Figshare](https://img.shields.io/badge/Figshare-10.6084%2Fm9.figshare.32115253-blue.svg)](https://doi.org/10.6084/m9.figshare.32115253)

Systematic evaluation of learnable electron-bias parameters applied to grammar tokens in MolGramTreeNet for molecular property prediction. This work demonstrates that property-aware feature modifications must be co-optimized with model architecture.

## Key Findings

- **Architecture-Modification Interaction**: Electron-rich token biasing provides performance benefits only when combined with shallow graph convolutional networks (3 layers) and appropriate regularization.
- **ESOL Performance**: Achieved 0.4655 RMSE on aqueous solubility prediction, representing 5.9% improvement over matched no-bias controls and 14.9% improvement over published MolGramTreeNet baselines.
- **Over-Smoothing Effects**: Deep GCNs (5 layers) suppress token-level bias signals through over-smoothing, while shallow networks (3 layers) preserve them.
- **Task-Dependent Utility**: ESOL shows benefits (+5.9%), FreeSolv shows convergence to zero, and preliminary QM9 experiments show tenfold parameter growth, indicating strong electron-dependence effects.

## Architecture

The model extends MolGramTreeNet with a learnable electron_bias parameter applied to electron-relevant grammar tokens:

```
h0 = LayerNorm(Etok(x) + Epos(p) + Melectron(x) ⊙ β · 1dmodel)
```

Where:
- `β` is a single learnable scalar parameter
- `Melectron(x)` is a binary mask for electron-relevant tokens (N, O, S, aromatic atoms, pi bonds)
- Only +1 parameter added to the entire model

## Results

### ESOL Ablation Study

| Variant | Description | GCN Layers | Dropout | Test RMSE | Δ vs Baseline |
|---------|-------------|------------|---------|-----------|---------------|
| 0 | Baseline (5L, bias) | 5 | 0.1 | 0.4851 | - |
| 1 | No Bias (5L) | 5 | 0.1 | 0.4748 | +2.1% |
| 2 | Wrong Chemistry (5L) | 5 | 0.1 | 0.5089 | -4.9% |
| 3 | Static Bias (5L) | 5 | 0.1 | 0.5261 | -8.5% |
| 4 | Arch Tuned + Bias (3L) | 3 | 0.2 | **0.4655** | **+4.0%** |
| 5 | Arch Tuned + No Bias (3L) | 3 | 0.2 | 0.4929 | -1.6% |

Published MolGramTreeNet baseline: 0.547 RMSE

### Task-Dependent Behavior

| Dataset | Electron Token Freq. | Final β | Interpretation |
|---------|---------------------|---------|----------------|
| ESOL | 8.3 tokens/mol | ≈ 0.10 | High density, bias helps |
| FreeSolv | 4.1 tokens/mol | 0.00 | Low density, bias not helpful |
| QM9 | Not quantified | 0.00033 | Very high quantum effects, bias grows 10x |

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- PyTorch Geometric
- RDKit
- NLTK
- NumPy
- Pandas

### Setup

```bash
# Clone repository
git clone https://github.com/ahtarv/ryeon_0.1-0.2.git
cd ryeon_0.1-0.2

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install rdkit nltk pandas numpy
```

## Usage

### ESOL Experiments (Version 0.1)

```bash
# Run ablation study on ESOL dataset
jupyter notebook 0.1v/ablationstudyonesolofryueon0.1v.ipynb
```

The notebook contains 6 systematic variants testing the architecture-modification interaction.

### QM9 Experiments (Version 0.2)

```bash
# Run QM9 quantum property prediction
jupyter notebook 0.2v/rryueon0-2vqm9.ipynb
```

Preliminary experiments on dipole moment, polarizability, HOMO, LUMO, and gap.

## Datasets

### ESOL (Aqueous Solubility)
- Source: Delaney (2004)
- Total molecules: 1035 valid (from 1128)
- Property: Log aqueous solubility at 25°C
- Range: -11.6 to 1.6 log S
- Split: 843 train / 96 val / 96 test

### FreeSolv (Hydration Free Energy)
- Source: Mobley & Guthrie (2014)
- Total molecules: 544 valid (from 642)
- Property: Hydration free energy (ΔGhydration)
- Range: -25.5 to 4.5 kcal/mol
- Split: 448 train / 48 val / 48 test

### QM9 (Quantum Properties)
- Source: Quantum Machine 9 dataset
- Total molecules: 132,180 valid (from 133,886)
- Properties: Dipole moment, polarizability, HOMO, LUMO, gap
- Split: 105,744 train / 13,218 val / 13,218 test

## Recommended Configuration

For ESOL-like small-dataset molecular property prediction tasks:

| Hyperparameter | Value |
|----------------|-------|
| GCN layers | 3 (shallow to prevent over-smoothing) |
| Dropout (predictor) | 0.2 (high regularization for small data) |
| electron_bias init | 0.1 (learnable) |
| Learning rate | 2 × 10⁻⁵ (conservative) |
| Weight decay | 0.0 (dropout sufficient) |
| Batch size | 16 (limited by dataset) |

## Project Structure

```
ryeon_0.1-0.2/
├── 0.1v/
│   ├── ablationstudyonesolofryueon0.1v.ipynb    # Main ESOL ablation study
│   ├── ablationv2.ipynb                          # Alternative ablation experiments
│   └── firstrunofesol_freesol_v1.ipynb          # Initial ESOL/FreeSolv runs
├── 0.2v/
│   └── rryueon0-2vqm9.ipynb                     # QM9 quantum property experiments
└── README.md
```

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{patil2026ryeon,
  title={Ryeon: Architectural Co-Optimization for Property-Aware Molecular Embeddings},
  author={Patil, Atharv Kamlesh and Karnavat, Payal},
  journal={arXiv preprint arXiv:2026.xxxxx},
  year={2026}
}
```

## Authors

- Atharv Kamlesh Patil - SVKM's Dwarkadas J. Sanghvi College of Engineering, Mumbai, India
- Payal Karnavat - SVKM's Dwarkadas J. Sanghvi College of Engineering, Mumbai, India

## Acknowledgments

- MolGramTreeNet authors for the base architecture
- NVIDIA Academic Partnership program for computational resources
- Kaggle for GPU compute resources

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References

1. Zhang et al. (2026). MolGramTreeNet: A multimodal molecular property prediction model via grammar tree-constrained molecular representation. iScience, 29:114928.
2. Kipf & Welling (2017). Semi-supervised classification with graph convolutional networks. ICLR.
3. Gilmer et al. (2017). Neural message passing for quantum chemistry. ICML.
4. Delaney (2004). ESOL: Estimating aqueous solubility directly from molecular structure. J. Chem. Inf. Comput. Sci., 44(3):1000-1005.
5. Mobley & Guthrie (2014). FreeSolv: a database of experimental and calculated hydration free energies. J. Comput. Aided Mol. Des., 28(7):711-720.

## Contact

For questions or issues, please open an issue on GitHub or contact:
- ATHARV.PATIL71@svkmmumbai.onmicrosoft.com
- PAYAL.KARNAVAT08@svkmmumbai.onmicrosoft.com
