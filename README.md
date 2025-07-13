# CosmoFlow

This is the repository that goes along with the paper _CosmoFlow: Scale-Aware Representation Learning for Cosmology with Flow Matching_

We implement a flow-matching based model for representation learning of cold dark matter density fields.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/sidk2/cosmo-compression.git
   cd cosmo-compression
   ```
2. Install dependencies:
   ```bash
   pip install .
   ```
   
### Usage
- **Training**: See `src/cosmo_compression/train.py` for training script.
- **Experiments**: Explore the `examples/notebooks/` directory for Jupyter notebooks demonstrating model usage, interpolation, and downstream tasks.
- **Pretrained Models**: Pretrained checkpoints are available in the `models/` directory.
- **Data**: We include a subset of the 1P dataset from CAMELS to facilitate running the examples.

### Example: Running a Notebook
Open a notebook in `examples/notebooks/` and follow the instructions in the cells to reproduce experiments and visualizations.

## Citation
If you use this codebase in your research, please cite the corresponding paper (citation to be added).

## Contact
For questions or contributions, please open an issue or contact skannan@ucsb.edu.
