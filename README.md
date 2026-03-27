# ZIP-RC: Zero-Overhead Introspection for Adaptive Test-Time Compute

An implementation of the **ZIP-RC** framework, enabling Large Language Models (LLMs) to predict their own success (reward) and computational cost (generation length) in real-time—without any additional inference overhead.

This method allows models to "introspect" and adaptively decide when to stop, when to branch, and which paths to prune during generation.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{manvi2024ziprc,
  title={Zero-Overhead Introspection for Adaptive Test-Time Compute},
  author={Manvi, Rohin and Hong, Joey and Seyde, Tim and Labonne, Maxime and Lechner, Mathias and Levine, Sergey},
  journal={arXiv preprint arXiv:2512.01457},
  year={2024}
}
```

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-username/zip-rc.git
cd zip-rc

# Install dependencies
pip install -r requirements.txt
```

### Run Locally

```bash
# Step 1: Generate rollouts from sample prompts
python generate_rollouts.py

# Step 2: Score the rollouts
python score_rollouts.py

# Step 3: Build the prefix dataset for training
python build_prefix_dataset.py
```

### Run on Google Colab

Open the notebook in Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/your-username/zip-rc/blob/main/zip_rc_colab.ipynb)

Or manually:
1. Upload the repository files to Colab
2. Run `!pip install -r requirements.txt`
3. Follow the pipeline steps above

**Note:** Running on Colab requires a GPU runtime for reasonable performance with LLaMA models.
