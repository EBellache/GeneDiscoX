# GeneDiscoX

*GeneDiscoX* is an enhanced fork of the [GeneDisco](https://github.com/genedisco/genedisco) benchmarking framework for **active learning in gene discovery** and high-throughput experimental design. This fork introduces **DiscoBAX**, a novel Bayesian optimization strategy, and adapts the pipeline from BoTorch/PyTorch to **JAX** for improved performance on GPU hardware. This project grew out of collaborative work with a longevity-focused biotechnology startup, aiming to expedite the discovery of gene targets and mechanisms relevant to aging and other complex phenotypes.

---

## Scientific Motivation & Relevance

Selecting which genetic perturbations (e.g., CRISPR knockouts, drug hits) to test from a large search space is a central challenge in computational biology and translational research. **Active learning** can help triage these possibilities, focusing on interventions that are most informative about the underlying biological system.

Originally, **GeneDisco** provided a flexible framework to benchmark different acquisition and modeling strategies for experimental design and gene discovery tasks (e.g. sifting through potential therapeutic targets). In *GeneDiscoX*, we extend those capabilities:

1. **New DiscoBAX Algorithm**: Incorporates a Bayesian lookahead that balances exploration of uncertain gene pathways with exploitation of known high-impact regions.  
2. **Two-Stage Selection**: Improves upon the original subset selection strategy. We first apply DiscoBAX to identify top candidate interventions, then apply a JAX-based *Mini-Batch K-Means* to ensure coverage of diverse biological mechanisms.  
3. **JAX Acceleration**: Replaces PyTorch/BoTorch backends with JAX (notably, [GPJax](https://github.com/pyMCLabs/GPJax)) for GPU acceleration and just-in-time compilation. This yields efficiency gains when scaling to larger or more complex datasets.

As such, *GeneDiscoX* aligns strongly with the demands of modern high-throughput biology, where *in silico* strategies must handle large candidate pools, parallel GPU acceleration, and sophisticated Bayesian modeling.

---

## Key Features and Contributions

1. **DiscoBAX Acquisition Strategy**  
   - **Bayesian Active Exploration**: Monte Carlo sampling from a GP (or neural model) identifies candidates likely to yield major improvements on the target phenotype.  
   - **Diverse Mechanism Coverage**: The two-step procedure ensures experiments come from multiple mechanistic “clusters,” reducing redundancy.  
   - **Downstream Utility**: Ideal for drug discovery, target identification, or longevity gene screens—where different mechanistic pathways can be highly synergistic or require independent validation.

2. **Full JAX Pipeline**  
   - **GPJax Surrogate Models**: Sparse or exact Gaussian Processes with GPU support, beneficial for large-scale CRISPR-based or drug library tasks.  
   - **Mini-Batch K-Means in JAX**: Pure JAX implementation for Stage 2 clustering, minimizing data transfers and enabling an end-to-end GPU pipeline.  
   - **Reduced Dependencies**: While the original GeneDisco required BoTorch/PyTorch, *GeneDiscoX* mostly runs in a JAX environment, simplifying the computational stack.

3. **Biologically Motivated Benchmarks**  
   - **Single-Cell & Functional Genomics Data**: Provides example tasks (e.g., `single_cell_cycle`) for quick exploration.  
   - **Easily Extended**: One may integrate custom phenotypic or omics datasets by following the standard `AbstractDataSource` interface from SlingPy.

4. **Robust Software Engineering**  
   - **Modular Abstractions**: Pipeline code is structured so new models, acquisition functions, or clustering strategies can be swapped in.  
   - **Conda Environment & GPU Setup**: Straightforward scripts ensure minimal friction for HPC or local GPU usage.

---

## Installation & Setup

We recommend a **Miniconda** environment with Python ≥ 3.9:

```bash
conda create -n genediscox python=3.9 -y
conda activate genediscox
```

**Install JAX with CUDA** (if your system has a compatible GPU):

```bash
pip install --upgrade pip
pip install --upgrade "jax[cuda]" \
  -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

Then clone and install *GeneDiscoX*:

```bash
git clone https://github.com/EBellache/GeneDiscoX.git
cd GeneDiscoX
pip install -e .
```

If the installation completes successfully, you can confirm GPU availability:

```bash
python -c "import jax; print(jax.devices())"
```

---

## Quickstart: Running an Experiment

Below is a minimal command to run an active learning loop on a built-in benchmark (e.g. `single_cell_cycle`), using our **JAX-based GP** model and **DiscoBAX** acquisition:

```bash
python run_pipeline.py \
  --dataset_name="single_cell_cycle" \
  --feature_set_name="achilles" \
  --model_name="jax_sparse_gp" \
  --acquisition_function_name="custom" \
  --acquisition_function_path="genedisco/active_learning_methods/acquisition_functions/disco_bax_two_stage_jax.py" \
  --acquisition_batch_size=64 \
  --num_active_learning_cycles=8
```

Explanation:

- **`--model_name="jax_sparse_gp"`**: Uses a custom JAX-based Gaussian Process model (GPJax) for training each cycle.  
- **`--acquisition_function_name="custom"` + `--acquisition_function_path="..."`**: Dynamically loads the two-stage DiscoBAX approach outlined in `disco_bax_two_stage_jax.py`.  
- **`--acquisition_batch_size=64`** and `--num_active_learning_cycles=8` specify how many new gene interventions are chosen each cycle, and how many cycles the pipeline will run.  

The pipeline logs progress to your console or specified output folder. On each cycle, the GP is retrained with newly “observed” data, and the DiscoBAX routine proposes the next batch of experiments. By the final cycle, you can examine a set of gene hits that are predicted to be **both impactful and diverse** in their modes of action.

---

## Current Status & Future Directions

- **Validation**: The approach has been tested on standard GeneDisco tasks, showing robust performance in discovering multiple pathways of interest. Early experiments indicate that two-stage DiscoBAX yields broader coverage of potential gene hits than a purely value-seeking approach.  
- **Next Steps**: Plans include integrating more advanced kernel structures (e.g. multi-task GPs) and an ensemble-based approach for modeling epistatic interactions. We also aim to expand the library of built-in benchmarks to cover more single-cell data sets and applied problems like synergy-based drug screening.  
- **Open to Collaboration**: We welcome feedback or collaborations from the computational biology community. If you have a specialized dataset or a new acquisition idea, feel free to open an issue or discuss merging contributions.

---

## Author and Contact

**GeneDiscoX** was developed by **Anass Bellachehab**, a researcher with a Ph.D. in computer science, and principal scientific computing engineer at Synchrotron Soleil radiation source in the Paris area in France. The project emerged out of consulting work for a biotech startup in longevity, aiming to automate gene discovery for age-related disease models. My broader research focuses on integrating Bayesian deep learning with high dimensional data to expedite combinatorial search problems in large problem spaces. Something very important in accelerator tuning physics which is what I currently do, but also in biomedical applications which is my topic of passion

If you’re interested in discussing advanced experimental design, neural surrogates for genomics, or collaborative projects, please feel free to reach out (see my GitHub profile)

---
