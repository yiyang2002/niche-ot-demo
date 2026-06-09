# Niche OT Remodeling

Niche OT Remodeling is an analysis workflow for aligning condition-specific
niche cluster libraries with an unbalanced optimal-transport model. It compares
reference and perturbed tissue-state libraries using cell-type composition
features and cluster mass, then summarizes source expansion, reduction, branch
structure, target residual mass, and composition remodeling across a penalty
path.

## Repository Layout

- `niche_ot_demo.ipynb`: transport-remodeling analysis notebook.
- `transport_remodeling.py`: helper module for harmonization, composition-cost
  construction, unbalanced transport, event summaries, and plotting.
- `demo_data/healthy_cluster_library.csv`: synthetic reference niche library.
- `demo_data/disease_cluster_library.csv`: synthetic perturbed-condition niche
  library.
- `requirements.txt`: minimal Python dependencies for inspecting or running the
  notebook.

## Input Schema

The included CSV files use the cluster-level schema expected by the analysis:

- `cluster_id`: niche cluster identifier.
- `N_Cells`: cluster mass used by the transport helper.
- `tissue_mass`: normalized tissue-mass proxy included for readability.
- `N_Slides`: recurrence count across synthetic samples.
- `Niche_Signature`: short text label for the cluster state.
- `Top_Enriched_Cells (log2FC)`: readable cell-type enrichment summary.
- `Comp_*`: cell-type proportions used to build the composition-distance cost.

## Workflow

The notebook follows the transport-remodeling workflow:

1. Configure reference and perturbed cluster-library inputs.
2. Harmonize cell-type labels and load cluster metrics.
3. Build aligned composition matrices, masses, and pairwise costs.
4. Solve the default unbalanced OT model.
5. Summarize source and target remodeling events.
6. Inspect penalty-path behavior and a source-regularized diagnostic run.
7. Review selected-source composition remodeling across the sweep.

The bundled data are synthetic and are intended to make the complete notebook
structure inspectable without external datasets.
