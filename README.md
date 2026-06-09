# Niche OT Demo

This repository is a standalone demonstration mirror of the optimal-transport (OT) niche-library remodeling analysis from a larger spatial transcriptomics research codebase.

The goal is to show the analysis shape without publishing the original project, real datasets, generated outputs, or repository history. The demo aligns two synthetic niche cluster libraries representing different conditions. Each cluster has cell-type composition features (`Comp_*`) and a tissue-mass proxy (`tissue_mass`, mirrored by `N_Cells` for compatibility with the current helper code).

## Contents

- `transport_remodeling.py`: current OT helper module copied from the research project.
- `niche_ot_demo.ipynb`: one demo notebook derived from the canonical current OT notebook, with outputs stripped and paths pointed at `demo_data/`.
- `demo_data/healthy_cluster_library.csv`: synthetic healthy/reference niche cluster library.
- `demo_data/disease_cluster_library.csv`: synthetic disease/target niche cluster library.

## Demo Data Schema

The CSV inputs use the same cluster-level fields expected by the helper:

- `cluster_id`: niche cluster identifier.
- `N_Cells`: cluster mass used by the current helper code.
- `tissue_mass`: normalized synthetic tissue mass for reader-facing clarity.
- `N_Slides`: synthetic recurrence count.
- `Niche_Signature` and `Top_Enriched_Cells (log2FC)`: readable labels.
- `Comp_*`: cell-type proportions used to compute composition distance.

This repository is for demonstration and inspection. It is not the full research pipeline and does not include the private source datasets.
