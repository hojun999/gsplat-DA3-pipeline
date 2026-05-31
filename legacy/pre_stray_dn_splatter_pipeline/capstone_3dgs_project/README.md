# Capstone 3DGS Project

Main execution notebook:

- `notebooks/01_main_splatfacto_pipeline.ipynb`

This main notebook runs the stable Nerfstudio path only:

1. RGB smartphone video input
2. `ns-process-data video` with COLMAP
3. `ns-train splatfacto`
4. `ns-export gaussian-splat`
5. Drive result packaging

Experiment notebooks are kept separate from the main pipeline.
