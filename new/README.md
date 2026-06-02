# DA3 gsplat Pipeline

This directory contains the current RGB-video-to-DA3-to-gsplat Colab pipeline.

## Files

- `create_da3_gsplat_colab_notebook.py`
  - Generates the base Colab notebook.
- `update_notebook_da3_gsplat_bridge.py`
  - Inserts the verified DA3 dataset adapter and gsplat trainer bridge.
- `da3_gsplat_colab_pipeline.ipynb`
  - Ready-to-upload Colab notebook.

## Regenerate The Notebook

Run both commands from this directory:

```powershell
py create_da3_gsplat_colab_notebook.py
py update_notebook_da3_gsplat_bridge.py
```

The generator intentionally writes `da3_gsplat_colab_pipeline.ipynb` into the
current directory. The bridge updater then inserts the DA3-to-gsplat adapter.

## Current Pipeline Order

```text
RGB video
-> candidate frame extraction
-> RGB keyframe filtering
-> initial DA3 inference
-> frame-level trajectory and cross-view depth filtering
-> refined DA3 inference
-> refined geometry validation
-> point-level multi-view filtering
-> wall-filtered and unfiltered point-cloud comparison outputs
-> conservative initialization point cloud
-> DA3-to-gsplat dataset conversion
-> gsplat training
-> Gaussian PLY export and ZIP packaging
```

