# Introduction

This repository is built and modified on top of [Evolutor](https://github.com/KhartesViewer/evolutor), which takes a 2D scroll slice as input and produces an undeformed version of the slice as output.

This repository is divided into two parts: one involves data processing, located in the Python scripts in the root directory, and the other part focuses on data visualization, located in the JavaScript in the `visualize` folder.

In the data processing section, I modified Evolutor to work with the Ome-Zarr format (for 2D scroll slices, the z-axis is 1). This allows each 2D chunk to be processed independently and integrates information across the hierarchical levels.

In the visualization section, you can interact with the processed Ome-Zarr data to understand the deformation process and compare the deformation differences across different Ome-Zarr hierarchies.

## Generate Ome-Zarr for 2D scroll slice

Khartes wrote a script called [scroll_to_ome.py](https://github.com/KhartesViewer/scroll2zarr/blob/main/scroll_to_ome.py), which can generate Ome-Zarr format for a TIFF stack image. I made some small modifications to allow it to generate Ome-Zarr data for a single image. Take scroll1 `02000.tif` as an example.
```python
python scroll_to_ome.py ./evol1/02000.tif ./evol1/scroll.zarr
```

## Notes

More details later ...