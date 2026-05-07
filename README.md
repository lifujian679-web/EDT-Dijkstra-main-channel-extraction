# EDT–Dijkstra Main-Channel Centerline Extraction

This repository contains Python code for extracting the main-channel centerline from multi-channel river imagery using an **Euclidean Distance Transform (EDT) + Dijkstra shortest-path** framework.

The workflow was prepared to support the reproducibility of a manuscript on intra-annual main-channel migration in the braided reach of the Lower Yellow River.

## Main functions

The script performs two major tasks:

1. **River water-body extraction**

   * mosaics Sentinel-2 band files
   * reprojects rasters to a common CRS
   * clips rasters to the study boundary
   * computes MNDWI from B03 and B11
   * extracts the main connected river water body
   * optionally removes the influence of water-related structures such as bridges
2. **Main-channel centerline extraction**

   * binarizes the extracted water mask
   * computes the squared Euclidean distance transform
   * extracts the skeleton of the binary river mask
   * builds an undirected graph from skeleton pixels
   * assigns edge weights using geometric distance divided by local EDT-derived width
   * uses Dijkstra shortest path to extract the widest through-going corridor as the main-channel centerline

## Repository contents

* `translated\_edt\_dijkstra\_code.py` — main Python script with English comments and messages
* `requirements.txt` — Python package dependencies
* `LICENSE` — MIT License
* `CITATION.cff` — suggested citation metadata
* `CODE\_AVAILABILITY.txt` — code availability statement
* `README.md` — this file

## Software environment

The script was written in Python and uses the following main libraries:

* rasterio
* fiona
* scipy
* scikit-image
* numpy
* opencv-python
* networkx
* geopandas
* shapely

## Required input data

The script expects the following inputs:

### 1\. Parent folder containing subfolders of satellite imagery

Each subfolder should correspond to one image date and contain Sentinel-2 JP2 files, including:

* `\*B03\*.jp2` — green band
* `\*B11\*.jp2` — shortwave infrared band

### 2\. Vector files

* `Bridge.shp` — shapefile of water-related structures (optional but used in the script)
* `Boundary.shp` — shapefile of the study area boundary

### 3\. User-defined local paths

The following variables must be updated before running the script:

* `deal\_path`
* `bridges\_shp\_path`
* `Boundary\_shp\_path`

## Output files

For each image subfolder, the script writes:

* `water\_mask\_<folder>.tif` — extracted river water mask
* `mndwi\_<folder>.tif` — MNDWI raster
* `centerline\_<folder>.shp` — extracted main-channel centerline
* `skeleton\_<folder>.shp` — skeleton graph edges
* `width\_transform\_<folder>.tif` — EDT-derived width raster

## Processing workflow

### Step 1. Band mosaicking and clipping

Band files are mosaicked when the study area spans multiple scenes. All rasters are reprojected to a common CRS and cropped to the study boundary.

### Step 2. MNDWI-based water extraction

The script reads B03 and B11, resamples B11 to match B03 when needed, computes MNDWI, and applies a fixed threshold of 0 to identify water.

### Step 3. Bridge correction

If a bridge shapefile is provided, pixels inside bridge polygons are set to NaN and filled using nearest valid neighboring values.

### Step 4. Largest connected water body

The script labels connected components and retains only the largest connected water body to remove lakes and isolated non-river water patches.

### Step 5. EDT and skeleton extraction

The cleaned river mask is binarized. The script then computes the squared Euclidean distance transform and extracts a 1-pixel skeleton.

### Step 6. Graph construction

Each skeleton pixel becomes one graph node. Neighboring skeleton pixels are connected using 8-neighborhood connectivity. Edge weights are defined as geometric distance divided by average local width, so wider corridors have lower cost.

### Step 7. Source–target selection

Skeleton endpoints are identified automatically. The leftmost endpoint is used as the source and the rightmost endpoint as the target. This assumes the river generally flows from left to right in the raster.

### Step 8. Dijkstra path extraction

The shortest path between source and target is extracted with NetworkX. Because edge weights are inversely related to local width, the minimum-cost path follows the dominant wide channel corridor.

## How to run

1. Install the required packages:

```bash
   pip install -r requirements.txt
   ```

2. Edit the file paths in `translated\_edt\_dijkstra\_code.py`:

   * `bridges\_shp\_path`
   * `Boundary\_shp\_path`
   * `deal\_path`
3. Run the script:

```bash
   python translated\_edt\_dijkstra\_code.py
   ```

## Important notes

* The current source–target selection assumes the river is oriented approximately from left to right in the image.
* The script is intended as a reproducible implementation of the EDT–Dijkstra extraction workflow and may need adaptation for other study areas.
* Example input data are **not included** in this package. Users should prepare their own Sentinel-2 imagery and shapefiles.

## Suggested manuscript statement

A short code availability statement is provided in `CODE\_AVAILABILITY.txt`.

