from rasterio.mask import mask
from rasterio.features import rasterize
import fiona
from rasterio.merge import merge
from rasterio.warp import calculate_default_transform, reproject, Resampling
import fiona.transform
from scipy.ndimage import distance_transform_edt
from skimage.measure import label, regionprops
import rasterio
import numpy as np
import cv2
from skimage.morphology import skeletonize
import networkx as nx
import geopandas as gpd
from shapely.geometry import LineString
import glob
import os


# Read all subfolders in the parent directory for batch processing
def get_all_folders(deal_path):
    items = os.listdir(deal_path)
    folders = [item for item in items if os.path.isdir(os.path.join(deal_path, item))]  # Filter all folders
    return folders


# Reproject raster to a unified coordinate reference system
def reproject_to_match(src_path, dst_crs, temp_folder):
    with rasterio.open(src_path) as src:
        if src.crs == dst_crs:  # Skip reprojection if CRS already matches the target CRS
            return src_path
        else:
            transform, width, height = calculate_default_transform(
                src.crs, dst_crs, src.width, src.height, *src.bounds
            )  # Calculate transform, width, and height under the target CRS

            # Build temporary output path
            dst_path = os.path.join(temp_folder, os.path.basename(src_path).replace(".jp2", "_reproj.tif"))
            if os.path.exists(dst_path):  # Remove the existing file first if it already exists
                os.remove(dst_path)

            # Coordinate parameters
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': dst_crs,
                'transform': transform,
                'width': width,
                'height': height,
                'driver': 'GTiff'
            })

            # Reproject to the unified CRS
            with rasterio.open(dst_path, 'w', **kwargs) as dst:
                for i in range(1, src.count + 1):
                    reproject(
                        source=rasterio.band(src, i),
                        destination=rasterio.band(dst, i),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=dst_crs,
                        resampling=Resampling.bilinear
                    )

    return dst_path


# Mosaic and crop band files
def merge_bands_with_roi_crop(jp2_files, input_folder, output_path, roi_shp):
    """
    jp2_files: band files to be processed
    roi_shp: shapefile of the study area used to remove data outside the region of interest
             and improve computational efficiency
    """
    # Check whether files exist
    if len(jp2_files) == 0:
        raise FileNotFoundError("No .jp2 files found")

    # Create a temporary folder for mosaicked outputs
    temp_folder = os.path.join(input_folder, "temp_reproj")
    os.makedirs(temp_folder, exist_ok=True)

    # Use the first file as the reference CRS
    with rasterio.open(jp2_files[1]) as ref_src:
        ref_crs = ref_src.crs

    # Reproject all files to the same CRS
    reprojected_paths = [reproject_to_match(f, ref_crs, temp_folder) for f in jp2_files]
    src_files_to_mosaic = [rasterio.open(p) for p in reprojected_paths]

    # Mosaic bands
    mosaic, out_transform = merge(src_files_to_mosaic)
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

    if_crop = 'YES'
    if if_crop == 'YES':
        # Crop to the region of interest (ROI)
        with fiona.open(roi_shp, "r") as shapefile:
            shp_crs = shapefile.crs
            if shp_crs != ref_crs.to_dict():
                # Reproject vector geometry to the raster CRS
                shapes = [fiona.transform.transform_geom(shp_crs, ref_crs.to_dict(), feature["geometry"])
                          for feature in shapefile]
            else:
                shapes = [feature["geometry"] for feature in shapefile]

        # Write mosaic to memory first, then crop
        with rasterio.io.MemoryFile() as memfile:
            with memfile.open(**out_meta) as dataset:
                dataset.write(mosaic)
                # Crop image (crop=True keeps the cropped boundary size)
                out_image, out_transform = mask(dataset, shapes, crop=True)
                out_meta.update({
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform
                })
    else:
        pass

    # Write output file
    with rasterio.open(output_path, "w", **out_meta) as dest:
        dest.write(out_image)

    print(f"✅ Mosaic and crop completed: {output_path}")
    return


# Unify band resolution when input bands have different resolutions
def resampled(b3, b3_transform, b3_crs, b11, b11_transform, b11_crs):
    """
    Example for Sentinel imagery:
    resample the 20 m SWIR band (B11) to 10 m
    """
    if b3_transform != b11_transform:  # Resample B11 if resolutions differ (e.g., 10 m vs 20 m)
        # Create target array with the same size as B03
        b11_resampled = np.empty_like(b3, dtype=np.float32)
        # Perform resampling using bilinear interpolation
        reproject(
            source=b11,
            destination=b11_resampled,
            src_transform=b11_transform,
            src_crs=b11_crs,
            dst_transform=b3_transform,
            dst_crs=b3_crs,
            resampling=Resampling.bilinear
        )

        print("Resolution has been unified")
        return b11_resampled


# Calculate MNDWI and generate an initial water mask using thresholding
def detect_mndwi_water_mask(merge_b3_path, merge_b11_path):
    """
    merge_b3_path: path to the green band
    merge_b11_path: path to the shortwave infrared band
    """
    # Read green band
    with rasterio.open(merge_b3_path) as red_src:
        b3 = red_src.read(1)
        b3_transform = red_src.transform
        b3_crs = red_src.crs

    # Read shortwave infrared band
    with rasterio.open(merge_b11_path) as nir_src:
        b11 = nir_src.read(1)
        b11_transform = nir_src.transform
        b11_crs = nir_src.crs

    # Unify resolution when necessary
    # Example: Sentinel provides B03 at 10 m and B11 at 20 m.
    # Resampling is required before computing MNDWI.
    if_resampled = "YES"
    if if_resampled == "YES":
        b11 = resampled(b3, b3_transform, b3_crs, b11, b11_transform, b11_crs)
    else:
        pass

    # Compute MNDWI
    mndwi = (b3 - b11) / (b3 + b11 + 1e-10)

    # Identify water body using thresholding
    mndwi_threshold = 0
    water_mask = (mndwi > mndwi_threshold).astype(np.uint8)

    print("MNDWI has been calculated")
    print("Water area has been extracted")
    return mndwi, water_mask, b3_transform, b3_crs


# Remove the influence of water-related structures
def detect_bridge(water_mask, bridges_shp_path, transform):
    """
    Principle:
    replace bridge pixels with nearby values (water or non-water, represented as 1 or 0 in the mask)

    1. Create a bridge mask
    2. Identify bridge locations
    3. Set corresponding pixels in water_mask to NaN
    4. Repair gaps using nearest-neighbor interpolation
    """
    # Read bridge vector data
    bridge_gdf = gpd.read_file(bridges_shp_path)
    bridge_shapes = [geom.__geo_interface__ for geom in bridge_gdf.geometry]

    # Rasterize bridge vectors to generate a bridge mask
    bridge_mask = rasterize(
        bridge_shapes,
        out_shape=water_mask.shape,
        transform=transform,
        fill=0,  # Non-bridge area = 0
        all_touched=True
    ).astype('uint8')

    # Set bridge area to NaN before interpolation
    water_mask = water_mask.astype(float)
    water_mask[bridge_mask == 1] = np.nan

    # Create valid-value mask (non-NaN)
    valid_mask = ~np.isnan(water_mask)

    # Compute distance to the nearest valid pixel and return nearest indices
    distance, indices = distance_transform_edt(valid_mask == 0, return_indices=True)

    # Replace NaN pixels with nearest valid values
    filled_water = water_mask.copy()
    filled_water[np.isnan(water_mask)] = water_mask[tuple(i[np.isnan(water_mask)] for i in indices)]

    print("Water-related structure area has been filled using nearest valid pixels")
    return filled_water


# Save MNDWI raster
def save_mndwi(mndwi, transform, crs):
    mndwi_file = f"{input_dir}" + '\\' + f"mndwi_{i}.tif"

    with rasterio.open(
        mndwi_file, 'w',
        driver='GTiff',
        height=mndwi.shape[0],
        width=mndwi.shape[1],
        count=1,
        dtype=mndwi.dtype,
        crs=crs,
        transform=transform
    ) as dst:
        dst.write(mndwi, 1)

    print("MNDWI has been saved")
    return


# Save water mask raster
def save_water_mask(water_mask, water_mask_file, transform, crs):
    water_mask = water_mask.astype(np.uint8)  # rasterio does not support boolean type directly
    nodata_value = 0  # Set NoData to 0 to remove non-water area

    with rasterio.open(
        water_mask_file, 'w',
        driver='GTiff',
        height=water_mask.shape[0],
        width=water_mask.shape[1],
        count=1,
        dtype=water_mask.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata_value
    ) as dst:
        dst.write(water_mask, 1)

    print("Water mask has been saved")
    return


# Save centerline shapefile
def save_centerline(path, i, input_dir, transform, crs):
    output_shp = input_dir + '\\' + "centerline_{}.shp".format(i)

    coords = [rasterio.transform.xy(transform, y, x) for y, x in path]
    line = LineString(coords)
    gdf = gpd.GeoDataFrame({'geometry': [line]}, crs=crs)
    gdf.to_file(output_shp)

    print("Main-channel centerline has been saved")
    return


# Save skeleton shapefile
def save_skeleton(G, i, input_dir, transform, crs):
    output_shp = input_dir + '\\' + "skeleton_{}.shp".format(i)

    lines = []
    for u, v in G.edges:  # Convert skeleton edges into LineString
        y1, x1 = u
        y2, x2 = v
        x1c, y1c = rasterio.transform.xy(transform, y1, x1, offset='center')
        x2c, y2c = rasterio.transform.xy(transform, y2, x2, offset='center')
        line = LineString([(x1c, y1c), (x2c, y2c)])
        lines.append(line)
    gdf = gpd.GeoDataFrame(geometry=lines, crs=crs)  # Save as Shapefile
    gdf.to_file(output_shp)

    print("Skeleton has been saved")
    return


# Save width heat map
def save_width_transform(dist_transform, i, input_dir, transform, crs):
    output_tif = input_dir + '\\' + "width_transform_{}.tif".format(i)

    with rasterio.open(
        output_tif,
        'w',
        driver='GTiff',
        height=dist_transform.shape[0],
        width=dist_transform.shape[1],
        count=1,
        dtype=dist_transform.dtype,
        crs=crs,
        transform=transform,
        nodata=0
    ) as dst:
        dst.write(dist_transform, 1)

    print("Width heat map has been saved")
    return


# Main program
if __name__ == "__main__":
    """
    The workflow includes two main parts:
    (1) extract river water body
    (2) identify the main-channel centerline based on the extracted water body
    """
    # Shapefiles used in river-water extraction; replace with actual paths
    bridges_shp_path = "Bridge.shp"     # Water-related structures (e.g., bridges), used to remove their influence
    Boundary_shp_path = "Boundary.shp"  # Study area boundary
    # print(bridges_shp_path, Boundary_shp_path)

    # Parent folder containing subfolders to process; replace with actual path
    deal_path = r'deal'

    # Get all subfolders for batch processing
    folders_list = get_all_folders(deal_path)

    # Batch-process satellite imagery in each subfolder
    for i in folders_list:
        print("-------------------------------------------------")
        print(f'Start processing satellite imagery in "{i}"')
        input_dir = deal_path + '\\' + i

        ### (1) River water-body extraction
        # Band file paths
        input_b3_path = sorted(glob.glob(os.path.join(input_dir, '*B03*.jp2')))   # Green band, e.g., 10 m for Sentinel
        input_b11_path = sorted(glob.glob(os.path.join(input_dir, '*B11*.jp2')))  # SWIR band, e.g., 20 m for Sentinel

        # Mosaic bands when the study area spans multiple scenes
        merged_path = input_dir + r"\merged"
        if not os.path.exists(merged_path):
            os.makedirs(merged_path)

        # Decide whether band mosaicking is required
        if_merge = 'YES'
        if if_merge == 'YES':
            # Mosaic green band
            merge_bands_with_roi_crop(
                jp2_files=input_b3_path,
                input_folder=input_dir,
                output_path=merged_path + r"\B03_merged.tif",
                roi_shp=Boundary_shp_path
            )

            # Mosaic SWIR band
            merge_bands_with_roi_crop(
                jp2_files=input_b11_path,
                input_folder=input_dir,
                output_path=merged_path + r"\B11_merged.tif",
                roi_shp=Boundary_shp_path
            )

            # Paths of mosaicked bands
            merge_b3_path = input_dir + r"\merged\B03_merged.tif"
            merge_b11_path = input_dir + r"\merged\B11_merged.tif"
        else:
            merge_b3_path = input_b3_path[0]
            merge_b11_path = input_b11_path[0]

        # Compute MNDWI and identify water mask
        mndwi, water_mask, transform, crs = detect_mndwi_water_mask(merge_b3_path, merge_b11_path)

        # Decide whether to remove the influence of water-related structures
        if_detect = 'YES'
        if if_detect == 'YES':
            water_mask = detect_bridge(water_mask, bridges_shp_path, transform)
        else:
            pass

        # Remove non-river water bodies such as lakes
        labeled = label(water_mask)
        regions = regionprops(labeled)
        max_region = max(regions, key=lambda r: r.area)
        water_mask = (labeled == max_region.label).astype(np.uint8)  # Keep the largest connected water body

        # Save river-water TIFF for later centerline extraction
        water_mask_file = f"{input_dir}" + '\\' + f"water_mask_{i}.tif"
        save_water_mask(water_mask, water_mask_file, transform, crs)

        ### (2) Main-channel centerline extraction
        # Read water-mask raster
        with rasterio.open(water_mask_file) as src:
            raster_array = src.read(1)
        print("Water-mask raster has been loaded")

        # Binarize water mask
        _, binary_image = cv2.threshold(raster_array, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary = binary_image.astype(np.uint8)
        print("Binary conversion completed")

        # Squared Euclidean distance transform
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        dist_transform = dist_transform ** 2
        print("Distance transform completed")

        # Skeleton extraction
        skeleton = skeletonize(binary).astype(np.uint8)
        print("Skeleton extraction completed")

        # Map width values onto skeleton
        skeleton_widths = dist_transform * skeleton
        print("Skeleton-width mapping completed")

        # Build skeleton graph
        height, width = skeleton.shape
        G = nx.Graph()
        length_list = []
        ys, xs = np.where(skeleton > 0)
        for y, x in zip(ys, xs):  # Build an undirected graph from skeleton pixels
            G.add_node((y, x))
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx_ = y + dy, x + dx
                    if 0 <= ny < height and 0 <= nx_ < width and skeleton[ny, nx_] > 0:
                        w1, w2 = skeleton_widths[y, x], skeleton_widths[ny, nx_]
                        if w1 > 0 and w2 > 0:
                            avg_width = (w1 + w2) / 2
                            length = ((y - ny) ** 2 + (x - nx_) ** 2) ** 0.5
                            length_list.append(length)
                            G.add_edge(
                                (y, x),
                                (ny, nx_),
                                weight=length / avg_width
                            )  # Wider paths receive lower cost
        print("Skeleton graph construction completed")

        # Identify all skeleton endpoints for automatic source/target selection
        endpoints = []
        for y, x in zip(ys, xs):
            neighbors = [
                (y + dy, x + dx)
                for dy in [-1, 0, 1]
                for dx in [-1, 0, 1]
                if not (dy == 0 and dx == 0) and 0 <= y + dy < height and 0 <= x + dx < width
            ]
            count = sum(1 for ny, nx_ in neighbors if skeleton[ny, nx_] > 0)
            if count == 1:
                endpoints.append((y, x))
        print("Skeleton endpoints identified")

        # Automatically select start and end points
        # Example assumes the river generally flows from left to right
        endpoints = np.array(endpoints)
        start = tuple(endpoints[np.argmin(endpoints[:, 1])])  # Leftmost endpoint
        end = tuple(endpoints[np.argmax(endpoints[:, 1])])    # Rightmost endpoint
        print("Start and end nodes selected automatically")

        # Extract the main-channel centerline (widest path)
        path = nx.shortest_path(
            G,
            source=start,
            target=end,
            weight='weight'
        )  # Minimum-cost path is biased toward wider channel corridors
        print("Widest path extraction completed")

        # Save outputs
        save_centerline(path, i, input_dir, transform, crs)          # Save centerline
        save_mndwi(mndwi, transform, crs)                            # Save MNDWI (optional)
        save_skeleton(G, i, input_dir, transform, crs)               # Save skeleton (optional)
        save_width_transform(dist_transform, i, input_dir, transform, crs)  # Save width heat map (optional)
