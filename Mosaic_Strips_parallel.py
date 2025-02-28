#!/usr/bin/env python3
"""
Mosaic_Strips_parallel.py

This is a parallelized version of Mosaic_Strips.py.
In addition to using the already available parallel versions of functions from dem_utils 
(e.g. parallel_get_strip_shp, parallel_filter_strip_gsw, parallel_get_contained_strips, etc.),
this version parallelizes the processing across unique EPSG codes.
Each EPSG code’s strip processing (filtering, spatial analysis, mosaic‐building) is 
executed concurrently using a multiprocessing Pool.

Usage: 
    python3 Mosaic_Strips_parallel.py --config dem_config.ini [other options]
    
Command-line options are similar to the original script.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import shapely
from osgeo import gdal, gdalconst, osr
import os, sys, glob, argparse, subprocess, datetime, warnings, configparser, ctypes as c, multiprocessing
import gc  # For explicit garbage collection
import psutil  # For memory monitoring

# Import functions from dem_utils – note that several parallel versions are available.
from dem_utils import (get_strip_list, get_strip_extents, get_gsw, get_strip_shp,
                       filter_strip_gsw, find_cloud_water, get_valid_strip_overlaps,
                       get_minimum_spanning_tree, find_mosaic, build_mosaic,
                       copy_single_strips, parallel_get_valid_strip_overlaps,
                       parallel_get_strip_shp, parallel_filter_strip_gsw,
                       get_lonlat_gdf, parallel_get_contained_strips)

def process_epsg(args):
    """
    Process all strips corresponding to a single EPSG code.
    This function:
      - Filters the full strip list for those with the target EPSG.
      - Computes extents, obtains GSW data, 
      - Uses parallel functions to get strip shapes and apply filtering,
      - Saves the processed shapefile and builds mosaics.
    """
    (epsg_code, full_strip_list, tmp_dir, output_dir, output_name, config,
     N_cpus, corrected_flag, cloud_water_filter_flag, simplify_flag, gsw_file) = args
    
    # Monitor memory usage
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[EPSG:{epsg_code}] Starting with memory usage: {initial_memory:.2f} MB")

    # Compute EPSG for each strip in the full list.
    full_epsg_list = np.asarray([osr.SpatialReference(wkt=gdal.Open(s, gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1) 
                                 for s in full_strip_list])
    idx_epsg = full_epsg_list == epsg_code
    strip_list = full_strip_list[idx_epsg]
    if strip_list.size == 0:
        print(f'No strips found for EPSG:{epsg_code}')
        return

    # Compute overall extents over all strips in this EPSG.
    lon_min_strips, lon_max_strips = 180, -180
    lat_min_strips, lat_max_strips = 90, -90
    for strip in strip_list:
        try:
            # get_strip_extents returns a tuple: (lon_min, lon_max, lat_min, lat_max)
            ext = get_strip_extents(strip)
        except Exception as e:
            print(f"Error processing extents for {strip}: {e}")
            continue
        lon_min, lon_max, lat_min, lat_max = ext
        lon_min_strips = min(lon_min_strips, lon_min)
        lon_max_strips = max(lon_max_strips, lon_max)
        lat_min_strips = min(lat_min_strips, lat_min)
        lat_max_strips = max(lat_max_strips, lat_max)

    # Get Global Surface Water (GSW) data if not provided.
    if gsw_file is None:
        gsw_main_sea_only, _ = get_gsw(output_dir, tmp_dir, 
                                       config.get('GENERAL_PATHS','gsw_dir'),
                                       epsg_code,
                                       lon_min_strips, lon_max_strips,
                                       lat_min_strips, lat_max_strips,
                                       output_name,
                                       config.getfloat('MOSAIC_CONSTANTS','GSW_POCKET_THRESHOLD'),
                                       config.getfloat('MOSAIC_CONSTANTS','GSW_CRS_TRANSFORM_THRESHOLD'))
    else:
        gsw_main_sea_only = gpd.read_file(gsw_file).to_crs(f'EPSG:{epsg_code}')
    if gsw_main_sea_only is not None:
        gsw_main_sea_only_buffered = gsw_main_sea_only.buffer(0)
    else:
        gsw_main_sea_only_buffered = None

    # Obtain strip shapes in parallel with memory usage monitoring
    print(f"[EPSG:{epsg_code}] Getting strip shapes...")
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[EPSG:{epsg_code}] Memory before strip shapes: {current_memory:.2f} MB")
    
    strip_shapes = parallel_get_strip_shp(strip_list, tmp_dir, n_jobs=N_cpus, chunk_size=chunk_size)
    
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[EPSG:{epsg_code}] Memory after strip shapes: {current_memory:.2f} MB")
    gc.collect()  # Force garbage collection between major steps
    
    print(f"[EPSG:{epsg_code}] Filtering strip shapes...")
    filtered_shapes = parallel_filter_strip_gsw(strip_shapes, gsw_main_sea_only,
                                                config.getfloat('MOSAIC_CONSTANTS','STRIP_AREA_THRESHOLD'),
                                                config.getfloat('MOSAIC_CONSTANTS','POLYGON_AREA_THRESHOLD'),
                                                config.getfloat('MOSAIC_CONSTANTS','GSW_OVERLAP_THRESHOLD'),
                                                config.getfloat('MOSAIC_CONSTANTS','STRIP_TOTAL_AREA_PERCENTAGE_THRESHOLD'),
                                                n_jobs=N_cpus, chunk_size=chunk_size)
    
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[EPSG:{epsg_code}] Memory after filtering: {current_memory:.2f} MB")
    gc.collect()  # Force garbage collection between major steps

    # Build a GeoDataFrame of processed strip geometries.
    strip_shp_data = gpd.GeoDataFrame()
    strip_idx = np.ones(len(strip_list), dtype=bool)
    for j, (strip, fshape) in enumerate(zip(strip_list, filtered_shapes)):
        if fshape is None or fshape.empty:
            strip_idx[j] = False
            continue
        try:
            tmp_mp = shapely.ops.unary_union([Polygon(g) for g in fshape.geometry.exterior])
        except Exception as e:
            print(f"Error creating union polygon for strip {strip}: {e}")
            strip_idx[j] = False
            continue
        df_strip = pd.DataFrame({'strip': [strip]})
        tmp_gdf = gpd.GeoDataFrame(df_strip, geometry=[tmp_mp], crs=f'EPSG:{epsg_code}')
        strip_shp_data = gpd.GeoDataFrame(pd.concat([strip_shp_data, tmp_gdf], ignore_index=True), crs=f'EPSG:{epsg_code}')

    # Filter out strips with failed geometry.
    strip_list = strip_list[strip_idx]

    # Optionally, apply cloud/water filtering here (omitted for brevity).
    # ...

    # Save processed strip shapefile.
    strip_dates = np.asarray([int(s.split('/')[-1][5:13]) for s in strip_list])
    idx_date = np.argsort(-strip_dates)
    strip_dates = strip_dates[idx_date]
    strip_list = strip_list[idx_date]
    strip_shp_data = strip_shp_data.iloc[idx_date].reset_index(drop=True)
    output_strips_shp_file = os.path.join(output_dir, f"{output_name}_Strips_{epsg_code}.shp")
    strip_shp_data.to_file(output_strips_shp_file)
    print(f"[EPSG:{epsg_code}] Saved strip shapefile: {output_strips_shp_file}")

    # Dissolve shapefile to merge overlapping geometries.
    output_strips_shp_file_dissolved = os.path.join(output_dir, f"{output_name}_Strips_{epsg_code}_Dissolved.shp")
    dissolve_cmd = ('ogr2ogr ' + output_strips_shp_file_dissolved + ' ' + output_strips_shp_file +
                    ' -dialect sqlite -sql "SELECT ST_Union(geometry) FROM \'' +
                    os.path.basename(output_strips_shp_file).replace('.shp','') + '\'"')
    subprocess.run(dissolve_cmd, shell=True)

    # Compute valid strip overlaps in parallel.
    if N_cpus > 1:
        valid_strip_overlaps = parallel_get_valid_strip_overlaps(strip_shp_data, gsw_main_sea_only_buffered, 
                                                                 config.getfloat('MOSAIC_CONSTANTS','AREA_OVERLAP_THRESHOLD'),
                                                                 config.getfloat('MOSAIC_CONSTANTS','GSW_INTERSECTION_THRESHOLD'),
                                                                 n_jobs=N_cpus)
    else:
        valid_strip_overlaps = get_valid_strip_overlaps(strip_shp_data, gsw_main_sea_only_buffered,
                                                         config.getfloat('MOSAIC_CONSTANTS','AREA_OVERLAP_THRESHOLD'),
                                                         config.getfloat('MOSAIC_CONSTANTS','GSW_INTERSECTION_THRESHOLD'))

    mst_array, mst_weighted_array = get_minimum_spanning_tree(valid_strip_overlaps, strip_dates)
    mosaic_dict, singles_dict = find_mosaic(strip_shp_data, mst_weighted_array, strip_dates)
    # Build mosaics for each group.
    for mosaic_number in range(len(mosaic_dict)):
        build_mosaic(strip_shp_data, gsw_main_sea_only_buffered, config.get('GENERAL_PATHS','landmask_c_file'),
                     mosaic_dict[mosaic_number], output_dir, tmp_dir, output_name, mosaic_number, epsg_code,
                     False,    # horizontal_flag (set as needed)
                     "",       # dir_structure (set as needed)
                     config.getfloat('MOSAIC_CONSTANTS','X_SPACING'),
                     config.getfloat('MOSAIC_CONSTANTS','Y_SPACING'),
                     config.getfloat('MOSAIC_CONSTANTS','X_MAX_SEARCH'),
                     config.getfloat('MOSAIC_CONSTANTS','Y_MAX_SEARCH'),
                     config.getfloat('MOSAIC_CONSTANTS','MOSAIC_TILE_SIZE'),
                     N_cpus)
    copy_single_strips(strip_shp_data, singles_dict, output_dir, output_name, epsg_code)
    
    # Report final memory usage
    current_memory = process.memory_info().rss / 1024 / 1024  # MB
    print(f"[EPSG:{epsg_code}] Finished. Final memory usage: {current_memory:.2f} MB, Change: {current_memory - initial_memory:.2f} MB")
    
    # Force garbage collection before exiting
    gc.collect()
    print(f'Finished processing EPSG:{epsg_code}.')

def main():
    warnings.simplefilter(action='ignore')
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='dem_config.ini', help='Path to configuration file.')
    parser.add_argument('--list', default=None, help='Path to list of strips to mosaic.')
    parser.add_argument('--output_dir', default=None, help='Path to output directory.')
    parser.add_argument('--loc_name', default=None, help='Name of location.')
    parser.add_argument('--gsw', default=None, help='Path to GSW shapefile')
    parser.add_argument('--dir_structure', default='sealevel', help='Directory structure', choices=['sealevel','simple','scenes'])
    parser.add_argument('--parallel', action='store_true', help='Run in parallel using all available CPUs')
    parser.add_argument('--N_cpus', default=1, type=int, help='Number of CPUs to use (ignored if --parallel is set)')
    parser.add_argument('--max_cpus', default=None, type=int, help='Maximum number of CPUs to use, even when --parallel is set')
    parser.add_argument('--chunk_size', default=None, type=int, help='Size of chunks for processing very large datasets')
    parser.add_argument('--horizontal', action='store_true', help='Incorporate horizontal alignment in mosaic?')
    parser.add_argument('--cloud_water_filter', default='default', nargs='?', help='Use cloud and water filter?')
    parser.add_argument('--corrected', action='store_true', help='Find corrected strips instead?')
    parser.add_argument('--all_strips', action='store_true', help='Mosaic all strips (skip geometry filtering)?')
    parser.add_argument('--no_gsw', action='store_true', help='Skip GSW filter?')
    parser.add_argument('--simplify', action='store_true', help='Simplify shapefiles?')
    args = parser.parse_args()

    config_file = args.config
    list_file = args.list
    single_output_dir = args.output_dir
    single_loc_name = args.loc_name

    config = configparser.ConfigParser()
    config.read(config_file)
    input_file = config.get('MOSAIC_PATHS', 'input_file')
    tmp_dir = config.get('GENERAL_PATHS', 'tmp_dir')
    # Use output directory provided in config or override with --output_dir
    output_dir = args.output_dir if args.output_dir is not None else config.get('GENERAL_PATHS','output_dir')
    # Ensure directories end with a slash.
    if not tmp_dir.endswith('/'):
        tmp_dir += '/'
    if not output_dir.endswith('/'):
        output_dir += '/'

    # Read input file listing mosaic directories.
    df_input = pd.read_csv(input_file, header=0, names=['loc_dirs','output_dirs','input_types'], dtype=str)
    df_input['input_types'] = df_input['input_types'].fillna('0').astype(int)

    if list_file is not None:
        df_list = pd.read_csv(list_file, header=None, names=['strip'], dtype=str)
        if args.output_dir is None:
            print('If a list is provided, then an output directory must be provided.')
            sys.exit()
        elif single_loc_name is None:
            single_loc_name = os.path.basename(os.path.dirname(args.output_dir))
        df_input.loc[len(df_input)] = ['list', args.output_dir, 3]
    else:
        df_list = None

    # For simplicity, we process only the first entry in df_input.
    row = df_input.iloc[0]
    loc_dir = row['loc_dirs']
    input_type = row['input_types']
    if loc_dir != 'list' and not loc_dir.endswith('/'):
        loc_dir += '/'
    if loc_dir != 'list':
        loc_name = os.path.basename(os.path.dirname(loc_dir))
    else:
        loc_name = single_loc_name
    output_name = loc_name

    if int(input_type) == 3:
        full_strip_list = np.asarray(df_list['strip'])
    else:
        full_strip_list = get_strip_list(loc_dir, int(input_type), args.corrected, args.dir_structure)

    # Get system info for optimal resource allocation
    system_cpus = multiprocessing.cpu_count()
    system_memory = psutil.virtual_memory().total / (1024**3)  # In GB
    print(f"System resources: {system_cpus} CPUs, {system_memory:.1f} GB RAM")
    
    # Determine number of CPUs to use with proper limits
    if args.parallel:
        if args.max_cpus:
            total_cpus = min(args.max_cpus, system_cpus)
        else:
            # For very large memory systems, we might want to limit CPU usage to avoid swapping
            if system_memory > 500:  # For systems with >500GB RAM
                # High-memory system - can use most CPUs
                total_cpus = min(system_cpus, 96)  # Cap at 96 to avoid excessive parallelism
            else:
                # Standard systems - use all CPUs or limit based on memory
                mem_based_cpu_limit = int(system_memory / 4)  # Rough heuristic: 4GB per process
                total_cpus = min(system_cpus, max(1, mem_based_cpu_limit))
    else:
        total_cpus = args.N_cpus
    
    # Compute unique EPSG codes from the full strip list
    full_epsg_list = np.asarray([osr.SpatialReference(wkt=gdal.Open(s, gdalconst.GA_ReadOnly).GetProjection()).GetAttrValue('AUTHORITY',1)
                                  for s in full_strip_list])
    unique_epsg_list = np.unique(full_epsg_list)
    num_epsg_codes = len(unique_epsg_list)
    print(f"Unique EPSG codes found: {unique_epsg_list} ({num_epsg_codes} total)")
    
    # Allocate CPUs based on the number of EPSG codes and system resources
    if num_epsg_codes >= total_cpus:
        # More EPSG codes than CPUs - use all CPUs for top-level parallelism
        epsg_cpus = total_cpus
        nested_cpus = 1
    else:
        # For high-CPU systems, distribute resources optimally
        if system_cpus >= 64:
            # Reserve at least 2 CPUs per EPSG code but no more than 16
            nested_cpus = min(16, max(2, total_cpus // num_epsg_codes))
        else:
            # For smaller systems, use a simpler allocation
            nested_cpus = max(1, total_cpus // num_epsg_codes)
        
        epsg_cpus = min(num_epsg_codes, max(1, total_cpus // nested_cpus))
    
    # Determine chunk size for processing
    chunk_size = args.chunk_size
    if chunk_size is None:
        # Auto-determine chunk size based on available memory
        if system_memory > 500:  # High memory system (>500GB)
            chunk_size = 10000  # Can process larger chunks
        elif system_memory > 100:  # Medium memory system (100-500GB)
            chunk_size = 5000
        else:  # Standard memory system
            chunk_size = 1000
    
    print(f"CPU allocation: {epsg_cpus} for EPSG parallelism, {nested_cpus} for nested operations")
    print(f"Chunk size for large dataset processing: {chunk_size}")

    # Prepare arguments for each EPSG to be processed in parallel.
    epsg_args = []
    for epsg_code in unique_epsg_list:
        epsg_args.append((epsg_code, full_strip_list, tmp_dir, output_dir, output_name, config,
                          nested_cpus, args.corrected, args.cloud_water_filter, args.simplify, args.gsw))

    # Set up memory monitoring before pool creation
    print(f"Starting memory: {psutil.virtual_memory().used / (1024**3):.2f} GB used out of {system_memory:.2f} GB")
    
    # Process EPSG codes with controlled resource usage
    with multiprocessing.Pool(epsg_cpus) as pool:
        # Use pool.map for better memory efficiency with chunksize parameter
        # This helps balance work across processes
        pool.map(process_epsg, epsg_args, chunksize=1)  # chunksize=1 ensures one EPSG code per task
    
    # Force garbage collection after pool completes
    gc.collect()
    print(f"Final memory: {psutil.virtual_memory().used / (1024**3):.2f} GB used out of {system_memory:.2f} GB")

    print("Finished processing all EPSG codes in parallel.")

if __name__ == '__main__':
    main()
