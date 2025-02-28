# CLAUDE.md - Guidelines for DEM Repository

## Build/Run Commands
- Run mosaic creation: `python Mosaic_Strips.py [--config <config_file>] [--horizontal] [--N_cpus <N>]`
- Run inundation computation: `python Compute_Inundation.py --input_file <dem> --grid_extents <lon_min lon_max lat_min lat_max> --coastline <file>`
- Coregister DEM to ground truth: `python Simple_Coregistration.py --raster <path> --csv <path> [--median]`
- Download global DEMs: `python Global_DEMs.py --product <srtm|aster|copernicus> --extents <lon_min lon_max lat_min lat_max>`

## Code Style Guidelines
- Imports: Group standard library, third-party, and local imports in separate blocks
- Type hints: Not consistently used but encouraged for new code
- Error handling: Use try/except blocks with specific exceptions
- Naming: 
  - Variables: snake_case
  - Constants: UPPER_CASE
  - Functions: snake_case
  - Parameters: Often prefixed with UPPER_CASE in config files
- Documentation: Use docstrings for functions and detailed comments for complex algorithms
- Parallel processing: Use multiprocessing Pool for CPU-intensive operations

## Project Structure
- Configuration via INI files
- C code in C_Code/ directory
- Dependencies in requirements.txt or environment YML files
- Command-line interfaces with argparse
