from pathlib import Path
import uNBT as nbt
from typing import Callable, Tuple
from time import perf_counter as clock
from ..common import DimScanData, AIR_BLOCKS, log, crop_histogram, sum_blocks_selection
from .alpha import scan_world_dimension_alpha
from .region import scan_world_dimension_old
from .anvil_ids import scan_world_dimension
from .anvil_flat import scan_world_dimension_new


def determine_scan_function(save_path: str) -> Callable[..., DimScanData]:
    # Since 1.9 version is stored in [level.dat].Data.DataVersion
    # First version after Flattening: 17w47a (data version: 1451)
    level_data = nbt.read_nbt_file(str(Path(save_path) / 'level.dat'))['Data']
    if 'DataVersion' in level_data and level_data['DataVersion'].value >= 1451:
        return scan_world_dimension_new
    
    # Distinguish versions by presence of files
    # Newer versions get priority
    if any(Path(save_path).rglob('r.*.*.mca')):
        return scan_world_dimension
    elif any(Path(save_path).rglob('r.*.*.mcr')):
        return scan_world_dimension_old
    else:
        return scan_world_dimension_alpha


def create_scan(
    save_path: str,
    dim_id: int,
    scan_limit: int,
    bounds: Tuple[int, int]
) -> DimScanData:
    scan_func = determine_scan_function(save_path)
    log(f'Using scanner: {scan_func.__name__}')

    t_start = clock()
    
    scan_data = scan_func(
        save_path=save_path,
        dim_id=dim_id,
        scan_limit=scan_limit,
        bounds=bounds,
    )

    # Crop histogram since scanner function may return more data than needed
    try:
        crop_histogram(scan_data, bounds)
    except ValueError as err:
        log(err)  # Ignore errors to not lose data on large scans
    
    t_finish = clock()

    log(f'Scanned {scan_data.chunks_scanned} chunks in {t_finish - t_start:.3f} seconds')
    log(f'{scan_data.state_count} unique block states')

    scan_volume = scan_data.chunks_scanned * scan_data.height * 16**2
    nonair_count = scan_volume - sum_blocks_selection(scan_data, AIR_BLOCKS).sum()
    log(f'{nonair_count:,}/{scan_volume:,} non-air blocks (Y levels {bounds[0]} ~ {bounds[1]})')
    
    return scan_data
