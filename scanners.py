import numpy as np
import uNBT as nbt
from typing import Tuple
import traceback
import os
from time import perf_counter as clock
from common import *
from accelerators import *

CTR_DTYPE = np.int64 # Histogram counter type


def scan_world_dimension(save_path: str, dim_id: int, scan_limit: int, bounds: Tuple[int, int]) -> DimScanData:
    """Scanner for versions >=1.2.1 <=1.12.2"""

    ID_LIM = 2**12
    STATE_LIM = 16 * ID_LIM
    
    blockstate_to_idx = get_blockid_mapping(save_path)
    max_map_id = max(blockstate_to_idx.values())
    print('Maximum ID is', max_map_id)
    if max_map_id >= ID_LIM:
        print('Expanding ID range')
        ID_LIM = max_map_id + 1
        STATE_LIM = ID_LIM * 16
    
    # Accelerator always gets 'Add' and 'Add2' arrays passed in if they are present, but has no bounds checks
    # Can there be situations where there is no FML mapping / it lies / world was converted weirdly?
    
    # For now let's silently truncate invalid ranges
    mn_sy = max(bounds[0] // 16, 0)
    mx_sy = min((bounds[1] + 15) // 16, 16)
    height_limit = (mx_sy - mn_sy) * 16

    chunks_scanned = 0
    M_ZERO = np.zeros(16**3, np.uint8) # filler for missing 'Add' or 'Add2'
    block_counts = np.zeros((height_limit, STATE_LIM), CTR_DTYPE)
    for rfile_info in nbt.enumerate_world(save_path)[dim_id]:
        print('Scanning', rfile_info.path)
        region = nbt.Region.from_file(rfile_info.path)
        for chunk in region.iter_nonempty():
            level_data = chunk.nbt['Level']
            if not ('TerrainPopulated' in level_data and level_data['TerrainPopulated'].value):
                continue
            
            for section in level_data['Sections']:
                sy = section['Y'].value
                if not (mn_sy <= sy < mx_sy):
                    continue
                sy -= mn_sy # To make calculations below be 0-based
                
                # 16x16x16 arrays in YZX order
                scan_v2_accel(
                    block_counts,
                    STATE_LIM,
                    sy,
                    np.array(section['Blocks'].value, np.uint8),
                    np.array(section['Add'].value, np.uint8) if 'Add' in section else M_ZERO,
                    np.array(section['Add2'].value, np.uint8) if 'Add2' in section else M_ZERO,
                    np.array(section['Data'].value, np.uint8),
                )

            chunks_scanned += 1
            if chunks_scanned >= scan_limit:
                break
        if chunks_scanned >= scan_limit:
            break
    
    # Recalculate air block counts (may be lower because of empty chunk sections)
    layer_volume = chunks_scanned * 16**2
    block_counts[:,:16] = 0
    block_counts[:,0] = layer_volume - block_counts[:,16:].sum(axis=1)

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=mn_sy * 16
    )


M_OK_STATUSES = set(['light', 'spawn', 'heightmaps', 'full', 'fullchunk'])
def scan_world_dimension_new(save_path: str, dim_id: int, scan_limit: int, bounds: Tuple[int, int]) -> DimScanData:
    """Scanner for versions 1.13+"""

    # Set some constant states for convinience
    blockstate_to_idx = {
        'minecraft:air': 0,
        'minecraft:cave_air': 1,
        'minecraft:void_air': 2,
    }
    STATE_LIM = max(blockstate_to_idx.values()) + 1
    
    mn_sy = bounds[0] // 16
    mx_sy = (bounds[1] + 15) // 16
    height_limit = (mx_sy - mn_sy) * 16
    
    chunks_scanned = 0
    block_counts = np.zeros((height_limit, STATE_LIM), CTR_DTYPE)
    ZERO_LAYER = np.zeros((height_limit, 1), CTR_DTYPE)
    index_map = np.zeros(16**3, np.uint32)
    section_ids = np.zeros(16**3, np.uint16)

    for rfile_info in nbt.enumerate_world(save_path)[dim_id]:
        print('Scanning', rfile_info.path)
        region = nbt.Region.from_file(rfile_info.path)
        for chunk in region.iter_nonempty():
            has_carry = chunk.nbt['DataVersion'].value < 2534 # version <20w19a (pre 1.16)
            level_data = chunk.nbt['Level']
            if level_data['Status'].value not in M_OK_STATUSES:
                continue
            
            for section in level_data['Sections']:
                sy = section['Y'].value
                if not (mn_sy <= sy < mx_sy):
                    continue
                sy -= mn_sy # To make calculations below be 0-based
                
                if 'Palette' not in section: # or 'BlockStates' not in section:
                    continue
                
                for idx, block in enumerate(section['Palette']):
                    name = block['Name'].value
                    # if 'Properties' in block:
                    #     blockstate = (name, frozenset((k, v.value) for k, v in block['Properties'].items()))
                    # else:
                    #     blockstate = (name, frozenset())
                    try:
                        bid = blockstate_to_idx[name]
                    except KeyError:
                        blockstate_to_idx[name] = bid = len(blockstate_to_idx)
                        block_counts = np.hstack((block_counts, ZERO_LAYER))
                    index_map[idx] = bid
                # `idx` holds maximum index in palette
                
                # `BlockStates` is packed 16x16x16 array in YZX order
                scan_v13_accel(
                    block_counts,
                    block_counts.shape[1],
                    sy,
                    np.array(section['BlockStates'].value),
                    index_map,
                    idx,
                    has_carry
                )

            chunks_scanned += 1
            if chunks_scanned >= scan_limit:
                break
        if chunks_scanned >= scan_limit:
            break
    
    # Adapter to old system with metadata
    ID_LIM = len(blockstate_to_idx)
    STATE_LIM = ID_LIM * 16
    tmp = np.zeros((height_limit, STATE_LIM), CTR_DTYPE)
    tmp[:,::16] = block_counts
    block_counts = tmp
    
    # Recalculate air block counts (may be lower because of empty chunk sections)
    layer_volume = chunks_scanned * 16**2
    block_counts[:,:16] = 0
    block_counts[:,0] = layer_volume - block_counts[:,16:].sum(axis=1)

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=mn_sy * 16
    )


def scan_world_dimension_old(save_path: str, dim_id: int, scan_limit: int, bounds: Tuple[int, int]) -> DimScanData:
    """Scanner for versions <1.2.1"""

    # Hardcoded limits for old versions
    # TODO: old modpacks had ID extender mod (something analogous to 'Add' array in newer versions?)
    ID_LIM = 256
    STATE_LIM = ID_LIM * 16
    MAX_HEIGHT = 128

    blockstate_to_idx = load_old_blockid_mapping()
    
    chunks_scanned = 0
    block_counts = np.zeros((MAX_HEIGHT, STATE_LIM), CTR_DTYPE)
    for rfile_info in nbt.enumerate_world(save_path, 'region')[dim_id]:
        print('Scanning', rfile_info.path)
        region = nbt.Region.from_file(rfile_info.path)
        for chunk in region.iter_nonempty():
            level_data = chunk.nbt['Level']
            if not ('TerrainPopulated' in level_data and level_data['TerrainPopulated'].value):
                continue

            # 16x16x128 arrays in XZY order
            scan_v0_accel(
                block_counts,
                np.array(level_data['Blocks'].value, np.uint8),
                np.array(level_data['Data'].value, np.uint8),
            )

            chunks_scanned += 1
            if chunks_scanned >= scan_limit:
                break
        if chunks_scanned >= scan_limit:
            break
    
    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=0
    )


def create_scan(save_path: str, dim_id: int, scan_limit: int, bounds: Tuple[int, int]) -> DimScanData:
    # Determine scan function automatically
    scan_func = scan_world_dimension
    try:
        # Since 1.9 can use [level.dat].Data.DataVersion
        # First version after Flattening: 17w47a (data version: 1451)
        # 1.13 data version: 1519
        level_data = nbt.read_nbt_file(os.path.join(save_path, 'level.dat'))['Data']
        if 'DataVersion' in level_data and level_data['DataVersion'].value >= 1451:
            scan_func = scan_world_dimension_new
        else:
            # For Region just scan for .mcr files?
            for rfile in os.listdir(os.path.join(save_path, f'DIM{dim_id}' if dim_id else '', 'region')):
                if rfile.endswith('.mca'):
                    # Probably was converted. New version has priority.
                    break
            else:
                # If no .mca files assume it is a Region world

                # TODO: Add Alpha format support
                # Alpha format has same data as Region, but chunks are stored in individual files
                # Numbers are base36 encoded
                # 2 folder layers: first is X % 64, second is Z % 64
                # Chunk file name: c.X.Y.dat

                scan_func = scan_world_dimension_old
    except Exception as err:
        traceback.print_exc()
    print(f'Using scanner: {scan_func.__name__}')

    t_start = clock()
    
    scan_data = scan_func(
        save_path=save_path,
        dim_id=dim_id,
        scan_limit=scan_limit,
        bounds=bounds
    )
    try:
        # Crop histogram since scanner function may return more data than needed
        scan_data.crop_histogram(bounds)
    except ValueError as err:
        print(err) # Ignore errors to not lose data of large scans
    
    t_finish = clock()

    print(f'Scanned in {t_finish - t_start:.3f}s')
    print(f'Scanned {scan_data.chunks_scanned} chunks')
    scan_volume = scan_data.chunks_scanned * (scan_data.height * 16**2)
    nonair_count = np.sum(scan_data.histogram[:,16:]) # TODO: subtract air properly
    print(f'{nonair_count:,}/{scan_volume:,} non-air blocks (Y levels {bounds[0]} ~ {bounds[1]})')
    
    return scan_data
