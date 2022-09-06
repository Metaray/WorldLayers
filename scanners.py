from pathlib import Path
import numpy as np
import uNBT as nbt
from typing import Iterable, Tuple
from time import perf_counter as clock
from common import *

_CTR_DTYPE = np.int64 # Histogram counter type


def iter_world_chunks(save_path: str, dim_id: int, world_type: str) -> Iterable[nbt.TagCompound]:
    if world_type == 'alpha':
        # Alpha format has same data as Region, but chunks are stored in individual files
        # Numbers are base36 encoded
        # 2 folder layers: first is X % 64, second is Z % 64
        # Chunk file name: c.X.Y.dat
        for file in Path(save_path).glob('*/*/c.*.*.dat'):
            yield nbt.read_nbt_file(str(file))
    
    else:
        for rfile_info in nbt.enumerate_world(save_path, world_type)[dim_id]:
            print('Scanning', rfile_info.path)
            region = nbt.Region.from_file(rfile_info.path)
            for chunk in region.iter_nonempty():
                yield chunk.nbt


def scan_world_dimension(
    save_path: str,
    dim_id: int,
    scan_limit: int,
    bounds: Tuple[int, int],
    **_
) -> DimScanData:
    """Scanner for versions >=1.2.1 <=1.12.2 (anvil format, before Flattening)"""
    from accelerators import scan_v2_accel

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
    
    min_section = bounds[0] // 16
    max_section = (bounds[1] + 15) // 16
    height_limit = (max_section - min_section) * 16

    chunks_scanned = 0
    ZERO_ADD = np.zeros(16**3, np.uint8)  # Filler for missing 'Add' or 'Add2'
    block_counts = np.zeros((height_limit, STATE_LIM), _CTR_DTYPE)

    for chunk_nbt in iter_world_chunks(save_path, dim_id, 'anvil'):
        level_data = chunk_nbt['Level']
        if not ('TerrainPopulated' in level_data and level_data['TerrainPopulated'].value):
            continue
        
        for section in level_data['Sections']:
            section_y = section['Y'].value
            if not (min_section <= section_y < max_section):
                continue
            
            # 16x16x16 arrays in YZX order
            scan_v2_accel(
                block_counts,
                STATE_LIM,
                section_y - min_section,
                np.array(section['Blocks'].value, np.uint8),
                np.array(section['Add'].value, np.uint8) if 'Add' in section else ZERO_ADD,
                np.array(section['Add2'].value, np.uint8) if 'Add2' in section else ZERO_ADD,
                np.array(section['Data'].value, np.uint8),
            )

        chunks_scanned += 1
        if chunks_scanned >= scan_limit:
            break
    
    # Recalculate air block counts (may be lower because of empty chunk sections)
    layer_volume = chunks_scanned * 16**2
    block_counts[:, :16] = 0
    block_counts[:, 0] = layer_volume - block_counts[:, 16:].sum(axis=1)

    block_counts, blockstate_to_idx, old_to_new = convert_to_new_bs_format(block_counts, blockstate_to_idx)

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=min_section * 16,
        old_to_new=old_to_new,
    )


def scan_world_dimension_new(
    save_path: str,
    dim_id: int,
    scan_limit: int,
    bounds: Tuple[int, int],
    save_properties: bool = True,
    **_
) -> DimScanData:
    """Scanner for versions 1.13+ (anvil format, after Flattening)"""
    from accelerators import scan_v13_accel

    # Data version notes:
    # 2529 - 20w17a (pre 1.16) - Blockstate bits no longer packed across 64 bit words
    # 2836 - 21w39a (pre 1.18) - Moved "BlockStates" and "Palette" inside new "block_states" tag
    # 2844 - 21w43a (pre 1.18) - Removed "Level" tag (all contained tags moved "lower"), tag naming changes

    # Chunk statuses after all terrain and features have been generated
    STATUSES_READY = {
        'decorated', 'lighted', 'mobs_spawned', 'finalized', 'postprocessed',  # 1.13
        'fullchunk',
        'features', 'light', 'spawn', 'heightmaps', 'full',  # 1.14+
    }

    # Set air index to zero for convinience
    blockstate_to_idx = {'minecraft:air': 0}
    
    min_section = bounds[0] // 16
    max_section = (bounds[1] + 15) // 16
    height_limit = (max_section - min_section) * 16
    
    chunks_scanned = 0
    block_counts = np.zeros((height_limit, len(blockstate_to_idx)), _CTR_DTYPE)
    ZERO_LAYER = np.zeros((height_limit, 1), _CTR_DTYPE)
    index_map = np.zeros(16**3, np.uint32)

    for chunk_nbt in iter_world_chunks(save_path, dim_id, 'anvil'):
        data_version = chunk_nbt['DataVersion'].value
        level_data = chunk_nbt['Level'] if data_version < 2844 else chunk_nbt
        if level_data['Status'].value not in STATUSES_READY:
            continue
        
        has_carry = data_version < 2529
        for section in level_data['Sections' if data_version < 2844 else 'sections']:
            section_y = section['Y'].value
            if not (min_section <= section_y < max_section):
                continue
            
            if data_version < 2836:
                if 'Palette' not in section:
                    # Somtimes Palette is not written (section is empty - only air?)
                    continue
                states = section['BlockStates'].value
                palette = section['Palette']
            else:
                if 'data' in section['block_states']:
                    states = section['block_states']['data'].value
                else:
                    states = None
                palette = section['block_states']['palette']
            
            for idx, block in enumerate(palette):
                name = block['Name'].value
                if save_properties and 'Properties' in block:
                    # This must correctly keep in sync with `serialize_blockstate()`
                    props = [(k, v.value) for k, v in block['Properties'].items()]
                    if props:
                        props.sort()
                        props = ','.join(f'{k}={v}' for k, v in props)
                        name = f'{name}[{props}]'
                
                try:
                    bid = blockstate_to_idx[name]
                except KeyError:
                    blockstate_to_idx[name] = bid = len(blockstate_to_idx)
                    block_counts = np.hstack((block_counts, ZERO_LAYER))
                index_map[idx] = bid
            
            if states is not None:
                # `states` is packed array of 16x16x16 area of blocks in YZX order
                scan_v13_accel(
                    block_counts,
                    block_counts.shape[1],
                    section_y - min_section,
                    np.array(states, np.int64),
                    index_map,
                    len(palette) - 1,
                    has_carry,
                )
            else:
                # Block states were omitted - section is filled with one block
                y_base = (section_y - min_section) * 16
                block_counts[y_base:y_base+16, index_map[0]] += 16**2

        chunks_scanned += 1
        if chunks_scanned >= scan_limit:
            break
    
    # Recalculate air block counts (may be lower because of empty chunk sections)
    layer_volume = chunks_scanned * 16**2
    block_counts[:, 0] = layer_volume - block_counts[:, 1:].sum(axis=1)

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=min_section * 16,
    )


def scan_world_dimension_old(
    save_path: str,
    dim_id: int,
    scan_limit: int,
    **_
) -> DimScanData:
    """Scanner for versions <1.2.1 (region format)"""
    from accelerators import scan_v0_accel

    # Hardcoded limits for old versions
    # TODO: old modpacks had ID extender mod (something analogous to 'Add' array in newer versions?)
    ID_LIM = 256
    STATE_LIM = ID_LIM * 16
    MAX_HEIGHT = 128

    blockstate_to_idx = load_old_blockid_mapping()
    chunks_scanned = 0
    block_counts = np.zeros((MAX_HEIGHT, STATE_LIM), _CTR_DTYPE)

    for chunk_nbt in iter_world_chunks(save_path, dim_id, 'region'):
        level_data = chunk_nbt['Level']
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
    
    block_counts, blockstate_to_idx, old_to_new = convert_to_new_bs_format(block_counts, blockstate_to_idx)

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=0,
        old_to_new=old_to_new,
    )


def scan_world_dimension_alpha(
    save_path: str,
    scan_limit: int,
    **_
) -> DimScanData:
    """Scanner for versions before beta 1.3 (infdev format)"""
    from accelerators import scan_v0_accel

    # Hardcoded limits for old versions
    STATE_LIM, MAX_HEIGHT = 256 * 16, 128

    blockstate_to_idx = load_old_blockid_mapping()
    chunks_scanned = 0
    block_counts = np.zeros((MAX_HEIGHT, STATE_LIM), _CTR_DTYPE)

    for chunk_nbt in iter_world_chunks(save_path, 0, 'alpha'):
        level_data = chunk_nbt['Level']
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
    
    block_counts, blockstate_to_idx, old_to_new = convert_to_new_bs_format(block_counts, blockstate_to_idx)

    return DimScanData(
        histogram=block_counts,
        chunks_scanned=chunks_scanned,
        blockstate_to_idx=blockstate_to_idx,
        base_y=0,
        old_to_new=old_to_new,
    )


def determine_scan_function(save_path: str):
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
    print(f'Using scanner: {scan_func.__name__}')

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
        print(err)  # Ignore errors to not lose data on large scans
    
    t_finish = clock()

    print(f'Scanned {scan_data.chunks_scanned} chunks in {t_finish - t_start:.3f} seconds')
    print(f'{scan_data.state_count} unique block states')

    scan_volume = scan_data.chunks_scanned * scan_data.height * 16**2
    nonair_count = scan_volume - sum_blocks_selection(scan_data, AIR_BLOCKS).sum()
    print(f'{nonair_count:,}/{scan_volume:,} non-air blocks (Y levels {bounds[0]} ~ {bounds[1]})')
    
    return scan_data
