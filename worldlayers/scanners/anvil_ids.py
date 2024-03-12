import numpy as np
from typing import Any, Tuple, Dict
from ..common import DimScanData, log, get_blockid_mapping, convert_to_new_bs_format
from .common import iter_world_chunks, CTR_DTYPE


def scan_world_dimension(
    save_path: str,
    dim_id: int,
    scan_limit: int,
    bounds: Tuple[int, int],
    **_: Dict[str, Any]
) -> DimScanData:
    """Scanner for versions >=1.2.1 <=1.12.2 (anvil format, before Flattening)"""
    from .accelerators import scan_v2_accel

    ID_LIM = 2**12
    STATE_LIM = 16 * ID_LIM
    
    blockstate_to_idx = get_blockid_mapping(save_path)
    max_map_id = max(blockstate_to_idx.values())
    log('Maximum ID is', max_map_id)
    if max_map_id >= ID_LIM:
        log('Expanding ID range')
        ID_LIM = max_map_id + 1
        STATE_LIM = ID_LIM * 16
    
    # Accelerator always gets 'Add' and 'Add2' arrays passed in if they are present, but has no bounds checks
    # Can there be situations where there is no FML mapping / it lies / world was converted weirdly?
    
    min_section = bounds[0] // 16
    max_section = (bounds[1] + 15) // 16
    height_limit = (max_section - min_section) * 16

    chunks_scanned = 0
    ZERO_ADD = np.zeros(16**3, np.uint8)  # Filler for missing 'Add' or 'Add2'
    block_counts = np.zeros((height_limit, STATE_LIM), CTR_DTYPE)

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
