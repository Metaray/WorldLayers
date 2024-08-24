import numpy as np
from typing import Any, Tuple, Dict
from ..common import DimScanData
from .common import iter_world_chunks, CTR_DTYPE


def scan_world_dimension_new(
    save_path: str,
    dim_id: int,
    scan_limit: int,
    bounds: Tuple[int, int],
    save_properties: bool = True,
    **_: Dict[str, Any]
) -> DimScanData:
    """Scanner for versions 1.13+ (anvil format, after Flattening)"""
    from .accelerators import scan_v13_accel

    # Data version notes:
    # 2529 - 20w17a (pre 1.16) - Blockstate bits no longer packed across 64 bit words
    # 2836 - 21w39a (pre 1.18) - Moved "BlockStates" and "Palette" inside new "block_states" tag
    # 2844 - 21w43a (pre 1.18) - Removed "Level" tag (all contained tags moved "lower"), tag naming changes

    # Chunk statuses after all terrain and features have been generated
    STATUSES_READY = {
        'decorated', 'lighted', 'mobs_spawned', 'finalized', 'postprocessed',  # 1.13
        'fullchunk',
        'features', 'light', 'spawn', 'heightmaps', 'full',  # 1.14+
        'minecraft:features', 'minecraft:initialize_light', 'minecraft:light', 'minecraft:spawn', 'minecraft:full',  # 1.18+ (?)
    }

    # Set air index to zero for convinience
    blockstate_to_idx = {'minecraft:air': 0}
    
    min_section = bounds[0] // 16
    max_section = (bounds[1] + 15) // 16
    height_limit = (max_section - min_section) * 16
    
    chunks_scanned = 0
    block_counts = np.zeros((height_limit, len(blockstate_to_idx)), CTR_DTYPE)
    ZERO_LAYER = np.zeros((height_limit, 1), CTR_DTYPE)
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
