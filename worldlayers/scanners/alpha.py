import numpy as np
from typing import Any, Dict
from ..common import DimScanData, load_old_blockid_mapping, convert_to_new_bs_format
from .common import iter_world_chunks, CTR_DTYPE


def scan_world_dimension_alpha(
    save_path: str,
    scan_limit: int,
    **_: Dict[str, Any]
) -> DimScanData:
    """Scanner for versions before beta 1.3 (infdev format)"""
    from .accelerators import scan_v0_accel

    # Hardcoded limits for old versions
    STATE_LIM, MAX_HEIGHT = 256 * 16, 128

    blockstate_to_idx = load_old_blockid_mapping()
    chunks_scanned = 0
    block_counts = np.zeros((MAX_HEIGHT, STATE_LIM), CTR_DTYPE)

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
